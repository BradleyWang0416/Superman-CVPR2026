from typing import Any, Optional, Union
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import constant_, xavier_uniform_

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast
try:
    from transformers.utils import is_torchdynamo_compiling
except ImportError:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import is_torchdynamo_compiling
from transformers.utils import can_return_tuple, auto_docstring
from transformers.loss.loss_utils import fixed_cross_entropy

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

logger = logging.get_logger(__name__)


class MultiScaleDeformableKeypointSampler(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.offset_predictor = nn.Linear(d_model, n_heads * n_points * 2)
        self.weight_predictor = nn.Linear(d_model, n_heads * n_points)

        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset_predictor.weight.data, 0.)

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1

        with torch.no_grad():
            self.offset_predictor.bias = nn.Parameter(0.01 * grid_init.view(-1))

        constant_(self.weight_predictor.weight.data, 0.)
        constant_(self.weight_predictor.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, video_features, reference_points):
        B_T, J, _ = reference_points.shape

        initial_queries = F.grid_sample(
            video_features, reference_points.unsqueeze(2),
            align_corners=False, mode='bilinear'
        ).squeeze(-1).permute(0, 2, 1) # -> [B*T, J, C]

        offsets = self.offset_predictor(initial_queries).view(B_T, J, self.n_heads, self.n_points, 2)

        weights = self.weight_predictor(initial_queries).view(B_T, J, self.n_heads * self.n_points)
        weights = torch.softmax(weights, -1).view(B_T, J, self.n_heads, self.n_points, 1)

        final_grid = (reference_points.view(B_T, J, 1, 1, 2) + offsets).clamp(-1, 1)

        sampled_features = F.grid_sample(
            video_features,
            final_grid.view(B_T, J * self.n_heads * self.n_points, 1, 2),
            align_corners=False,
            mode='bilinear'
        ).view(B_T, self.d_model, J, self.n_heads, self.n_points).permute(0, 2, 3, 4, 1)

        output_features = (sampled_features * weights).sum(dim=(2, 3))

        return self.output_proj(output_features)


class Qwen3VLForConditionalGenerationWithSkeleton(Qwen3VLForConditionalGeneration):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)

        hidden_size = config.vision_config.hidden_size
        self.model.deformable_sampler = MultiScaleDeformableKeypointSampler(
            d_model=hidden_size,
            n_heads=8,
            n_points=4,
        )
        self.model.pose_video_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )
        self.model.pose_video_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.model.norm_q = nn.LayerNorm(hidden_size)
        self.model.norm_kv = nn.LayerNorm(hidden_size)
        self.model.norm_ffn = nn.LayerNorm(hidden_size)

        self.model.forward = types.MethodType(custom_qwen3_vl_model_forward_SkelCrsAttn, self.model)
        self.model.language_model.forward = types.MethodType(custom_qwen3_vltextmodel_forward, self.model.language_model)

    def loss_function(
            self,
            logits,
            labels,
            vocab_size: int,
            num_items_in_batch: Optional[torch.Tensor] = None,
            ignore_index: int = -100,
            shift_labels: Optional[torch.Tensor] = None,
            **kwargs,
            ) -> torch.Tensor:
        logits = logits.float()

        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)
        loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]):
        try:
            return GenerationMixin._validate_model_kwargs(self, model_kwargs)
        except ValueError as valueerror:
            return


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:


        skeleton_data_dict = None
        if pixel_values_videos is not None:
            skeleton_data_keys = kwargs.pop('skeleton_data_keys').item().split(',')
            skeleton_data_dict = {key: kwargs.pop(key) for key in skeleton_data_keys}

            affine_trans = skeleton_data_dict['affine_trans']
            vqvae_data_key = skeleton_data_dict['vqvae_data_key'].item()
            norm_scale = skeleton_data_dict[vqvae_data_key.replace('_normed', '_scale')].unsqueeze(-2)
            norm_offset = skeleton_data_dict[vqvae_data_key.replace('_normed', '_transl')].unsqueeze(-2)
            joint2d_cpn = skeleton_data_dict['joint2d_cpn']
            joint2d_cpn_xy1 = torch.cat(
                [joint2d_cpn, joint2d_cpn.new_ones(joint2d_cpn[..., :1].shape)],
                dim=-1,
            )
            joint2d_cpn_affined = torch.einsum('btij,btkj->btik', joint2d_cpn_xy1, affine_trans)
            skeleton_data_dict['joint2d_cpn_affined_normed'] = (
                joint2d_cpn_affined / norm_scale[..., :2] - norm_offset[..., :2]
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            skeleton_data_dict=skeleton_data_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)


        final_output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

        return final_output


def custom_qwen3_vl_model_forward_SkelCrsAttn(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    skeleton_data_dict = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)

        hidden_states = self.visual.patch_embed(pixel_values_videos)

        pos_embeds = self.visual.fast_pos_embed_interpolate(video_grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.visual.rot_pos_emb(video_grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(video_grid_thw[:, 1] * video_grid_thw[:, 2], video_grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=video_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                deepstack_feature = self.visual.deepstack_merger_list[self.visual.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        C = hidden_states.shape[-1]

        joint2d_coords = skeleton_data_dict['joint2d_cpn_affined_normed']

        B = video_grid_thw.shape[0]
        T_grid, H_grid, W_grid = video_grid_thw[0].tolist()

        vit_features_grid = hidden_states.reshape(B, T_grid, H_grid, W_grid, C)

        T_skel, J = joint2d_coords.shape[1:3]
        aligned_joint2d_coords = None
        if T_skel == 2 * T_grid:
            aligned_joint2d_coords = (joint2d_coords[:, 0::2, :, :] + joint2d_coords[:, 1::2, :, :]) / 2.0
        else:
             aligned_joint2d_coords = joint2d_coords

        vit_features_grid_for_sampler = vit_features_grid.view(B * T_grid, H_grid, W_grid, C).permute(0, 3, 1, 2).contiguous()

        reference_points = aligned_joint2d_coords.view(B * T_grid, -1, 2)

        skeleton_visual_features = self.deformable_sampler(
            vit_features_grid_for_sampler,
            reference_points
        ).view(B, -1, C) # [B, T*J, C]

        q_features = vit_features_grid.view(B, -1, C) # [B, TotalTokensPerVideo, C]

        q = self.norm_q(q_features)
        k = self.norm_kv(skeleton_visual_features)
        v = k

        attended_info, _ = self.pose_video_cross_attention(q, key=k, value=v)

        fused_embeds = q_features + attended_info

        processed_embeds = self.pose_video_ffn(self.norm_ffn(fused_embeds))

        enhanced_vit_features = fused_embeds + processed_embeds

        hidden_states = enhanced_vit_features.view(-1, C)

        hidden_states = self.visual.merger(hidden_states)

        video_embeds = hidden_states # Shape: [TotalMergedTokens, C_merged]
        deepstack_video_embeds = deepstack_feature_lists

        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def custom_qwen3_vltextmodel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
