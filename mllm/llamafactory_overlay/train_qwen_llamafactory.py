#%%
import os
import sys
from pathlib import Path
import torch
import transformers
import json
from time import time
import joblib
import copy
#%%
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
workspace_root = project_root.parent.parent

sys.path.append(str(workspace_root / "src"))
sys.path.append(str(project_root))

from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.extras.constants import IGNORE_INDEX

if transformers.__version__ == '5.4.0':
    from llamafactory.hparams.model_args import ModelArguments
    del ModelArguments.__dataclass_fields__["use_cache"]

# Qwen imports
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.extras_byBrad.modeling_qwen3_vl_byBrad import Qwen3VLForConditionalGenerationWithSkeleton
from peft import LoraConfig, get_peft_model, PeftModel


def load_mocap_assets(dataset_config):
    """Load the fixed Superman dataset bundle referenced by a YAML dataset entry."""
    dataset_file = Path(dataset_config["file_name"])
    split = dataset_file.name.removesuffix("_data.jsonl")
    if split not in {"train", "test"}:
        raise ValueError(f"Expected train_data.jsonl or test_data.jsonl, got {dataset_file}.")

    dataset_dir = dataset_file.parent
    with (dataset_dir / f"{split}_dataset_args.json").open() as file:
        dataset_args = json.load(file)

    dataset_args["get_item_list"] = [
        item for item in dataset_args["get_item_list"] if item != "video_rgb"
    ]
    print(f"\nLoading {split} mocap dataset...", end=" ")
    start_time = time()
    mocap_dataset = Multimodal_Mocap_Dataset(**dataset_args)
    print(f"Took {time() - start_time:.1f} seconds\n")

    vqvae_output = joblib.load(dataset_dir / f"{split}_vqvae_output.pkl")
    with (dataset_dir / f"{split}_prompt_config.json").open() as file:
        prompt_config = json.load(file)

    expected_prompt_config = {
        "task": "Vid2Skel",
        "prompt_type": "BodypartAwareExplicit_text",
        "get_skel_str_func": {
            "name": "get_skeleton_token_str_wTextualBodyPart_SplitByFrame",
            "input": "skeleton_indices",
            "extra_args": {},
        },
    }
    if prompt_config != expected_prompt_config:
        raise ValueError(
            f"Dataset prompt config does not match the released Superman format: {prompt_config}"
        )

    return mocap_dataset, vqvae_output, prompt_config


def configure_mm_plugin(template, dataset_config):
    mocap_dataset, vqvae_output, prompt_config = load_mocap_assets(dataset_config)
    plugin = template.mm_plugin
    plugin.mocap_dataset = mocap_dataset
    plugin.vqvae_output = vqvae_output
    return mocap_dataset


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train():
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.dataset_eval_range is not None:
        assert data_args.max_samples is None, "Setting max_samples could mess up the dataset_eval_range. Don't do that."

    print("Loading tokenizer and template...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    eval_template = copy.deepcopy(template)

    expected_subsets = ["placeholder", "eval_dataset"]
    if data_args.dataset != expected_subsets or set(data_args.dataset_dir) != set(expected_subsets):
        raise ValueError(
            "The released Superman configs require dataset: [placeholder, eval_dataset] "
            "with matching dataset_dir entries."
        )

    configure_mm_plugin(template, data_args.dataset_dir["placeholder"])
    mocap_dataset_eval = configure_mm_plugin(
        eval_template, data_args.dataset_dir["eval_dataset"]
    )


    print("Loading dataset...")
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        return_dict=True,
        eval_template=eval_template,
        **tokenizer_module
    )


    print("Loading model...")
    model = Qwen3VLForConditionalGenerationWithSkeleton.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.resize_vocab:
        model.resize_token_embeddings(len(tokenizer))

    for token_name in ["pad_token_id", "bos_token_id", "eos_token_id"]:
        if hasattr(tokenizer, token_name):
            token_id = getattr(tokenizer, token_name)
            setattr(model.config, token_name, token_id)
            if hasattr(model, "generation_config") and model.generation_config:
                 setattr(model.generation_config, token_name, token_id)

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
             model.enable_input_require_grads()
        else:
             def make_inputs_require_grad(module, input, output):
                 output.requires_grad_(True)
             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if finetuning_args.finetuning_type == 'lora':
        if model_args.adapter_name_or_path:
            print(f"Loading LoRA adapter from {model_args.adapter_name_or_path}...")
            for adapter_path in model_args.adapter_name_or_path:
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=training_args.do_train)
        else:
            print("Applying LoRA...")
            lora_config = LoraConfig(
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                target_modules=finetuning_args.lora_target,
                lora_dropout=finetuning_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=finetuning_args.additional_target,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    print("Creating data collator...")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model.dtype,
        **tokenizer_module,
    )


    if training_args.do_predict:
        import math
        from datetime import datetime

        import numpy as np
        from llamafactory.extras.misc import get_logits_processor
        from _llamafactory_skeleton_byBrad.data_utils.convert_skel_token import (
            parse_skeleton_token_str_wTextualBodyPart_SplitByFrame,
        )

        gen_kwargs = generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + getattr(
            tokenizer,
            "additional_special_tokens_ids",
            getattr(tokenizer, "all_special_ids", []),
        )
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()

        from superman_stage1 import VisionGuidedMotionTokenizer, load_motion_tokenizer_weights
        skeleton_processor = VisionGuidedMotionTokenizer(model_args.vqvae_config.vqvae_config.encoder,
                                        model_args.vqvae_config.vqvae_config.decoder,
                                        model_args.vqvae_config.vqvae_config.vq,
                                        vision_config=model_args.vqvae_config.vision_config,
                                        joint_data_type=model_args.vqvae_config.vqvae_config.joint_data_type,
        )
        vqvae_checkpoint = model_args.vqvae_config.vqvae_config.resume_path
        load_motion_tokenizer_weights(skeleton_processor, vqvae_checkpoint)
        skeleton_processor.eval()
        for param in skeleton_processor.parameters():
            param.requires_grad = False
        skeleton_processor = skeleton_processor.cuda()

        eval_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    def do_custom_evaluation(trainer_instance):
        is_main_process = training_args.process_index == 0
        if is_main_process:
            eval_start_time = time()

        eval_subbatch_size = data_args.eval_subbatch_size
        eval_data_size = len(dataset_module.get("eval_dataset", []))
        if eval_data_size == 0:
            print("No eval dataset found!")
            return

        batch_size = eval_subbatch_size or eval_data_size
        num_batches = math.ceil(eval_data_size / batch_size)
        all_mpjpe = []
        all_mpjpe_final = []

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        epoch = trainer_instance.state.epoch if trainer_instance.state.epoch is not None else "eval"
        save_data_path = Path(sys.argv[1]).parent / f"SAVE_EVAL_DATA_{eval_time_str}" / f"epoch_{epoch}"
        raw_output_dir = save_data_path / "raw_outputs"
        motion_mm_dir = save_data_path / "mm_beforeRtRel"
        if is_main_process:
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            motion_mm_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, eval_data_size)
            if is_main_process:
                print(f"Evaluating batch {batch_idx+1}/{num_batches}: samples {start} -- {end} / {eval_data_size}")
            try:
                from datasets import Dataset
                if eval_subbatch_size is not None:
                    eval_dataset = Dataset.from_dict(dataset_module["eval_dataset"][start:end])
                else:
                    eval_dataset = dataset_module["eval_dataset"]
                current_eval_dataset_len = len(eval_dataset)

                predict_results = trainer_instance.predict(eval_dataset, metric_key_prefix="predict", **gen_kwargs)

                if not is_main_process:
                    continue

                decoded_outputs = []
                for sample_id in range(predict_results.predictions.shape[0]):
                    if sample_id >= current_eval_dataset_len:
                        continue
                    sample_prediction = predict_results.predictions[sample_id]
                    sample_label = predict_results.label_ids[sample_id]

                    text_prediction = tokenizer.decode(sample_prediction[sample_prediction != -100], skip_special_tokens=False)
                    text_prediction = text_prediction.replace('<|endoftext|>', '')
                    text_label = tokenizer.decode(sample_label[sample_label != -100], skip_special_tokens=False)
                    decoded_outputs.append((sample_id, text_prediction, text_label))

                raw_output_dict = [
                    {"pred": text_prediction, "label": text_label}
                    for _, text_prediction, text_label in decoded_outputs
                ]
                with (raw_output_dir / f"{start:04d}-{end:04d}.json").open("w") as file:
                    json.dump(raw_output_dict, file, indent=4)

                motion_labels = []
                motion_predictions = []
                successful_sample_ids = []
                for sample_id, text_prediction, text_label in decoded_outputs:

                    motion_id_label = parse_skeleton_token_str_wTextualBodyPart_SplitByFrame(text_label)
                    motion_id_label = np.array(motion_id_label)
                    motion_id_label = torch.from_numpy(motion_id_label).long().unsqueeze(0).cuda()
                    motion_label = skeleton_processor.decode(motion_id_label).squeeze(0).cpu().numpy()

                    try:
                        motion_id_prediction = parse_skeleton_token_str_wTextualBodyPart_SplitByFrame(
                            text_prediction
                        )
                        motion_id_prediction = np.array(motion_id_prediction)
                        motion_id_prediction = torch.from_numpy(motion_id_prediction).long().unsqueeze(0).cuda()
                        motion_prediction = skeleton_processor.decode(
                            motion_id_prediction
                        ).squeeze(0).cpu().numpy()

                        motion_labels.append(motion_label)
                        motion_predictions.append(motion_prediction)
                        successful_sample_ids.append(sample_id)
                    except Exception as e:
                        print(f"[SampleID {sample_id + start}] {e}. Skipping this sample")
                        continue
                if not motion_labels:
                    print(f"[Batch {batch_idx+1}] No valid predictions in this batch, skipping metric computation.")
                    continue

                motion_labels = np.stack(motion_labels, axis=0)
                try:
                    motion_predictions = np.stack(motion_predictions, axis=0)
                except ValueError:
                    valid_shape_indices = [
                        index
                        for index, prediction in enumerate(motion_predictions)
                        if prediction.shape == motion_labels[0].shape
                    ]
                    if not valid_shape_indices:
                        print(f"[Batch {batch_idx+1}] Predictions have incompatible shapes, skipping.")
                        continue
                    motion_predictions = np.stack(
                        [motion_predictions[index] for index in valid_shape_indices], axis=0
                    )
                    motion_labels = motion_labels[valid_shape_indices]
                    successful_sample_ids = [
                        successful_sample_ids[index] for index in valid_shape_indices
                    ]

                all_mpjpe.append(
                    np.linalg.norm(
                        (motion_labels - motion_labels[..., 0:1, :])
                        - (motion_predictions - motion_predictions[..., 0:1, :]),
                        axis=-1,
                    ).mean((-2, -1))
                    * 1000
                )

                skel_info_dict_list = sum(dataset_module["eval_dataset"]['skeletons'][start:end], [])
                data_key = model_args.vqvae_config.vqvae_config.joint_data_type
                scale_key = data_key.replace("_normed", "_scale")
                translation_key = data_key.replace("_normed", "_transl")
                coordinate_keys = [scale_key, translation_key, "affine_trans_inv", "factor_2_5d"]
                coordinate_data = {key: [] for key in coordinate_keys}
                for local_sample_id in successful_sample_ids:
                    sample_id = skel_info_dict_list[local_sample_id]["sample_id"]
                    sample = mocap_dataset_eval[sample_id]
                    for key in coordinate_keys:
                        coordinate_data[key].append(sample[key])
                coordinate_data = {
                    key: np.stack(values, axis=0) for key, values in coordinate_data.items()
                }

                skeleton_scale = coordinate_data[scale_key][..., None, :]
                skeleton_offset = coordinate_data[translation_key][..., None, :]
                trans_inv = coordinate_data["affine_trans_inv"]
                factor_2_5d = coordinate_data["factor_2_5d"][..., None, None]
                motion_predictions_affined = (motion_predictions + skeleton_offset) * skeleton_scale
                motion_labels_affined = (motion_labels + skeleton_offset) * skeleton_scale

                motion_predictions_xy1 = np.concatenate(
                    [motion_predictions_affined[..., :2], np.ones_like(motion_predictions_affined[..., :1])],
                    axis=-1,
                )
                motion_labels_xy1 = np.concatenate(
                    [motion_labels_affined[..., :2], np.ones_like(motion_labels_affined[..., :1])],
                    axis=-1,
                )
                motion_predictions_xy = np.einsum('btij,btkj->btki', trans_inv, motion_predictions_xy1)
                motion_labels_xy = np.einsum('btij,btkj->btki', trans_inv, motion_labels_xy1)
                motion_predictions_mm = np.concatenate(
                    [motion_predictions_xy, motion_predictions_affined[..., 2:]], axis=-1
                ) * factor_2_5d
                motion_labels_mm = np.concatenate(
                    [motion_labels_xy, motion_labels_affined[..., 2:]], axis=-1
                ) * factor_2_5d

                mpjpe_mm = np.linalg.norm(motion_labels_mm - motion_predictions_mm, axis=-1).mean()
                np.save(
                    motion_mm_dir / f"{start:04d}-{end:04d}_pred_{mpjpe_mm:.1f}.npy",
                    motion_predictions_mm,
                )
                np.save(
                    motion_mm_dir / f"{start:04d}-{end:04d}_label_{mpjpe_mm:.1f}.npy",
                    motion_labels_mm,
                )

                motion_predictions_rootrel = motion_predictions_mm - motion_predictions_mm[..., 0:1, :]
                motion_labels_rootrel = motion_labels_mm - motion_labels_mm[..., 0:1, :]
                all_mpjpe_final.append(
                    np.linalg.norm(motion_labels_rootrel - motion_predictions_rootrel, axis=-1).mean((-2, -1))
                )
            except Exception as e:
                print(f"[Batch {batch_idx+1}] Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        if is_main_process:
            if all_mpjpe_final:
                all_mpjpe = np.concatenate(all_mpjpe, axis=0)
                final_mpjpe = np.concatenate(all_mpjpe_final, axis=0).mean()
                print(
                    f"[All] avg mpjpe_all: ({all_mpjpe.shape} samples) {all_mpjpe.mean()}"
                    f"\tFinal MPJPE (mm, root-rel): {final_mpjpe}"
                )
            else:
                print("No valid predictions in all batches.")
            print(f"Evaluation completed in {time() - eval_start_time:.2f} seconds.")

        tokenizer.padding_side = original_padding_side


    print("Initializing Trainer...")
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_module['train_dataset'],
        eval_dataset=dataset_module.get('eval_dataset') if training_args.do_predict else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        processor=processor,
        finetuning_args=finetuning_args,
    )

    if training_args.do_train:
        print("Starting training...")
        if list(Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        print("Saving model...")
        trainer.save_state()
        model.config.use_cache = True
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

        if processor:
            processor.save_pretrained(training_args.output_dir)
        else:
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        do_custom_evaluation(trainer)


if __name__ == "__main__":
    train()
