"""Stage 1 tokenizer configuration, as consumed by the Stage 2 trainer.

Imported via PYTHONPATH by train.sh / test.sh / generate_data.sh. It takes the
architecture defaults from the Stage 1 checkout and pins the settings the released
tokenizer was trained with.

Two things are resolved from the environment:

    MOTION_TOKENIZER_DIR   the motion_tokenizer/ directory of a Superman-CVPR2026
                           checkout (see llamafactory_overlay/superman_stage1.py)
    VGMT_CHECKPOINT        the released tokenizer checkpoint directory
"""

import os
import os.path as osp

from easydict import EasyDict as edict

# Pulls in Stage 1's encoder / decoder / VQ / vision-backbone configs through the
# single integration point, so no sys.path juggling is needed here.
from superman_stage1 import (
    MOTION_TOKENIZER_DIR,
    vision_backbone_config as vision_config,
    vqvae_config,
)

# The released Stage 1 checkpoint (a directory containing model.safetensors).
DEFAULT_CHECKPOINT = osp.join(
    MOTION_TOKENIZER_DIR,
    "checkpoint", "superman_motion_tokenizer", "checkpoint_epoch_165_step_500000",
)
VGMT_CHECKPOINT = osp.abspath(os.environ.get("VGMT_CHECKPOINT", DEFAULT_CHECKPOINT))


vqvae_update_config = edict(
    nb_code=8192,
    codebook_dim=2048,
    is_train=False,
    downsample_time=[1, 2],
    frame_upsample_rate=[2.0, 1.0],
)

vision_update_config = edict(
    hrnet_output_level=[0, 1, 2, 3],
    vision_guidance_ratio=0.5,
    vision_guidance_fuse='ada_sample',
)

extra_config = edict(
    joint_data_type='joint3d_image_affined_normed',
    resume_path=VGMT_CHECKPOINT,
)


vqvae_config.encoder.out_channels = vqvae_update_config.codebook_dim
vqvae_config.decoder.in_channels = vqvae_update_config.codebook_dim
vqvae_config.vq.nb_code = vqvae_update_config.nb_code
vqvae_config.vq.code_dim = vqvae_update_config.codebook_dim
vqvae_config.vq.is_train = vqvae_update_config.is_train

vqvae_config.encoder.downsample_time = vqvae_update_config.downsample_time
vqvae_config.decoder.frame_upsample_rate = vqvae_update_config.frame_upsample_rate

assert not hasattr(vqvae_config, 'joint_data_type') and not hasattr(vqvae_config, 'resume_path')
setattr(vqvae_config, 'joint_data_type', extra_config.joint_data_type)
setattr(vqvae_config, 'resume_path', extra_config.resume_path)

# The vision-guidance design is fixed inside the model (motion_tokenizer/models/vgmt.py):
# HRNet levels [0,1,2,3], a 50/50 vision / skeleton code split, and deformable
# "ada_sample" fusion via VisualSkeletonAttention. vision_update_config above documents
# those values; there is nothing to override on the config object.
