#!/bin/bash
# Superman Stage 2 -- encode Human3.6M into motion tokens with the Stage 1 tokenizer.
#
# Runnable from anywhere; it cd's into the LLaMA-Factory clone itself.
#
# Resolved from the environment so the scripts work in any checkout:
#   LLAMAFACTORY_ROOT     the LLaMA-Factory clone the overlay was copied into
#   MOTION_TOKENIZER_DIR  Superman-CVPR2026/motion_tokenizer (Stage 1)
#   VGMT_CHECKPOINT       the released Stage 1 checkpoint directory
#   SUPERMAN_DATA_ROOT    root that the relative paths in images_source.pkl resolve against
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${LLAMAFACTORY_ROOT:?set LLAMAFACTORY_ROOT to your LLaMA-Factory clone}"
: "${MOTION_TOKENIZER_DIR:?set MOTION_TOKENIZER_DIR to Superman-CVPR2026/motion_tokenizer}"
export MOTION_TOKENIZER_DIR
export VGMT_CHECKPOINT="${VGMT_CHECKPOINT:-$MOTION_TOKENIZER_DIR/checkpoint/superman_motion_tokenizer/checkpoint_epoch_165_step_500000}"
# Leaving this unset makes the dataset resolve image paths against ./data of the
# LLaMA-Factory root, which silently fails far downstream -- so default it here.
export SUPERMAN_DATA_ROOT="${SUPERMAN_DATA_ROOT:-$MOTION_TOKENIZER_DIR/data}"

save_root=_llamafactory_skeleton_byBrad/data/joint_and_image
save_subdir_raw=${SAVE_SUBDIR_RAW:-joint3d_image_affined_448x448/f8s1d8}
save_subdir_vqvae=cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl0123_adaSmpl_hrnetPretrained/step_500000
# save_subdir_jsonl=Vid2Skel/BodypartAwareExplicit
save_subdir_jsonl=h36m/Vid2Skel


############################# dataset config #######################################################
data_split=${DATA_SPLIT:-train}
num_frame=8
sample_stride=1
data_stride=${DATA_STRIDE:-8}
return_extra="[['image']]"
# joint2d_cpn feeds the released model's deformable sampler.
# joint3d_cam / joint3d_cam_rootrel_meter / joint2d are not consumed downstream and their
# source keys are not part of the released annotation file, so they are not requested.
get_item_list="['joint3d_image','joint3d_image_normed','factor_2_5d','joint3d_image_scale','joint3d_image_transl','video_rgb','joint3d_image_affined','joint3d_image_affined_normed','joint3d_image_affined_scale','joint3d_image_affined_transl','slice_id','image_sources','sources','joint_2_5d_image','affine_trans','affine_trans_inv','joint2d_cpn']"
load_data_file="h36m/h36m_sh_conf_cam_source_final_wJ2dCpn.pkl"
load_image_source_file="h36m/images_source.pkl"
load_bbox_file="h36m/bboxes_xyxy.pkl"
load_text_source_file=''

normalize=anisotropic
filter_invalid_images=True
processed_image_shape="[448,448]"
backbone=hrnet_32


batch_size=32




# The generator writes under _llamafactory_skeleton_byBrad/data/, relative to the
# LLaMA-Factory root; vqvae_config.py is picked up from this directory via PYTHONPATH.
cd "$ROOT"

PYTHONPATH="${SCRIPT_DIR}":${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES=${GEN_GPU:-0} \
    python -u \
    _llamafactory_skeleton_byBrad/data_utils/generate_multimodal_data.py \
    --save_root ${save_root} \
    --save_subdir_raw ${save_subdir_raw} \
    --save_subdir_vqvae ${save_subdir_vqvae} \
    --save_subdir_jsonl ${save_subdir_jsonl} \
    \
    --data_split ${data_split} \
    --num_frames ${num_frame} \
    --sample_stride ${sample_stride} \
    --data_stride ${data_stride} \
    --return_extra ${return_extra} \
    --get_item_list ${get_item_list} \
    --load_data_file "${load_data_file}" \
    --load_image_source_file "${load_image_source_file}" \
    --load_bbox_file "${load_bbox_file}" \
    --load_text_source_file "${load_text_source_file}" \
    --normalize ${normalize} \
    --filter_invalid_images ${filter_invalid_images} \
    --processed_image_shape ${processed_image_shape} \
    \
    --batch_size ${batch_size} \
    --backbone ${backbone} \
    --if_resample True
