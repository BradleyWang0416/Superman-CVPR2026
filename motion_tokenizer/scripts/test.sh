#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Superman (CVPR 2026) -- evaluate the Vision-Guided Motion Tokenizer.
#
#   bash motion_tokenizer/scripts/test.sh
#   GPUS=1 RESUME_PTH=experiment/my_run/models/checkpoint_epoch_165_step_500000 \
#       bash motion_tokenizer/scripts/test.sh
# ---------------------------------------------------------------------------
set -euo pipefail

STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STAGE_ROOT"

# ------------------------------- settings ----------------------------------
GPUS=${GPUS:-0}                         # single device id
CONFIG=${CONFIG:-config/config.yaml}
RESUME_PTH=${RESUME_PTH:-checkpoint/superman_motion_tokenizer/checkpoint_epoch_165_step_500000}
BATCH_SIZE=${BATCH_SIZE:-16}

NUM_FRAMES=${NUM_FRAMES:-16}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
DATA_STRIDE=${DATA_STRIDE:-16}

NUM_CODE=${NUM_CODE:-8192}
CODE_DIM=${CODE_DIM:-2048}

# ---------------------------------------------------------------------------
if [ ! -d "$RESUME_PTH" ]; then
    echo "Checkpoint directory not found: $RESUME_PTH" >&2
    echo "--resume_pth expects the accelerate state DIRECTORY (the one containing" >&2
    echo "model.safetensors), not the .safetensors file itself." >&2
    exit 1
fi

echo "=============================================================="
echo " Superman motion tokenizer -- evaluation"
echo "   config     : $CONFIG"
echo "   checkpoint : $RESUME_PTH"
echo "   GPU        : $GPUS"
echo "   codebook   : ${NUM_CODE} x ${CODE_DIM}"
echo "=============================================================="

CUDA_VISIBLE_DEVICES=${GPUS} \
python test.py \
    --config "${CONFIG}" \
    --resume_pth "${RESUME_PTH}" \
    --batch_size "${BATCH_SIZE}" \
    --num_frames "${NUM_FRAMES}" \
    --sample_stride "${SAMPLE_STRIDE}" \
    --data_stride "${DATA_STRIDE}" \
    --nb_code "${NUM_CODE}" \
    --codebook_dim "${CODE_DIM}"
