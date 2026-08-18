#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Superman (CVPR 2026) -- train the Vision-Guided Motion Tokenizer.
#
#   bash motion_tokenizer/scripts/train.sh
#   GPUS=0,1 BATCH_SIZE=16 EXP_NAME=my_run bash motion_tokenizer/scripts/train.sh
#   RESUME_PTH=experiment/my_run/models/checkpoint_epoch_106_step_320000 \
#       bash motion_tokenizer/scripts/train.sh
#
# Every variable below can be overridden from the environment. Defaults
# reproduce the released model (Human3.6M, 8 GPUs x batch 4 = effective batch
# 32, 500K iterations, ~63 h on 8 x RTX 4090).
#
# The script cd's into motion_tokenizer/ itself, so it can be launched from
# anywhere; data/ and checkpoint/ are resolved relative to that directory.
# ---------------------------------------------------------------------------
set -euo pipefail

STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STAGE_ROOT"

# ------------------------------- launcher ----------------------------------
GPUS=${GPUS:-0,1,2,3,4,5,6,7}           # comma separated device ids
MAIN_PORT=${MAIN_PORT:-29240}           # accelerate rendezvous port

# ------------------------------ experiment ---------------------------------
CONFIG=${CONFIG:-config/config.yaml}
EXP_NAME=${EXP_NAME:-superman_motion_tokenizer}
PROJECT_DIR=${PROJECT_DIR:-experiment/${EXP_NAME}}
RESUME_PTH=${RESUME_PTH:-}              # checkpoint DIRECTORY, e.g. .../checkpoint_epoch_106_step_320000

# --------------------------------- data ------------------------------------
NUM_FRAMES=${NUM_FRAMES:-16}            # frames per clip
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}       # frame subsampling inside a video
DATA_STRIDE=${DATA_STRIDE:-16}          # stride between clips (== NUM_FRAMES -> no overlap)
BATCH_SIZE=${BATCH_SIZE:-4}             # PER GPU; effective batch = BATCH_SIZE x #GPUS

# -------------------------------- codebook ---------------------------------
NUM_CODE=${NUM_CODE:-8192}              # codebook entries
CODE_DIM=${CODE_DIM:-2048}              # code dimension (split 50/50 vision | skeleton)

# ------------------------------- optimizer ---------------------------------
LEARNING_RATE=${LEARNING_RATE:-2e-4}
TOTAL_ITER=${TOTAL_ITER:-500000}
WARM_UP_ITER=${WARM_UP_ITER:-5000}
COMMIT_RATIO=${COMMIT_RATIO:-0.5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
PRINT_ITER=${PRINT_ITER:-200}
SEED=${SEED:-6666}

# ---------------------------------------------------------------------------
NUM_PROCESSES=$(awk -F',' '{print NF}' <<< "$GPUS")

echo "=============================================================="
echo " Superman motion tokenizer -- training"
echo "   stage root   : $STAGE_ROOT"
echo "   config       : $CONFIG"
echo "   output dir   : $PROJECT_DIR"
echo "   GPUs         : $GPUS  (${NUM_PROCESSES} process(es))"
echo "   batch size   : $BATCH_SIZE per GPU  ->  $((BATCH_SIZE * NUM_PROCESSES)) effective"
echo "   codebook     : ${NUM_CODE} x ${CODE_DIM}"
echo "   clip         : ${NUM_FRAMES} frames (sample_stride=${SAMPLE_STRIDE}, data_stride=${DATA_STRIDE})"
echo "   iterations   : $TOTAL_ITER"
echo "   resume from  : ${RESUME_PTH:-<scratch>}"
echo "=============================================================="

mkdir -p "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=${GPUS} \
accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PORT}" \
    train.py \
    --config "${CONFIG}" \
    --project_dir "${PROJECT_DIR}" \
    --resume_pth "${RESUME_PTH}" \
    --num_frames "${NUM_FRAMES}" \
    --sample_stride "${SAMPLE_STRIDE}" \
    --data_stride "${DATA_STRIDE}" \
    --batch_size "${BATCH_SIZE}" \
    --nb_code "${NUM_CODE}" \
    --codebook_dim "${CODE_DIM}" \
    --learning_rate "${LEARNING_RATE}" \
    --total_iter "${TOTAL_ITER}" \
    --warm_up_iter "${WARM_UP_ITER}" \
    --commit_ratio "${COMMIT_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --save_interval "${SAVE_INTERVAL}" \
    --print_iter "${PRINT_ITER}" \
    --seed "${SEED}"
