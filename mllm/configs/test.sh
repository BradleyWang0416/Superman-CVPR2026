#!/bin/bash
# Full-test-set evaluation (8,292 samples) of the new-VGMT model.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Resolved from the environment so the scripts work in any checkout:
#   LLAMAFACTORY_ROOT     the LLaMA-Factory clone the overlay was copied into
#   MOTION_TOKENIZER_DIR  Superman-CVPR2026/motion_tokenizer (Stage 1)
#   VGMT_CHECKPOINT       the released Stage 1 checkpoint directory
#   SUPERMAN_DATA_ROOT    root that the relative paths in images_source.pkl resolve against
ROOT="${LLAMAFACTORY_ROOT:?set LLAMAFACTORY_ROOT to your LLaMA-Factory clone}"
: "${MOTION_TOKENIZER_DIR:?set MOTION_TOKENIZER_DIR to Superman-CVPR2026/motion_tokenizer}"
export MOTION_TOKENIZER_DIR
export VGMT_CHECKPOINT="${VGMT_CHECKPOINT:-$MOTION_TOKENIZER_DIR/checkpoint/superman_motion_tokenizer/checkpoint_epoch_165_step_500000}"
# Leaving this unset makes the dataset resolve image paths against ./data of the
# LLaMA-Factory root, which silently fails far downstream -- so default it here.
export SUPERMAN_DATA_ROOT="${SUPERMAN_DATA_ROOT:-$MOTION_TOKENIZER_DIR/data}"
CONFIG="${SCRIPT_DIR}/${CONFIG_NAME:-test_qwen3vl.yaml}"
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/test_$(date '+%Y%m%d_%H%M%S').log}"
mkdir -p "$(dirname "$LOG_FILE")"
cd "$ROOT"
echo "eval: $CONFIG on GPUs $GPUS ($NPROC proc)"
echo "log: $LOG_FILE"
DISABLE_VERSION_CHECK=1 ALLOW_EXTRA_ARGS=true \
PYTHONUNBUFFERED=1 \
PYTHONPATH="${SCRIPT_DIR}":${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun --master_port "${MASTER_PORT:-12399}" --nproc_per_node "${NPROC}" \
    train_qwen_llamafactory.py "$CONFIG" 2>&1 | tee "$LOG_FILE"
