#!/bin/bash
# Superman Stage 2 -- Qwen3-VL-8B SFT on the retrained VGMT's tokens.
#
#   bash train.sh                 # 8 GPUs
#   GPUS=0,1,2,3 bash train.sh    # override
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
CONFIG="${SCRIPT_DIR}/train_qwen3vl.yaml"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
PORT=${MASTER_PORT:-12345}

# The trainer is launched from the LLaMA-Factory root; vqvae_config.py is picked
# up from this directory through PYTHONPATH.
cd "$ROOT"

echo "=============================================================="
echo " Superman Stage 2 -- Qwen3-VL-8B SFT (new VGMT)"
echo "   config : $CONFIG"
echo "   GPUs   : $GPUS  (${NPROC} process(es))"
echo "=============================================================="

DISABLE_VERSION_CHECK=1 \
ALLOW_EXTRA_ARGS=true \
PYTHONPATH="${SCRIPT_DIR}":${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun --master_port "${PORT}" --nproc_per_node "${NPROC}" \
    train_qwen_llamafactory.py \
    "$CONFIG"
