# Stage 2 — Superman MLLM

Stage 2 of [Superman](../README.md) integrates the motion tokens produced by [VGMT](../motion_tokenizer/) into Qwen3-VL, unifying video-to-motion perception and text-to-motion generation.

> The Superman-specific code and configs are released as an overlay for LLaMA-Factory. Fine-tuned LoRA weights are not included in the current release.

## Installation

```bash
export SUPERMAN=/path/to/Superman-CVPR2026

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 10a446e

pip install -e . --no-deps
pip install -r "$SUPERMAN/mllm/requirements.txt"
cp -r "$SUPERMAN/mllm/llamafactory_overlay/." .

export LLAMAFACTORY_ROOT=$PWD
export MOTION_TOKENIZER_DIR=$SUPERMAN/motion_tokenizer
export SUPERMAN_DATA_ROOT=$MOTION_TOKENIZER_DIR/data
```

Stage 2 uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Download the model and update `model_name_or_path` in `configs/train_qwen3vl.yaml` and `configs/test_qwen3vl.yaml`.

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen3-VL-8B-Instruct
```

## Data Preparation

Prepare the Human3.6M data and Stage 1 checkpoint as described in the [motion tokenizer README](../motion_tokenizer/README.md). Then encode the train and test splits into motion tokens:

```bash
CFG=$SUPERMAN/mllm/configs

GEN_GPU=0 DATA_SPLIT=train DATA_STRIDE=8 \
SAVE_SUBDIR_RAW=joint3d_image_affined_448x448/f8s1d8 \
bash "$CFG/generate_new_train.sh"

GEN_GPU=1 DATA_SPLIT=test DATA_STRIDE=64 \
SAVE_SUBDIR_RAW=joint3d_image_affined_448x448/f8s1d64 \
bash "$CFG/generate_new_train.sh"
```

The default setup generates 194,680 training samples and 8,292 test samples under `$LLAMAFACTORY_ROOT/_llamafactory_skeleton_byBrad/data/`.

## Training

```bash
GPUS=0,1,2,3,4,5,6,7 bash "$SUPERMAN/mllm/configs/train.sh"
```

The released configuration uses LoRA rank 8 with MAFT and trains for 3 epochs on 8 GPUs.

## Evaluation

```bash
GPUS=0,1,2,3 bash "$SUPERMAN/mllm/configs/test.sh"
```

Use a new `tokenized_path` whenever the Stage 1 checkpoint changes; otherwise LLaMA-Factory may reuse motion tokens from an old codebook.

## Citation

Please see the [top-level README](../README.md#citation).
