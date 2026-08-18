# Stage 1 — Vision-Guided Motion Tokenizer (VGMT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)

Stage 1 of [Superman](../README.md). VGMT encodes a 16-frame 3D skeleton clip and its RGB frames into 136 discrete tokens from an 8192-entry codebook. These tokens are used by the Stage 2 MLLM for motion perception and generation.

## Installation

```bash
git clone https://github.com/BradleyWang0416/Superman-CVPR2026.git
cd Superman-CVPR2026/motion_tokenizer

conda create -n superman_vgmt python=3.10 -y
conda activate superman_vgmt

# Install the PyTorch build matching your CUDA version first.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

A CUDA-capable GPU is required. Our reference environment uses Python 3.10, PyTorch 2.7.1, and CUDA 12.6.

## Pretrained Weights

Place the pretrained weights at the following paths:

| Weight | Path |
| --- | --- |
| HRNet-W32 | `checkpoint/hrnet/pose_hrnet_w32_256x192.pth` |
| Superman motion tokenizer | `checkpoint/superman_motion_tokenizer/checkpoint_epoch_165_step_500000/model.safetensors` |

The HRNet checkpoint is required only for training from scratch. Pass the tokenizer checkpoint **directory**, rather than `model.safetensors`, to `RESUME_PTH`.

## Data Preparation

1. Download the following three annotation files from [Baidu Netdisk](https://pan.baidu.com/s/1-oPDoKGd67vW5-WAlYHHCg?pwd=ey9w) (extraction code: `ey9w`) and place them in `data/h36m/`:

   - `bboxes_xyxy.pkl`
   - `h36m_sh_conf_cam_source_final_wJ2dCpn.pkl`
   - `images_source.pkl`

2. Download the Human3.6M images from the [official Human3.6M website](http://vision.imar.ro/human3.6m/) and extract them into `data/h36m/images_fps50/`. The image files are not included in the Baidu Netdisk download and must be obtained from the official website.

After downloading the annotations and images, the data directory should contain:

```text
data/h36m/
├── h36m_sh_conf_cam_source_final_wJ2dCpn.pkl
├── images_source.pkl
├── bboxes_xyxy.pkl
├── images_fps50/
└── images_fps50_cropped_448x448/
```

3. Generate the 448 × 448 person crops in `data/h36m/images_fps50_cropped_448x448/`:

```bash
python tools/crop_images_448.py \
    --images-source data/h36m/images_source.pkl \
    --bboxes data/h36m/bboxes_xyxy.pkl \
    --splits train test \
    --workers 16
```

The annotation paths are configured in `config/config.yaml`. The three annotation files must remain index-aligned.

## Training

```bash
bash scripts/train.sh
```

The released model was trained for 500K iterations on 8 GPUs. Common options can be overridden with environment variables:

```bash
GPUS=0,1 BATCH_SIZE=16 EXP_NAME=my_run bash scripts/train.sh
```

To resume training:

```bash
RESUME_PTH=experiment/my_run/models/checkpoint_epoch_106_step_320000 \
bash scripts/train.sh
```

## Evaluation

```bash
bash scripts/test.sh
```

To evaluate another checkpoint:

```bash
GPUS=1 \
RESUME_PTH=experiment/my_run/models/checkpoint_epoch_165_step_500000 \
bash scripts/test.sh
```

The released checkpoint achieves **5.85 mm reconstruction MPJPE** on the Human3.6M test split.

## Citation

Please see the [top-level README](../README.md#citation).

## Acknowledgements

This stage builds on [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), [MotionBERT](https://github.com/Walter0807/MotionBERT), and [MTVCrafter](https://github.com/DINGYANB/MTVCrafter).

We also thank [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) for its vector-quantization implementation.
