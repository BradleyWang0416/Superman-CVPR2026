# Superman: Unifying Skeleton and Vision for Human Motion Perception and Generation

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-1b6ac9.svg)](https://cvpr.thecvf.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Superman: Unifying Skeleton and Vision for Human Motion Perception and Generation"** (CVPR 2026).

Superman represents human motion as discrete tokens, enabling a multimodal LLM to handle motion perception and generation in a unified next-token prediction framework.

## Overview

Superman contains two stages:

| Stage | Directory | Description |
| --- | --- | --- |
| Vision-Guided Motion Tokenizer (VGMT) | [`motion_tokenizer/`](motion_tokenizer/) | Encodes a 16-frame 3D skeleton clip and its RGB frames into 136 discrete motion tokens. |
| Superman MLLM | [`mllm/`](mllm/) | Integrates motion tokens into Qwen3-VL for motion perception and generation. |

Please follow the README of each stage for installation, data preparation, training, and evaluation:

- [Stage 1: Vision-Guided Motion Tokenizer](motion_tokenizer/README.md)
- [Stage 2: Superman MLLM](mllm/README.md)

## Data

Both stages use [Human3.6M](http://vision.imar.ro/human3.6m/). Please accept the dataset license and download the images from the official website. We provide only derived annotations.

## Citation

```bibtex
@inproceedings{superman2026,
  title     = {Superman: Unifying Skeleton and Vision for Human Motion Perception and Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), [MotionBERT](https://github.com/Walter0807/MotionBERT), and [MTVCrafter](https://github.com/DINGYANB/MTVCrafter).

We also thank [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for their excellent work.
