"""Generate the fixed Vid2Skel JSONL bundle used by Superman Stage 2."""

import argparse
import ast
import json
import os
import os.path as osp
from collections import defaultdict

import joblib
import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from llamafactory.extras.constants import PROMPT_PLACEHOLDER
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset
from superman_stage1 import VisionGuidedMotionTokenizer, load_motion_tokenizer_weights
from vqvae_config import vision_config, vqvae_config

from _llamafactory_skeleton_byBrad.data_utils.templates import TASK_TEMPLATE
from _llamafactory_skeleton_byBrad.data_utils.utils import data_prefetcher


TASK = "Vid2Skel"
PROMPT_TYPE = "BodypartAwareExplicit_text"
SKELETON_FORMAT = {
    "name": "get_skeleton_token_str_wTextualBodyPart_SplitByFrame",
    "input": "skeleton_indices",
    "extra_args": {},
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", default=osp.join("_llamafactory_skeleton_byBrad", "data"))
    parser.add_argument("--save_subdir_raw", required=True)
    parser.add_argument("--save_subdir_vqvae", required=True)
    parser.add_argument("--save_subdir_jsonl", required=True)

    parser.add_argument("--data_split", choices=["train", "test"], required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--sample_stride", type=int, required=True)
    parser.add_argument("--data_stride", type=int, required=True)
    parser.add_argument("--return_extra", required=True)
    parser.add_argument("--get_item_list", required=True)
    parser.add_argument("--load_data_file", required=True)
    parser.add_argument("--load_image_source_file", required=True)
    parser.add_argument("--load_bbox_file", required=True)
    parser.add_argument("--load_text_source_file", default="")
    parser.add_argument("--normalize", required=True)
    parser.add_argument("--filter_invalid_images", required=True)
    parser.add_argument("--processed_image_shape", required=True)
    parser.add_argument("--if_resample", default="True")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    args.return_extra = ast.literal_eval(args.return_extra)
    args.get_item_list = ast.literal_eval(args.get_item_list)
    args.processed_image_shape = ast.literal_eval(args.processed_image_shape)
    args.if_resample = ast.literal_eval(args.if_resample)
    args.filter_invalid_images = args.filter_invalid_images.lower() == "true"

    required_items = {
        vqvae_config.joint_data_type,
        "video_rgb",
        "slice_id",
        "image_sources",
        "joint2d_cpn",
        "affine_trans",
        "affine_trans_inv",
        "factor_2_5d",
    }
    missing_items = required_items.difference(args.get_item_list)
    if missing_items:
        raise ValueError(f"get_item_list is missing released Superman fields: {sorted(missing_items)}")
    return args


def prepare_motion_tokenizer():
    model = VisionGuidedMotionTokenizer(
        vqvae_config.encoder,
        vqvae_config.decoder,
        vqvae_config.vq,
        vision_config=vision_config,
        joint_data_type=vqvae_config.joint_data_type,
    )
    load_motion_tokenizer_weights(model, vqvae_config.resume_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.cuda()


def build_dataset(args, dataset_args_file, slice_indices_file):
    dataset_args = {
        "designated_split": args.data_split,
        "num_frames": args.num_frames,
        "sample_stride": args.sample_stride,
        "data_stride": args.data_stride,
        "return_extra": args.return_extra,
        "get_item_list": args.get_item_list,
        "load_data_file": args.load_data_file,
        "load_image_source_file": args.load_image_source_file,
        "load_bbox_file": args.load_bbox_file,
        "load_text_source_file": args.load_text_source_file,
        "normalize": args.normalize,
        "filter_invalid_images": args.filter_invalid_images,
        "processed_image_shape": args.processed_image_shape,
        "if_resample": args.if_resample,
        "backbone": args.backbone,
    }
    with open(dataset_args_file, "w") as file:
        json.dump(dataset_args, file, indent=4)

    dataset = Multimodal_Mocap_Dataset(**dataset_args)
    with open(slice_indices_file, "w") as file:
        for _, slice_indices, _, _ in tqdm(dataset.data_list):
            item = {
                "start_id": int(slice_indices[0]),
                "end_id": int(slice_indices[-1] + 1),
            }
            file.write(json.dumps(item) + "\n")
    return dataset


def load_or_build_dataset(args, dataset_args_file, slice_indices_file):
    if osp.exists(dataset_args_file) and osp.exists(slice_indices_file):
        with open(dataset_args_file) as file:
            return Multimodal_Mocap_Dataset(**json.load(file))
    return build_dataset(args, dataset_args_file, slice_indices_file)


def encode_dataset(args, dataset):
    motion_tokenizer = prepare_motion_tokenizer()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=16,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    prefetcher = data_prefetcher(dataloader, device=torch.device(0))
    batch = prefetcher.next()
    outputs = defaultdict(list)

    with torch.no_grad(), tqdm(total=len(dataloader)) as progress:
        while batch is not None:
            batch = edict(batch)
            joint3d_video = batch[vqvae_config.joint_data_type]
            video_rgb = batch["video_rgb"].cuda()
            codebook_indices, quant_shape = motion_tokenizer.encode(
                joint3d_video=joint3d_video,
                video_rgb=video_rgb,
            )

            outputs[f"{vqvae_config.joint_data_type}_code"].append(
                codebook_indices.cpu().numpy()
            )
            outputs["quant_shape"].append(
                np.repeat(np.array(quant_shape[1:])[None], quant_shape[0], axis=0)
            )
            outputs["slice_id"].append(batch["slice_id"].cpu().numpy())
            outputs["image_sources"].append(batch["image_sources"])

            progress.update(1)
            batch = prefetcher.next()

    outputs = {key: np.concatenate(values, axis=0) for key, values in outputs.items()}
    lengths = {len(values) for values in outputs.values()}
    if len(lengths) != 1:
        raise ValueError(f"Encoded dataset fields have inconsistent lengths: {lengths}")
    return outputs


def get_auxiliary_keys(dataset):
    candidates = [
        "affine_trans",
        "affine_trans_inv",
        "joint2d_cpn",
        "factor_2_5d",
    ]
    auxiliary_keys = [key for key in candidates if key in dataset.get_item_list]
    data_key = vqvae_config.joint_data_type
    if "normed" in data_key:
        auxiliary_keys = [
            data_key.replace("normed", "scale"),
            data_key.replace("normed", "transl"),
            *auxiliary_keys,
        ]
    return auxiliary_keys


def write_jsonl(dataset, encoded_data, jsonl_file):
    data_key = vqvae_config.joint_data_type
    auxiliary_keys = get_auxiliary_keys(dataset)
    with open(jsonl_file, "w") as file:
        for sample_id in tqdm(range(len(encoded_data["slice_id"]))):
            slice_indices = encoded_data["slice_id"][sample_id]
            skeleton = {
                "st_id": int(slice_indices[0]),
                "ed_id": int(slice_indices[-1]) + 1,
                "sample_id": sample_id,
                "data_key": data_key,
                "data_aux_key": auxiliary_keys,
                "task": TASK,
                "prompt_type": PROMPT_TYPE,
                "get_skel_str_func": SKELETON_FORMAT,
            }
            item = edict(TASK_TEMPLATE[TASK])
            item.conversations[0]["value"] = PROMPT_PLACEHOLDER
            item.videos = [[str(path) for path in encoded_data["image_sources"][sample_id].tolist()]]
            item.skeletons = [skeleton]
            file.write(json.dumps(item) + "\n")


def main():
    args = get_args()
    output_dir = osp.join(
        args.save_root,
        args.save_subdir_raw,
        args.save_subdir_vqvae,
        args.save_subdir_jsonl,
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset_args_file = osp.join(output_dir, f"{args.data_split}_dataset_args.json")
    slice_indices_file = osp.join(output_dir, f"{args.data_split}_slice_indices.jsonl")
    dataset = load_or_build_dataset(args, dataset_args_file, slice_indices_file)

    vqvae_output_file = osp.join(output_dir, f"{args.data_split}_vqvae_output.pkl")
    if osp.exists(vqvae_output_file):
        encoded_data = joblib.load(vqvae_output_file)
    else:
        encoded_data = encode_dataset(args, dataset)
        joblib.dump(encoded_data, vqvae_output_file)

    prompt_config = {
        "task": TASK,
        "prompt_type": PROMPT_TYPE,
        "get_skel_str_func": SKELETON_FORMAT,
    }
    with open(osp.join(output_dir, f"{args.data_split}_prompt_config.json"), "w") as file:
        json.dump(prompt_config, file, indent=4)

    jsonl_file = osp.join(output_dir, f"{args.data_split}_data.jsonl")
    write_jsonl(dataset, encoded_data, jsonl_file)
    print(f"Saved {len(encoded_data['slice_id'])} samples to {jsonl_file}.")


if __name__ == "__main__":
    main()
