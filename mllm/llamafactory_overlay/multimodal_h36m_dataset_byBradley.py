import os
import os.path as osp
import joblib
import numpy as np
import cv2
import torch
from collections import defaultdict
import json
from tqdm import tqdm
import sys


# Root that the relative paths inside images_source.pkl are resolved against.
# Defaults to the Stage 1 data directory; override with SUPERMAN_DATA_ROOT.
DATA_ROOT_PATH = os.environ.get("SUPERMAN_DATA_ROOT", "data")

# The annotation reader and the affine person-crop are shared with Stage 1, so that
# both stages slice clips and crop images in exactly the same way.
from superman_stage1 import DataReader as DataReaderMesh, get_affine_transform


joints_left = [4, 5, 6, 11, 12, 13]
joints_right = [1, 2, 3, 14, 15, 16]


class Multimodal_Mocap_Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=16, sample_stride=1, data_stride=16, data_mode="joint3d", designated_split='train',
                 load_data_file=osp.join(DATA_ROOT_PATH, "h36m/h36m_sh_conf_cam_source_final_wJ2dCpn.pkl"),
                 load_image_source_file=osp.join(DATA_ROOT_PATH, "h36m/images_source.pkl"),
                 load_bbox_file=osp.join(DATA_ROOT_PATH, "h36m/bboxes_xyxy.pkl"),
                 load_text_source_file="",
                 load_camera_file=osp.join(DATA_ROOT_PATH, "h36m/cameras.pkl"),
                 return_extra=[['image']],
                 # data preprocessing config
                 normalize='isotropic',  # isotropic (i.e., screen_coordinate_normalize), anisotropic
                 # image config
                 filter_invalid_images=True,
                 processed_image_shape=None,    # e.g., (192,256)
                 backbone='hrnet_32',
                 # dataloader config
                 get_item_list=[],
                 batch_return_type='dict',
                 max_samples=None,
                 samples_range=None,
                 read_confidence=False,
                 if_resample=True,
                 ):
        if len(load_data_file.split(',')) > len(load_camera_file.split(',')):
            load_camera_file = load_camera_file + ','*(len(load_data_file.split(','))-len(load_camera_file.split(',')))
        assert len(load_data_file.split(',')) == len(load_image_source_file.split(',')) == len(return_extra) == len(load_bbox_file.split(',')) == len(load_camera_file.split(','))

        self.num_frames = num_frames
        self.get_item_list = get_item_list
        assert len(self.get_item_list) > 0
        self.batch_return_type = batch_return_type
        assert self.batch_return_type in ['dict', 'tuple']

        self.backbone = backbone
        if self.backbone in ['hrnet_32', 'hrnet_48']:
            self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif self.backbone == 'cpn':
            self.img_mean = np.array([122.7717, 115.9465, 102.9801], dtype=np.float32)
            self.img_mean /= np.float32(255.)
            self.img_std = np.float32(1)    # placeholder
        elif self.backbone == 'qwen2.5vl':
            pass
        else:
            NotImplementedError

        data_dict = {}
        data_list = []
        self.VALID_IMG_INDICES = {}
        if 'camera_param' in get_item_list:
            assert designated_split == 'test', 'due to the current code implementation, only support [designated_split=test] when loading camera parameters now.'
            self.camera_dict = {}
        for dt_file, img_src_file, bbox_file, extra_modality_list, camera_file in zip(load_data_file.split(','), load_image_source_file.split(','), load_bbox_file.split(','), return_extra, load_camera_file.split(',')):

            # Resolve data paths for the current environment.
            if dt_file != '' and not dt_file.startswith('/'):
                dt_file = osp.join(DATA_ROOT_PATH, dt_file)
            if img_src_file != '' and not img_src_file.startswith('/'):
                img_src_file = osp.join(DATA_ROOT_PATH, img_src_file)
            if bbox_file != '' and not bbox_file.startswith('/'):
                bbox_file = osp.join(DATA_ROOT_PATH, bbox_file)
            if camera_file != '' and not camera_file.startswith('/'):
                camera_file = osp.join(DATA_ROOT_PATH, camera_file)

            if 'camera_param' in get_item_list:
                camera_dict = joblib.load(camera_file)
                self.camera_dict[dt_file] = camera_dict

                with open(camera_file.replace('cameras.pkl', 'h36m_name_map.json'), "r") as f:
                    self.get_real_source = json.load(f)


            # Load images and find valid indices before applying sample_stride.
            use_image = 'image' in extra_modality_list
            if use_image:
                img_list = joblib.load(img_src_file)[designated_split]

                # images_source.pkl stores paths relative to the data root (the same
                # convention Stage 1 uses); os.path.join leaves absolute paths untouched,
                # so both conventions work.
                img_list = [None if p is None else osp.join(DATA_ROOT_PATH, p) for p in img_list]

                img_list = np.array(img_list)


                if filter_invalid_images:
                    valid_img_indices = []
                    for frame_id, img_path in enumerate(img_list):
                        if img_path is None:
                            continue
                        valid_img_indices.append(frame_id)
                        #     break
                else:
                    valid_img_indices = list(range(len(img_list)))


                if samples_range is not None:
                    assert isinstance(samples_range, list) and len(samples_range) == 2
                    sample_range_start, samples_range_end = samples_range
                    valid_img_indices = valid_img_indices[sample_range_start:samples_range_end]
                    print(f"samples_range applied: {osp.basename(img_list[valid_img_indices[0]])} -- {osp.basename(img_list[valid_img_indices[-1]])}")
                elif max_samples is not None:
                    valid_img_indices = valid_img_indices[:max_samples]


                img_list = np.array(img_list)[valid_img_indices]   # resample according to valid_img_indices (sample_stride not applied yet here)
                img_list = img_list[::sample_stride]  # sample_stride applied here


                if processed_image_shape is not None:
                    img_list = img_list.tolist()
                    assert processed_image_shape[0] == 192 and processed_image_shape[1] == 256 or \
                        processed_image_shape[0] == 448 and processed_image_shape[1] == 448
                    for frame_id, img_path in enumerate(img_list):
                        if 'images_fps50' in img_path:
                            img_list[frame_id] = img_path.replace('images_fps50', f'images_fps50_cropped_{processed_image_shape[0]}x{processed_image_shape[1]}')
                        elif 'imageFiles' in img_path:
                            img_list[frame_id] = img_path.replace('imageFiles', f'imageFiles_cropped_{processed_image_shape[0]}x{processed_image_shape[1]}')
                        elif 'imageSequence' in img_path:
                            img_list[frame_id] = img_path.replace('imageSequence', f'imageSequence_cropped_{processed_image_shape[0]}x{processed_image_shape[1]}')
                        elif 'idea_merge_images' in img_path:
                            img_list[frame_id] = img_path.replace('idea_merge_images', f'idea_merge_images_cropped_{processed_image_shape[0]}x{processed_image_shape[1]}')
                        else:
                            raise NotImplementedError
                        if frame_id % 10000 == 0:
                            assert osp.exists(img_list[frame_id]), f'img_list[{frame_id}]={img_list[frame_id]} not exists.'
                    img_list = np.array(img_list)
            else:
                valid_img_indices = slice(None)   # all valid


                if samples_range is not None:
                    assert isinstance(samples_range, list) and len(samples_range) == 2
                    sample_range_start, samples_range_end = samples_range
                    valid_img_indices = slice(sample_range_start, samples_range_end)
                    print(f"samples_range applied: {sample_range_start} -- {samples_range_end}")
                elif max_samples is not None:
                    valid_img_indices = slice(0, max_samples)


            self.VALID_IMG_INDICES[dt_file] = valid_img_indices


            # Load bounding boxes.
            if use_image and ('bboxes_xyxy' in self.get_item_list or processed_image_shape is not None):
                bboxes_xyxy = joblib.load(bbox_file)[designated_split]
                bboxes_xyxy = bboxes_xyxy[valid_img_indices]
                bboxes_xyxy = bboxes_xyxy[::sample_stride]

            # Load joints and resample them to the valid image indices.
            datareader_config_unsplit = {'dt_file': dt_file,}
            datareader_config_split = {'chunk_len': num_frames,
                                       'sample_stride': sample_stride,
                                       'data_stride': data_stride,
                                       'read_confidence': read_confidence}
            datareader_config = {**datareader_config_unsplit, **datareader_config_split}
            datareader = DataReaderMesh(**datareader_config)
            unsplit_data = DataReaderMesh.load_dataset_static(**datareader_config_unsplit)

            for data_mode in unsplit_data[designated_split].keys():
                if isinstance(unsplit_data[designated_split][data_mode], list):
                    unsplit_data[designated_split][data_mode] = np.array(unsplit_data[designated_split][data_mode])[valid_img_indices].tolist()   # resample according to valid_img_indices (sample_stride not applied yet here)
                else:
                    unsplit_data[designated_split][data_mode] = unsplit_data[designated_split][data_mode][valid_img_indices]


            data_dict[dt_file] = {}


            datareader.dt_dataset = unsplit_data
            if any('joint3d_image' in get_item for get_item in self.get_item_list):
                joint3d_image = datareader.read_3d_image(designated_split=designated_split, do_screen_coordinate_normalize=False)     # (N,17,3). sample_stride applied here
                data_dict[dt_file]['joint3d_image'] = joint3d_image
            if any('joint3d_cam' in get_item for get_item in self.get_item_list):
                joint3d_cam = datareader.read_joint(key='joint_3d_cam', designated_split=designated_split)     # (N,17,3). sample_stride applied here
                data_dict[dt_file]['joint3d_cam'] = joint3d_cam
            if 'joint2d_cpn' in self.get_item_list:
                joint2d_cpn = datareader.read_joint(key='joint_2d_cpn', designated_split=designated_split)     # (N,17,3). sample_stride applied here
                data_dict[dt_file]['joint2d_cpn'] = joint2d_cpn
            if 'joint2d_gt' in self.get_item_list:
                joint2d_gt = datareader.read_joint(key='joint_2d_gt', designated_split=designated_split)     # (N,17,3). sample_stride applied here
                data_dict[dt_file]['joint2d_gt'] = joint2d_gt
            if 'joint2d' in self.get_item_list:
                joint2d = datareader.read_2d(designated_split=designated_split, do_screen_coordinate_normalize=False)     # (N,17,2) or (N,17,3). sample_stride applied here
                data_dict[dt_file]['joint2d'] = joint2d


            if any('joint3d_world' in get_item for get_item in self.get_item_list):
                joint3d_world = datareader.read_joint(key='joint_3d_world', designated_split=designated_split)     # (N,17,3). sample_stride applied here
                data_dict[dt_file]['joint3d_world'] = joint3d_world


            # Compute normalization factors from the original image dimensions.
            if any('joint3d_image' in get_item for get_item in self.get_item_list) or 'ori_img_wh' in self.get_item_list or \
                any('affined' in get_item for get_item in self.get_item_list):
                img_ori_wh = datareader.read_hw(designated_split=designated_split)    # (N,2). sampled_stride applied within read_hw
                img_ori_w, img_ori_h = img_ori_wh[:, 0:1], img_ori_wh[:, 1:2]   # (N,1); (N,1)
                data_dict[dt_file]['ori_img_wh'] = img_ori_wh
            if any('joint3d_image' in get_item for get_item in self.get_item_list):
                if normalize == 'isotropic':
                    joint3d_image_scale = np.concatenate([img_ori_w / 2, img_ori_w / 2, img_ori_w / 2], axis=-1) # (N,3)
                    joint3d_image_transl = np.concatenate([np.ones_like(img_ori_w), img_ori_h / img_ori_w, np.zeros_like(img_ori_w)], axis=-1) # (N,3)
                elif normalize == 'anisotropic':
                    joint3d_image_scale = np.concatenate([img_ori_w // 2, img_ori_h // 2, img_ori_w / 2], axis=-1) # (N,3)
                    joint3d_image_transl = np.concatenate([np.ones_like(img_ori_w), np.ones_like(img_ori_h), np.zeros_like(img_ori_w)], axis=-1) # (N,3)
                else:
                    NotImplementedError
                data_dict[dt_file]['joint3d_image_scale'] = joint3d_image_scale
                data_dict[dt_file]['joint3d_image_transl'] = joint3d_image_transl


            data_sources = datareader.read_source(designated_split=designated_split)    # sampled_stride applied within read_source
            data_dict[dt_file]['sources'] = data_sources

            if 'actions' in self.get_item_list:
                data_actions = datareader.read_action(designated_split=designated_split)    # sampled_stride applied within read_source
                data_dict[dt_file]['actions'] = data_actions

            # Load 2.5D factors and joint coordinates.
            if 'factor_2_5d' in self.get_item_list or 'joint_2_5d_image' in self.get_item_list:
                if designated_split == 'test':
                    factor_2_5d = datareader.read_2_5d_factor(designated_split=designated_split)    # sampled_stride applied within read_source
                    joint_2_5d_image = datareader.read_2_5d_image(designated_split=designated_split)    # sampled_stride applied within read_source
                else:
                    factor_2_5d = np.zeros((joint3d_image.shape[0],), dtype=np.float32)
                    joint_2_5d_image = np.zeros_like(joint3d_image)
                data_dict[dt_file]['2.5d_factor'] = factor_2_5d
                data_dict[dt_file]['joint_2.5d_image'] = joint_2_5d_image
            if use_image:
                data_dict[dt_file]['image_sources'] = img_list
                if 'bboxes_xyxy' in self.get_item_list:
                    data_dict[dt_file]['bboxes_xyxy'] = bboxes_xyxy


            if 'camera_param' in get_item_list:
                camera_param_all = []
                for source_w_sa in data_sources:
                    subject_id = source_w_sa.split('_')[1]
                    source_readable = self.get_real_source[source_w_sa]
                    camera_name = source_readable.split('.')[-1]
                    camera_param = camera_dict[(f'S{int(subject_id)}', camera_name)]    # dict_keys(['R', 'T', 'c', 'f', 'k', 'p', 'w', 'h', 'name', 'id'])
                    camera_param_all.append(camera_param)
                assert len(data_sources) == len(camera_param_all)
                data_dict[dt_file]['camera_param'] = camera_param_all


            # Affine-transform poses to align them with the images.
            if use_image and processed_image_shape is not None:

                if any('joint3d_image' in get_item for get_item in self.get_item_list) or \
                    'affine_trans' in self.get_item_list or 'affine_trans_inv' in self.get_item_list or \
                    'processed_img_wh' in self.get_item_list:


                    AFFINE_TRANS = []
                    AFFINE_TRANS_INV = []
                    joint3d_image_affined = np.zeros_like(joint3d_image)
                    for i in range(joint3d_image.shape[0]):
                        bbox = bboxes_xyxy[i]
                        center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
                        scale = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                        trans = get_affine_transform(center, scale, 0, processed_image_shape)

                        pose_xy = joint3d_image[i, :, :2].copy()   # (17,2)
                        pose_xy1 = np.concatenate([pose_xy, np.ones((pose_xy.shape[0],1))], axis=1)   # (17,3)
                        pose_xy_affined = np.einsum('ij,kj->ik', pose_xy1, trans)


                        trans_inv = get_affine_transform(center, scale, 0, processed_image_shape, inv=1)
                        AFFINE_TRANS.append(trans)
                        AFFINE_TRANS_INV.append(trans_inv)
                        pose_z = joint3d_image[i, :, 2:3].copy()   # (17,1). pose_z[0] should already be zero
                        pose_z_affined = pose_z - pose_z[0:1, :]   # root-relative. pose_z_affined should be the same as pose_z

                        joint3d_image_affined[i, :, :2] = pose_xy_affined
                        joint3d_image_affined[i, :, 2:3] = pose_z_affined

                    assert (joint3d_image_affined[..., 2] == joint3d_image[..., 2]).all()   # pose_z should be the same


                    AFFINE_TRANS = np.stack(AFFINE_TRANS)
                    AFFINE_TRANS_INV = np.stack(AFFINE_TRANS_INV)
                    data_dict[dt_file]['affine_trans'] = AFFINE_TRANS
                    data_dict[dt_file]['affine_trans_inv'] = AFFINE_TRANS_INV


                    data_dict[dt_file]['joint3d_image_affined'] = joint3d_image_affined
                    data_dict[dt_file]['processed_img_wh'] = np.array([processed_image_shape]*joint3d_image.shape[0], dtype=np.int32)   # (N,2)

                    processed_img_w, processed_img_h = processed_image_shape[0], processed_image_shape[1]
                    if normalize == 'isotropic':
                        joint3d_image_affined_scale = np.concatenate([np.array([[processed_img_w / 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                                    np.array([[processed_img_w / 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                                    img_ori_w / 2,
                                                                    ], axis=-1) # (N,3)
                        joint3d_image_affined_transl =np.array([[1, processed_img_h / processed_img_w, 0]]).repeat(joint3d_image.shape[0], axis=0) # (N,3)
                    elif normalize == 'anisotropic':
                        joint3d_image_affined_scale = np.concatenate([np.array([[processed_img_w // 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                                    np.array([[processed_img_h // 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                                    img_ori_w / 2,
                                                                    ], axis=-1) # (N,3)
                        joint3d_image_affined_transl =np.array([[1, 1, 0]]).repeat(joint3d_image.shape[0], axis=0) # (N,3)
                    else:
                        NotImplementedError
                    data_dict[dt_file]['joint3d_image_affined_scale'] = joint3d_image_affined_scale
                    data_dict[dt_file]['joint3d_image_affined_transl'] = joint3d_image_affined_transl

            split_id = datareader.get_split_id(designated_split=designated_split, if_resample=if_resample)   # 这里是用 unsplit_data 中的 'source' 来划分 split_id, 所以也要利用 valid_indices 作修改

            data_list.extend(zip([dt_file]*len(split_id), split_id, [use_image]*len(split_id), [None]*len(split_id)))

        self.data_dict = data_dict
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Local variable names must match get_item_list because values are collected via locals().
        dt_file, slice_id, use_image, caption = self.data_list[idx]

        if any('joint3d_world' in get_item for get_item  in self.get_item_list):
            joint3d_world = self.data_dict[dt_file]['joint3d_world'][slice_id]  # (num_frames, 17, 2)
            joint3d_world_meter = joint3d_world / 1000  # (num_frames, 17, 3)
            joint3d_world_rootrel_meter = (joint3d_world - joint3d_world[..., 0:1, :]) / 1000  # (num_frames, 17, 3)

        if 'joint2d' in self.get_item_list:
            joint2d = self.data_dict[dt_file]['joint2d'][slice_id]  # (num_frames, 17, 2)
        if 'joint2d_cpn' in self.get_item_list:
            joint2d_cpn = self.data_dict[dt_file]['joint2d_cpn'][slice_id]  # (num_frames, 17, 2)
        if 'joint2d_gt' in self.get_item_list:
            joint2d_gt = self.data_dict[dt_file]['joint2d_gt'][slice_id]  # (num_frames, 17, 2)
        if any('joint3d_cam' in get_item for get_item in self.get_item_list):
            joint3d_cam = self.data_dict[dt_file]['joint3d_cam'][slice_id]  # (num_frames, 17, 3)
            joint3d_cam_meter = joint3d_cam / 1000  # (num_frames, 17, 3)
            joint3d_cam_rootrel_meter = (joint3d_cam - joint3d_cam[..., 0:1, :]) / 1000  # (num_frames, 17, 3)
        if any('joint3d_image' in get_item for get_item in self.get_item_list):
            joint3d_image = self.data_dict[dt_file]['joint3d_image'][slice_id]  # (num_frames, 17, 3)
        if 'joint_2_5d_image' in self.get_item_list:
            joint_2_5d_image = self.data_dict[dt_file]['joint_2.5d_image'][slice_id]  # (num_frames, 17, 3)
        if 'factor_2_5d' in self.get_item_list:
            factor_2_5d = self.data_dict[dt_file]['2.5d_factor'][slice_id]  # (num_frames,) only for test
        if 'ori_img_wh' in self.get_item_list:
            ori_img_wh = self.data_dict[dt_file]['ori_img_wh'][slice_id]  # (num_frames, 2). element: (res_w, res_h)
        sources = self.data_dict[dt_file]['sources'][slice_id]  # (num_frames, 2). element: (res_w, res_h)
        if 'actions' in self.get_item_list:
            actions = self.data_dict[dt_file]['actions'][slice_id]

        # Normalize image-space joints.
        if any('joint3d_image' in get_item for get_item in self.get_item_list):
            joint3d_image_scale = self.data_dict[dt_file]['joint3d_image_scale'][slice_id]
            joint3d_image_transl = self.data_dict[dt_file]['joint3d_image_transl'][slice_id]
            joint3d_image_normed = joint3d_image / joint3d_image_scale[..., None, :] - joint3d_image_transl[..., None, :]

        if 'camera_param' in self.get_item_list:
            camera_param = [self.data_dict[dt_file]['camera_param'][frame_id] for frame_id in slice_id]

        if use_image:
            # Load and normalize images.
            image_sources = self.data_dict[dt_file]['image_sources'][slice_id]  # (num_frames,)

            if 'processed_img_wh' in self.get_item_list:
                processed_img_wh = self.data_dict[dt_file]['processed_img_wh'][slice_id]  # (num_frames, 2). element: (res_w, res_h)
            if 'affine_trans' in self.get_item_list:
                affine_trans = self.data_dict[dt_file]['affine_trans'][slice_id]  # (num_frames, 3, 2)
            if 'affine_trans_inv' in self.get_item_list:
                affine_trans_inv = self.data_dict[dt_file]['affine_trans_inv'][slice_id]  # (num_frames, 3, 2)
            if 'bboxes_xyxy' in self.get_item_list:
                bboxes_xyxy = self.data_dict[dt_file]['bboxes_xyxy'][slice_id]  # (num_frames, 4)


            if 'video_rgb' in self.get_item_list:
                if self.backbone == 'qwen2.5vl':
                    video_rgb = image_sources
                else:
                    video_bgr = []
                    for img_path in image_sources:
                        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        if image_bgr is None:
                            raise FileNotFoundError(f"Failed to read image: {img_path}")
                        video_bgr.append(image_bgr)
                    video_bgr = np.stack(video_bgr, axis=0)  # (num_frames, H, W, 3), BGR order

                    video_rgb = video_bgr[..., ::-1].astype(np.float32)  # Convert BGR to RGB
                    # Stay in float32: dividing uint8 by a Python float promotes to float64,
                    # which doubles the memory traffic for no gain -- the tensor is cast back
                    # down to float32 on the way out.
                    video_rgb = (video_rgb / np.float32(255.0) - self.img_mean) / self.img_std   # to [0,1], then normalize

            # Load and normalize affine-transformed image-space joints.
            if any('joint3d_image_affined' in get_item for get_item in self.get_item_list):
                joint3d_image_affined = self.data_dict[dt_file]['joint3d_image_affined'][slice_id]  # (num_frames, 17, 3)
                joint3d_image_affined_scale = self.data_dict[dt_file]['joint3d_image_affined_scale'][slice_id]  # (num_frames, 3)
                joint3d_image_affined_transl = self.data_dict[dt_file]['joint3d_image_affined_transl'][slice_id]
                joint3d_image_affined_normed = joint3d_image_affined / joint3d_image_affined_scale[..., None, :] - joint3d_image_affined_transl[..., None, :]

        slice_id = np.array(slice_id).astype(np.int64)

        return_dict = {}
        for get_item in self.get_item_list:
            item = locals()[get_item]
            try:
                item = torch.from_numpy(item)
                if item.dtype == torch.int64:
                    pass
                else:
                    item = item.float()
            except:
                pass
            return_dict[get_item] = item
        # e.g., return_dict = (joint3d_image, joint3d_image_normed, factor_2_5d, joint3d_image_scale, joint3d_image_transl)
        # e.g., return_dict = (joint3d_image, joint3d_image_normed, factor_2_5d, joint3d_image_scale, joint3d_image_transl,
        #                       video_rgb, joint3d_image_affined, joint3d_image_affined_normed, joint3d_image_affined_scale, joint3d_image_affined_transl)

        return return_dict


    def collate_fn(self, batch):
        if 'camera_param' in self.get_item_list:
            raise NotImplementedError('due to the camera_param is a list of dict, the current collate_fn cannot handle it. you can try to remove camera_param from get_item_list first.')
        return_dict = defaultdict(list)
        for b in batch:
            for k, v in b.items():
                return_dict[k].append(v)

        for k, v in return_dict.items():
            try:
                return_dict[k] = torch.stack(v, dim=0)
            except:
                pass

        if len(return_dict) == 1:
            return return_dict[ list(return_dict.keys())[0] ]
        if self.batch_return_type == 'tuple':
            return_dict = tuple([v for k, v in return_dict.items()])
        return return_dict
