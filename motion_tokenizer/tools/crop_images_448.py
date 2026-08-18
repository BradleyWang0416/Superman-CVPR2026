"""Generate the person-centred 448x448 crops that the dataloader expects.

`Multimodal_Mocap_Dataset` never crops at run time -- it rewrites every image
path by renaming one directory component (``images_fps50`` ->
``images_fps50_cropped_448x448``, etc.) and reads the result directly.  The
crops therefore have to exist on disk *before* training, and they have to be
produced with exactly the affine transform the dataloader applies to the
joints; otherwise the deformable sampling in `VisualSkeletonAttention` reads
pixels that do not correspond to the joint coordinates.

This script reuses `get_affine_transform` from ``lib/dataset.py`` verbatim, so the
two stay in sync by construction.

Usage (run from motion_tokenizer/)
---------------------------------
    python tools/crop_images_448.py \
        --images-source data/h36m/images_source.pkl \
        --bboxes        data/h36m/bboxes_xyxy.pkl \
        --splits train test \
        --workers 16

The output tree mirrors the input tree; files that already exist are skipped,
so the script is safe to re-run after an interruption.
"""

import argparse
import os
import os.path as osp
import sys
from multiprocessing import Pool

import cv2
import joblib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), os.pardir))
from lib.dataset import DATA_ROOT_PATH, get_affine_transform  # noqa: E402


# Same rename table as Multimodal_Mocap_Dataset.__init__.
DIR_MARKERS = ['images_fps50', 'imageFiles', 'imageSequence']


def to_cropped_path(img_path, size):
    """Map an original image path to its cropped counterpart."""
    for marker in DIR_MARKERS:
        if marker in img_path:
            return img_path.replace(marker, f'{marker}_cropped_{size}x{size}')
    raise NotImplementedError(
        f"image path does not contain any of {DIR_MARKERS}: {img_path}"
    )


def crop_one(job):
    src, dst, bbox, size = job

    if osp.exists(dst):
        return 'skipped', src

    img = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return 'unreadable', src

    center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    scale = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    trans = get_affine_transform(center, scale, 0, [size, size])
    cropped = cv2.warpAffine(img, trans, (size, size))

    os.makedirs(osp.dirname(dst), exist_ok=True)
    # Write to a temp name and rename, so an interrupted run never leaves a
    # half-written file behind. Keep the extension -- cv2 picks the encoder from it.
    stem, ext = osp.splitext(dst)
    tmp = f'{stem}.tmp{os.getpid()}{ext}'
    if not cv2.imwrite(tmp, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        return 'write_failed', src
    os.replace(tmp, dst)
    return 'done', src


def build_jobs(images_source, bboxes, splits, size):
    jobs = []
    for split in splits:
        if split not in images_source:
            raise KeyError(f"split '{split}' not in the images-source pickle")
        img_list = images_source[split]
        bbox_list = bboxes[split]
        if len(img_list) != len(bbox_list):
            raise ValueError(
                f"split '{split}': {len(img_list)} image paths but "
                f"{len(bbox_list)} bounding boxes -- the two pickles must be "
                f"index-aligned"
            )
        for img_path, bbox in zip(img_list, bbox_list):
            if img_path is None:          # frames without an image are dropped by the dataloader
                continue
            src = osp.join(DATA_ROOT_PATH, img_path)
            jobs.append((src, to_cropped_path(src, size), np.asarray(bbox, dtype=np.float64), size))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images-source', required=True,
                        help="the load_image_source_file pickle, {split: [path or None]}")
    parser.add_argument('--bboxes', required=True,
                        help="the load_bbox_file pickle, {split: (N, 4) xyxy array}")
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--size', type=int, default=448, help="output crop size (default: 448)")
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument('--limit', type=int, default=None,
                        help="only process the first N images (for a quick smoke test)")
    args = parser.parse_args()

    print(f"Loading {args.images_source} ...")
    images_source = joblib.load(args.images_source)
    print(f"Loading {args.bboxes} ...")
    bboxes = joblib.load(args.bboxes)

    jobs = build_jobs(images_source, bboxes, args.splits, args.size)
    if args.limit is not None:
        jobs = jobs[:args.limit]

    if not jobs:
        print("Nothing to do.")
        return 0

    print(f"{len(jobs):,} images to process with {args.workers} worker(s)")
    print(f"  example input : {jobs[0][0]}")
    print(f"  example output: {jobs[0][1]}")

    # cv2 spawns its own threads; keep it to one per worker process.
    cv2.setNumThreads(1)

    stats = {'done': 0, 'skipped': 0, 'unreadable': 0, 'write_failed': 0}
    failures = []
    with Pool(processes=args.workers) as pool:
        for status, src in tqdm(pool.imap_unordered(crop_one, jobs, chunksize=64),
                                total=len(jobs), desc='cropping'):
            stats[status] += 1
            if status in ('unreadable', 'write_failed'):
                failures.append((status, src))

    print("\n--- Summary ---")
    for key, value in stats.items():
        print(f"  {key:<13}: {value:,}")

    if failures:
        log_path = 'crop_images_448_failures.txt'
        with open(log_path, 'w') as fh:
            for status, src in failures:
                fh.write(f'{status}\t{src}\n')
        print(f"\n{len(failures):,} failure(s) written to {log_path}")
        print("The dataloader raises FileNotFoundError on any missing crop, so "
              "resolve these before training.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
