"""Single integration point between Stage 2 and the Stage 1 motion tokenizer.

Stage 2 needs four things from Stage 1:

    * ``VisionGuidedMotionTokenizer`` -- the tokenizer model, to encode motion into
      indices offline and to decode predicted indices back into 3D joints;
    * ``DataReader`` -- the Human3.6M annotation reader;
    * ``get_affine_transform`` -- the affine person-crop used to build the 448x448
      images, which must match Stage 1 exactly;
    * the encoder / decoder / VQ / vision-backbone configs.

Rather than scattering ``sys.path`` edits across the code base, everything is
resolved here, once.

Point ``MOTION_TOKENIZER_DIR`` at the ``motion_tokenizer/`` directory of a
Superman-CVPR2026 checkout::

    export MOTION_TOKENIZER_DIR=/path/to/Superman-CVPR2026/motion_tokenizer

If unset, a sibling checkout next to this LLaMA-Factory tree is assumed.
"""

import os
import os.path as osp
import sys
from contextlib import contextmanager

_HERE = osp.dirname(osp.abspath(__file__))
_DEFAULT_DIR = osp.normpath(osp.join(_HERE, os.pardir, "Superman-CVPR2026", "motion_tokenizer"))

MOTION_TOKENIZER_DIR = osp.abspath(os.environ.get("MOTION_TOKENIZER_DIR", _DEFAULT_DIR))

if not osp.isdir(MOTION_TOKENIZER_DIR):
    raise ImportError(
        "Stage 1 not found at {!r}.\n"
        "Clone https://github.com/BradleyWang0416/Superman-CVPR2026 and set\n"
        "    export MOTION_TOKENIZER_DIR=/path/to/Superman-CVPR2026/motion_tokenizer"
        .format(MOTION_TOKENIZER_DIR)
    )


@contextmanager
def _stage1_on_path():
    """Expose Stage 1's top-level packages (models/, lib/, config/) only while importing.

    They are appended rather than prepended, and removed afterwards, so they cannot
    shadow LLaMA-Factory's own modules.
    """
    sys.path.append(MOTION_TOKENIZER_DIR)
    try:
        yield
    finally:
        try:
            sys.path.remove(MOTION_TOKENIZER_DIR)
        except ValueError:
            pass


with _stage1_on_path():
    from models.vgmt import VisionGuidedMotionTokenizer          # noqa: E402
    from lib.dataset import (                                    # noqa: E402
        DataReader,
        get_affine_transform,
        split_clips,
        resample,
    )
    from config.vision_backbone import config as vision_backbone_config   # noqa: E402
    from config.vqvae import vqvae_config                                 # noqa: E402


def load_motion_tokenizer_weights(model, checkpoint_dir):
    from safetensors.torch import load_file

    checkpoint_path = osp.join(checkpoint_dir, "model.safetensors")
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Released Stage 1 weights not found at {checkpoint_path}")

    state_dict = load_file(checkpoint_path, device="cpu")
    return model.load_state_dict(state_dict, strict=True)

__all__ = [
    "MOTION_TOKENIZER_DIR",
    "VisionGuidedMotionTokenizer",
    "load_motion_tokenizer_weights",
    "DataReader",
    "get_affine_transform",
    "split_clips",
    "resample",
    "vision_backbone_config",
    "vqvae_config",
]
