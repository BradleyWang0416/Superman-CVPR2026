"""Skeleton-token text format used by the released Superman model."""

import re

from llamafactory.extras.constants import (
    BODY_PART_ORDER,
    BODY_PART_TOKENS,
    JOINT_GROUP_MAP,
    SKELETON_TOKEN_BASE,
)


def get_skeleton_token_str_wTextualBodyPart_SplitByFrame(skeleton_indices):
    """Format a (frame, joint) codebook-index array as body-part-aware text."""
    frame_strings = []
    for frame_id, frame_indices in enumerate(skeleton_indices, start=1):
        part_strings = []
        for part_id, part_name in enumerate(BODY_PART_ORDER):
            start_token, _ = BODY_PART_TOKENS[part_name]
            label = start_token.replace("<", "").replace(">", ": ")
            suffix = ". " if part_id < len(BODY_PART_ORDER) - 1 else "."
            tokens = "".join(
                SKELETON_TOKEN_BASE.format(int(frame_indices[joint_id]))
                for joint_id in JOINT_GROUP_MAP[part_name]
            )
            part_strings.append(f"{label}{tokens}{suffix}")

        frame_strings.append(f"Frame {frame_id}: {''.join(part_strings)}")

    prefix = (
        f"There are {len(frame_strings)} frames in total. "
        f"Here are the skeleton tokens for {len(BODY_PART_ORDER)} body parts in each frame:\n"
    )
    return prefix + "\n".join(frame_strings)


def parse_skeleton_token_str_wTextualBodyPart_SplitByFrame(skeleton_token_str):
    """Parse the released body-part-aware format back into joint-order indices."""
    ordered_joint_ids = [
        joint_id
        for part_name in BODY_PART_ORDER
        for joint_id in JOINT_GROUP_MAP[part_name]
    ]
    num_joints = len(set(ordered_joint_ids))
    data_anchor = f"{BODY_PART_ORDER[0]}:"
    token_pattern = re.compile(r"<skel_(\d+)>")
    frames = []

    for line in skeleton_token_str.splitlines():
        line = line.replace("<|skel_start|>", "").replace("<|skel_end|>", "").strip()
        anchor_index = line.find(data_anchor)
        if anchor_index == -1:
            continue

        extracted_tokens = token_pattern.findall(line[anchor_index:])
        if len(extracted_tokens) != len(ordered_joint_ids):
            continue

        frame = [0] * num_joints
        for joint_id, token in zip(ordered_joint_ids, extracted_tokens):
            frame[joint_id] = int(token)
        frames.append(frame)

    return frames
