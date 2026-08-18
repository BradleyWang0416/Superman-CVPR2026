"""Prompt and sample template for the released Vid2Skel task."""

PROMPT_TEMPLATE = (
    "Please describe the motion of the person in the video <video> using skeleton tokens. "
    'Your response should be formatted as: "Frame 1: torso: ... left_arm: ... '
    'right_arm: ... left_leg: ... right_leg: ...\nFrame 2: ... "'
)

TASK_TEMPLATE = {
    "Vid2Skel": {
        "conversations": [
            {"from": "human", "value": None},
            {"from": "gpt", "value": "<skeleton>"},
        ],
        "videos": [],
        "skeletons": None,
    }
}
