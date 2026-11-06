"""Observations in robot-base coordinates; no dependence on frame ordering."""

import torch
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

TCP_OFFSET = (0.0, 0.0, 0.1034)


def tcp_pose(env):
    robot = env.scene["robot"]
    hand = robot.find_bodies("panda_hand")[0][0]
    offset = torch.tensor(TCP_OFFSET, device=env.device).expand(env.num_envs, -1)
    pos, quat = combine_frame_transforms(robot.data.body_pos_w[:, hand], robot.data.body_quat_w[:, hand], offset)
    pos, quat = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos, quat)
    return torch.cat((pos, quat), dim=-1)


def fingers(env):
    robot = env.scene["robot"]
    ids = robot.find_joints("panda_finger_joint.*")[0]
    return robot.data.joint_pos[:, ids]


def goal_distance(env):
    goal = torch.tensor(env.cfg.point_b, device=env.device)
    return (tcp_pose(env)[:, :3] - goal).norm(dim=-1, keepdim=True)


def line_distance(env):
    a = torch.tensor(env.cfg.point_a, device=env.device)
    b = torch.tensor(env.cfg.point_b, device=env.device)
    delta = tcp_pose(env)[:, :3] - a
    progress = ((delta * (b - a)).sum(-1) / (b - a).square().sum()).clamp(0, 1)
    return (delta - progress[:, None] * (b - a)).norm(dim=-1, keepdim=True)


def success(env):
    return goal_distance(env) <= env.cfg.goal_tolerance
