# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_DRAWER_TARGET_OFFSET = (-0.16, 0.0, -0.04)


def _drawer_target_pos_w(
    env: ManagerBasedRLEnv, target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET
) -> torch.Tensor:
    handle_pos_w = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    cabinet_quat_w = env.scene["cabinet_frame"].data.source_quat_w
    offset_b = torch.tensor(target_offset, device=handle_pos_w.device, dtype=handle_pos_w.dtype).repeat(
        handle_pos_w.shape[0], 1
    )
    return handle_pos_w + math_utils.quat_apply(cabinet_quat_w, offset_b)


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the end-effector TCP to the drawer handle frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data
    return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fingertip positions relative to each environment origin."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)
    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector TCP position relative to each environment origin."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    return ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """End-effector TCP orientation in world frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat_w = ee_tf_data.target_quat_w[..., 0, :]
    return math_utils.quat_unique(ee_quat_w) if make_quat_unique else ee_quat_w


def ee_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector TCP pose as position relative to env origin plus world quaternion."""
    return torch.cat((ee_pos(env), ee_quat(env)), dim=-1)


def gripper_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Raw Franka gripper hand pose as position relative to env origin plus world quaternion."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    gripper_pos = ee_tf_data.target_pos_w[..., 3, :] - env.scene.env_origins
    gripper_quat = math_utils.quat_unique(ee_tf_data.target_quat_w[..., 3, :])
    return torch.cat((gripper_pos, gripper_quat), dim=-1)


def root_pose(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """Asset root pose as position relative to env origin plus world quaternion."""
    asset = env.scene[asset_name]
    pos = asset.data.root_pos_w - env.scene.env_origins
    quat = math_utils.quat_unique(asset.data.root_quat_w)
    return torch.cat((pos, quat), dim=-1)


def cabinet_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cabinet root pose as position relative to env origin plus world quaternion."""
    return root_pose(env, "cabinet")


def charger_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Charger root pose as position relative to env origin plus world quaternion."""
    return root_pose(env, "charger")


def left_table_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left table pose as position relative to env origin plus world quaternion."""
    table_pos = torch.tensor((0.30, 0.42, 0.42), device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    table_quat = torch.tensor((0.707, 0.0, 0.0, 0.707), device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    return torch.cat((table_pos, table_quat), dim=-1)


def table_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Backward-compatible alias for the only table currently in the scene."""
    return left_table_pose(env)


def charger_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Charger root position relative to each environment origin."""
    return env.scene["charger"].data.root_pos_w - env.scene.env_origins


def rel_ee_charger_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the end-effector TCP to the charger."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    charger_pos_w = env.scene["charger"].data.root_pos_w
    return charger_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_charger_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the charger to the approximate placement target inside the top drawer."""
    charger_pos_w = env.scene["charger"].data.root_pos_w
    return _drawer_target_pos_w(env) - charger_pos_w


# Backward-compatible aliases for older configs that still say "cube".
def cube_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return charger_pos(env)


def rel_ee_cube_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rel_ee_charger_distance(env)


def rel_cube_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rel_charger_drawer_distance(env)
