# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_DRAWER_TARGET_OFFSET = (-0.16, 0.0, -0.04)


def _resolve_joint_id(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> int:
    """Resolve a single joint id even when a term is called before manager resolution."""
    asset = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.joint_ids, slice):
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
        return joint_ids[0]
    return asset_cfg.joint_ids[0]


def _drawer_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    joint_id = _resolve_joint_id(env, asset_cfg)
    return env.scene[asset_cfg.name].data.joint_pos[:, joint_id]


def _drawer_target_pos_w(
    env: ManagerBasedRLEnv, target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET
) -> torch.Tensor:
    """Approximate the placement target inside the opened top drawer.

    The target is anchored at the top drawer handle position, but the offset is
    expressed in the cabinet frame. The handle frame is rotated for gripper
    alignment, so using it for placement would push the target sideways.
    """
    handle_pos_w = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    cabinet_quat_w = env.scene["cabinet_frame"].data.source_quat_w
    offset_b = torch.tensor(target_offset, device=handle_pos_w.device, dtype=handle_pos_w.dtype).repeat(
        handle_pos_w.shape[0], 1
    )
    return handle_pos_w + quat_apply(cabinet_quat_w, offset_b)


def _charger_in_drawer_zone(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET,
    half_extents: tuple[float, float, float] = (0.34, 0.28, 0.22),
) -> torch.Tensor:
    """Check a placement zone near the drawer, robust to imported USD root offsets."""
    charger_pos = env.scene["charger"].data.root_pos_w
    target_pos = _drawer_target_pos_w(env, target_offset)
    half_extents_t = torch.tensor(half_extents, device=charger_pos.device, dtype=charger_pos.dtype).unsqueeze(0)
    return torch.all(torch.abs(charger_pos - target_pos) < half_extents_t, dim=-1)


def _charger_placed_in_drawer(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET,
    half_extents: tuple[float, float, float] = (0.30, 0.24, 0.07),
) -> torch.Tensor:
    """Stricter placement zone for success, so hovering over the drawer does not count."""
    return _charger_in_drawer_zone(env, target_offset, half_extents)


def _get_success_stage_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Create/reset per-episode success-stage latches used by demo recording."""
    device = env.device
    needs_init = not hasattr(env, "_open_drawer_success_stages")
    if not needs_init:
        stages = env._open_drawer_success_stages
        needs_init = stages["drawer_opened"].shape[0] != env.num_envs

    if needs_init:
        env._open_drawer_success_stages = {
            "drawer_opened": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "charger_picked": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "charger_placed": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "drawer_closed": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        }
        env._open_drawer_charger_start_z = env.scene["charger"].data.root_pos_w[:, 2].clone()
        return env._open_drawer_success_stages

    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
        for stage in env._open_drawer_success_stages.values():
            stage[reset_mask] = False
        env._open_drawer_charger_start_z[reset_mask] = env.scene["charger"].data.root_pos_w[reset_mask, 2]

    return env._open_drawer_success_stages


def _print_new_stage(stage_name: str, newly_reached: torch.Tensor, enabled: bool) -> None:
    if not enabled or not torch.any(newly_reached):
        return
    env_ids = newly_reached.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    print(f"[OpenDrawer] {stage_name} success reached for envs {env_ids}", flush=True)


def _print_stage3_box_debug(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    enabled: bool,
    interval: int,
) -> None:
    """Print a throttled stage-3 placement-box diagnostic for env 0."""
    if not enabled or interval <= 0:
        return
    if int(env.episode_length_buf[0].item()) % interval != 0:
        return

    charger_pos = env.scene["charger"].data.root_pos_w[0]
    target_pos = _drawer_target_pos_w(env, target_offset)[0]
    half_extents_t = torch.tensor(half_extents, device=charger_pos.device, dtype=charger_pos.dtype)
    abs_delta = torch.abs(charger_pos - target_pos)
    axis_ok = abs_delta < half_extents_t

    values = {
        "charger": charger_pos.detach().cpu().tolist(),
        "target": target_pos.detach().cpu().tolist(),
        "abs_delta": abs_delta.detach().cpu().tolist(),
        "half_extents": half_extents_t.detach().cpu().tolist(),
        "axis_ok": axis_ok.detach().cpu().tolist(),
    }
    print(
        "[OpenDrawer] stage 3 box debug "
        f"charger={values['charger']} target={values['target']} "
        f"abs_delta={values['abs_delta']} half_extents={values['half_extents']} "
        f"axis_ok(x,y,z)={values['axis_ok']}",
        flush=True,
    )


def _get_debug_draw_interface(env: ManagerBasedRLEnv):
    """Get Isaac Sim's debug-draw interface, enabling the extension if needed."""
    if hasattr(env, "_open_drawer_debug_draw"):
        return env._open_drawer_debug_draw

    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ImportError:
        try:
            import omni.kit.app

            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                "isaacsim.util.debug_draw", True
            )
            from isaacsim.util.debug_draw import _debug_draw
        except Exception:
            return None

    env._open_drawer_debug_draw = _debug_draw.acquire_debug_draw_interface()
    return env._open_drawer_debug_draw


def _append_box_edges(
    starts: list[tuple[float, float, float]],
    ends: list[tuple[float, float, float]],
    colors: list[tuple[float, float, float, float]],
    sizes: list[float],
    center: torch.Tensor,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float, float],
    size: float = 3.0,
) -> None:
    """Append a wireframe axis-aligned box to debug-draw line buffers."""
    hx, hy, hz = half_extents
    offsets = torch.tensor(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=center.dtype,
    )
    corners = center.unsqueeze(0) + offsets
    edge_ids = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    starts.extend(tuple(corners[i].tolist()) for i, _ in edge_ids)
    ends.extend(tuple(corners[j].tolist()) for _, j in edge_ids)
    colors.extend([color] * len(edge_ids))
    sizes.extend([size] * len(edge_ids))


def _draw_open_drawer_debug_boxes(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float],
    placement_half_extents: tuple[float, float, float],
    draw_stage3_box: bool,
    draw_randomization_boxes: bool,
    charger_randomization_center: tuple[float, float, float],
    charger_randomization_half_extents: tuple[float, float, float],
    cabinet_randomization_center: tuple[float, float, float],
    cabinet_randomization_half_extents: tuple[float, float, float],
) -> None:
    """Render stage-3 and reset-randomization debug boxes in the viewport."""
    if not draw_stage3_box and not draw_randomization_boxes:
        return

    draw = _get_debug_draw_interface(env)
    if draw is None:
        return

    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    sizes: list[float] = []

    if draw_stage3_box:
        stage3_center = _drawer_target_pos_w(env, target_offset)[0].detach().cpu()
        _append_box_edges(starts, ends, colors, sizes, stage3_center, placement_half_extents, (0.0, 1.0, 0.0, 1.0), 3.0)

    if draw_randomization_boxes:
        env_origin = env.scene.env_origins[0].detach().cpu()
        charger_center = env_origin + torch.tensor(charger_randomization_center, dtype=env_origin.dtype)
        cabinet_center = env_origin + torch.tensor(cabinet_randomization_center, dtype=env_origin.dtype)
        _append_box_edges(
            starts,
            ends,
            colors,
            sizes,
            charger_center,
            charger_randomization_half_extents,
            (0.0, 0.75, 1.0, 1.0),
            2.0,
        )
        _append_box_edges(
            starts,
            ends,
            colors,
            sizes,
            cabinet_center,
            cabinet_randomization_half_extents,
            (1.0, 0.85, 0.0, 1.0),
            2.0,
        )

    draw.clear_lines()
    if starts:
        draw.draw_lines(starts, ends, colors, sizes)

def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward reaching the drawer handle with the end-effector TCP."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)
    return torch.where(distance <= threshold, 2.0 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward aligning the gripper frame with the drawer handle frame."""
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return true when the two fingertips straddle the drawer handle vertically."""
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    left_finger_pos = fingertips_w[..., 0, :]
    right_finger_pos = fingertips_w[..., 1, :]
    return (right_finger_pos[:, 2] < handle_pos[:, 2]) & (left_finger_pos[:, 2] > handle_pos[:, 2])


def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
    """Reward placing both fingertips close to the drawer handle in a graspable pose."""
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    left_finger_pos = fingertips_w[..., 0, :]
    right_finger_pos = fingertips_w[..., 1, :]

    left_finger_dist = torch.abs(left_finger_pos[:, 2] - handle_pos[:, 2])
    right_finger_dist = torch.abs(right_finger_pos[:, 2] - handle_pos[:, 2])
    is_graspable = (right_finger_pos[:, 2] < handle_pos[:, 2]) & (left_finger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - left_finger_dist) + (offset - right_finger_dist))


def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward closing the fingers when the TCP is close to the drawer handle."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold
    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward drawer opening, doubled when the gripper is in a graspable pose."""
    drawer_pos = _drawer_joint_pos(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env).float()
    return (is_graspable + 1.0) * drawer_pos


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Staged drawer-opening bonus for early, middle, and late progress."""
    drawer_pos = _drawer_joint_pos(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env).float()

    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.2) * is_graspable
    open_hard = (drawer_pos > 0.3) * is_graspable
    return open_easy + open_medium + open_hard


def drawer_opened(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Successful when the selected drawer joint is opened beyond the threshold."""
    return _drawer_joint_pos(env, asset_cfg) > threshold


def approach_ee_charger(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward reaching the charger with the end-effector TCP."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    charger_pos = env.scene["charger"].data.root_pos_w
    distance = torch.norm(charger_pos - ee_tcp_pos, dim=-1, p=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)
    return torch.where(distance <= threshold, 2.0 * reward, reward)


def charger_lifted(env: ManagerBasedRLEnv, minimal_height: float) -> torch.Tensor:
    """Reward lifting the charger above the environment origin."""
    charger_height = env.scene["charger"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(charger_height > minimal_height, 1.0 + charger_height, charger_height / minimal_height)


def charger_to_drawer(
    env: ManagerBasedRLEnv,
    drawer_open_threshold: float,
    target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET,
) -> torch.Tensor:
    """Reward moving the charger toward the placement target after the drawer is open enough."""
    charger_pos = env.scene["charger"].data.root_pos_w
    target_pos = _drawer_target_pos_w(env, target_offset)
    distance = torch.norm(target_pos - charger_pos, dim=-1, p=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)

    drawer_joint_ids, _ = env.scene["cabinet"].find_joints(["drawer_top_joint"])
    drawer_pos = env.scene["cabinet"].data.joint_pos[:, drawer_joint_ids[0]]
    charger_height = charger_pos[:, 2] - env.scene.env_origins[:, 2]
    stage_gate = (drawer_pos > drawer_open_threshold) & (charger_height > 0.10)
    return stage_gate.float() * reward


def charger_in_drawer_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET,
    zone_half_extents: tuple[float, float, float] = (0.34, 0.28, 0.22),
) -> torch.Tensor:
    """Bonus when the charger is near the target or inside a broad drawer zone."""
    charger_pos = env.scene["charger"].data.root_pos_w
    target_pos = _drawer_target_pos_w(env, target_offset)
    distance = torch.norm(target_pos - charger_pos, dim=-1, p=2)
    is_near_target = distance < threshold
    is_in_zone = _charger_in_drawer_zone(env, target_offset, zone_half_extents)
    return (is_near_target | is_in_zone).float()


def close_drawer_after_charger(
    env: ManagerBasedRLEnv,
    charger_threshold: float,
    asset_cfg: SceneEntityCfg,
    drawer_target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET,
    zone_half_extents: tuple[float, float, float] = (0.34, 0.28, 0.22),
) -> torch.Tensor:
    """Reward closing the drawer once the charger is inside it."""
    charger_is_in_drawer = charger_in_drawer_bonus(
        env, charger_threshold, drawer_target_offset, zone_half_extents
    )
    drawer_pos = _drawer_joint_pos(env, asset_cfg)
    close_reward = 1.0 - torch.clamp(drawer_pos / 0.39, min=0.0, max=1.0)
    return charger_is_in_drawer * close_reward


def drawer_closed_with_charger(
    env: ManagerBasedRLEnv,
    charger_threshold: float,
    drawer_closed_threshold: float,
    asset_cfg: SceneEntityCfg,
    drawer_target_offset: tuple[float, float, float] = DEFAULT_DRAWER_TARGET_OFFSET,
    zone_half_extents: tuple[float, float, float] = (0.34, 0.28, 0.22),
    placement_zone_half_extents: tuple[float, float, float] = (0.30, 0.24, 0.07),
    drawer_open_threshold: float = 0.25,
    pickup_height_delta: float = 0.08,
    print_success_stages: bool = True,
    print_stage3_box_debug: bool = False,
    stage3_debug_interval: int = 30,
    draw_stage3_box_debug: bool = False,
    draw_randomization_debug: bool = False,
    charger_randomization_center: tuple[float, float, float] = (0.30, 0.42, 0.445),
    charger_randomization_half_extents: tuple[float, float, float] = (0.08, 0.12, 0.01),
    cabinet_randomization_center: tuple[float, float, float] = (0.94, 0.0, 0.26),
    cabinet_randomization_half_extents: tuple[float, float, float] = (0.09, 0.09, 0.01),
) -> torch.Tensor:
    """Four-stage success: open drawer, pick charger, place charger, close drawer."""
    stages = _get_success_stage_state(env)
    drawer_pos = _drawer_joint_pos(env, asset_cfg)
    charger_z = env.scene["charger"].data.root_pos_w[:, 2]

    drawer_opened_now = drawer_pos > drawer_open_threshold
    charger_picked_now = charger_z > env._open_drawer_charger_start_z + pickup_height_delta
    charger_placed_now = _charger_placed_in_drawer(
        env, drawer_target_offset, placement_zone_half_extents
    )
    debug_stage3 = ~stages["charger_placed"]
    if torch.any(debug_stage3):
        _print_stage3_box_debug(
            env,
            drawer_target_offset,
            placement_zone_half_extents,
            print_stage3_box_debug,
            stage3_debug_interval,
        )
    _draw_open_drawer_debug_boxes(
        env,
        drawer_target_offset,
        placement_zone_half_extents,
        draw_stage3_box_debug,
        draw_randomization_debug,
        charger_randomization_center,
        charger_randomization_half_extents,
        cabinet_randomization_center,
        cabinet_randomization_half_extents,
    )
    drawer_closed_now = drawer_pos < drawer_closed_threshold

    stage1_new = drawer_opened_now & ~stages["drawer_opened"]
    stages["drawer_opened"] |= drawer_opened_now

    stage2_new = stages["drawer_opened"] & charger_picked_now & ~stages["charger_picked"]
    stages["charger_picked"] |= stages["drawer_opened"] & charger_picked_now

    stage3_new = stages["charger_picked"] & charger_placed_now & ~stages["charger_placed"]
    stages["charger_placed"] |= stages["charger_picked"] & charger_placed_now

    stage4_new = stages["charger_placed"] & drawer_closed_now & ~stages["drawer_closed"]
    stages["drawer_closed"] |= stages["charger_placed"] & drawer_closed_now

    # Keep the stage latches strictly ordered if they are reused by recorders or debug tooling.
    stages["charger_picked"] &= stages["drawer_opened"]
    stages["charger_placed"] &= stages["charger_picked"]
    stages["drawer_closed"] &= stages["charger_placed"]

    _print_new_stage("stage 1 drawer opened", stage1_new, print_success_stages)
    _print_new_stage("stage 2 charger picked up", stage2_new, print_success_stages)
    _print_new_stage("stage 3 charger placed in drawer", stage3_new, print_success_stages)
    _print_new_stage("stage 4 drawer closed with charger inside", stage4_new, print_success_stages)

    return stages["drawer_closed"]


# Backward-compatible aliases for older configs that still say "cube".
def approach_ee_cube(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    return approach_ee_charger(env, threshold)


def cube_lifted(env: ManagerBasedRLEnv, minimal_height: float) -> torch.Tensor:
    return charger_lifted(env, minimal_height)


def cube_to_drawer(env: ManagerBasedRLEnv, drawer_open_threshold: float) -> torch.Tensor:
    return charger_to_drawer(env, drawer_open_threshold)


def cube_in_drawer_bonus(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    return charger_in_drawer_bonus(env, threshold)


def close_drawer_after_cube(env: ManagerBasedRLEnv, cube_threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return close_drawer_after_charger(env, cube_threshold, asset_cfg)


def drawer_closed_with_cube(
    env: ManagerBasedRLEnv, cube_threshold: float, drawer_closed_threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    return drawer_closed_with_charger(env, cube_threshold, drawer_closed_threshold, asset_cfg)
