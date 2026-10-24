# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strategy-diverse, resumable motion planner for ``OpenDrawer-ManagerBased-v0``.

Scripted stand-in for the spacemouse operator of ``record_demos.py`` / ``record_dagger.py``. It does the task the
way the teleop demos do: grasp the top-drawer handle from the front (fingers closing vertically around the bar),
pull the drawer open, release, grasp the charger top-down across its short side, lift it over the drawer rim,
lower it into the open drawer and release, then push the drawer shut with the fingers.

Control. Every step commands the env's relative-IK action ``[dx, dy, dz, drx, dry, drz, gripper]`` (the space
the teleop device and the policy use) toward a target pose, closed loop, like the gear-mesh
``ScrewMotionPlanner``: the TCP moves along a straight line and the hand along the rotation geodesic. The action
is a P-controller on the pose error, capped by a per-stage speed, then clipped per axis to the spacemouse range
of the teleop demos (0.2 translation, 0.5 rotation), so planner actions stay inside the action distribution the
policy was trained on. Measured on this env: an action of 0.2 moves the TCP ~8 mm/step, 0.5 rotates it
~1.15 deg/step (``POS_GAIN``/``ROT_GAIN``).

The planner is a generator (``OpenDrawerPlanner.actions()``) that yields one action per env step and re-reads the
live scene between steps. Collection scripts drive it from reset; the DAgger recorder pulls one action per step
while the expert is in control and simply drops the generator when control returns to the policy.

Diversity. ``StrategyParams`` is sampled per episode (per takeover for DAgger) and drives everything that does not
change what success means: handle approach tilt/yaw/roll, grasp point on the bar, pre-grasp standoff, how far the
drawer is pulled, retreats, charger grasp point/tilt/yaw jitter, pre-grasp height, lift detour, carry height and
hand yaw, where in the drawer the charger is released and from how high, the push point/tilt/hand yaw and whether
the fingers are open while pushing, via-points on the transits, speeds and gains, and short dwell pauses after
gripper changes. Ranges are the module constants below, set from the 100 teleop demos in
``dataset/new_opendrawer_100`` (quoted next to each range).

The strategy is deliberately independent of the scene, as in ``maniskill/scripts/plug_charger_diverse_solver.py``:
the scene seed only drives the env's cabinet/charger randomisation, while strategies come from a separate
module-level RNG stream (``seed_strategy_rng`` / ``next_strategy_seed``). Every executed ``StrategyParams`` is
appended to ``used_strategies``; the scripts drain it per episode (``take_used_strategies``) and store it with the
episode, so a demo can be re-run with ``OpenDrawerPlanner(env, StrategyParams(**d))``.

Resumable. ``actions()`` decides what is left from the live scene -- the env's stage latches
(``env._open_drawer_success_stages``) plus the physical state: what is in the gripper, how far the drawer is open,
where the charger is -- and plans only the remaining stages. If the policy already holds the handle, it keeps
pulling; if it holds the charger, it carries it; if the drawer slid shut before the charger went in, it reopens it.

Expert mode (DAgger takeovers, ``expert=True``). Motion follows ``set_expert_mode``: "demo" (default here) moves
like the demos -- the strategy's sampled speeds, gains, dwell pauses and via points, full gripper openings -- so
DAgger data looks like the training demos; "fast" is the earlier DAgger expert (top of the teleop speed range, no
pauses or via points, partial openings). In both modes the expert corrects from where the policy left the scene
instead of backing out to a stage's starting pose (``EXPERT_IN_PLACE`` below): it goes straight for the handle when
the hand is already near it (grasping the bar where the hand is), descends straight onto the charger when hovering
over it, lowers a charger that is already over the drawer's release area from where it is, and pushes a drawer
whose front the hand is already at from there. Hand-back is the DAgger loop's business: it stops pulling actions on
the step control returns, whatever the planner was doing.

Geometry. The env's ``cabinet_frame`` handle target is not used: its (0.305, 0, 0.01) offset ignores the cabinet's
0.65 scale and sits ~11 cm in front of the real handle. Positions here are taken in the ``drawer_top`` body frame
(x out of the cabinet, z up; the drawer slides along x), measured from the scaled USD. The TCP is the ``ee_tcp``
frame of ``ee_frame`` looked up by name: which of ``ee_tcp``/``gripper`` sits at index 0 (the env's ``ee_pose``)
changes from process to process.
"""

from __future__ import annotations

import math
import os
import secrets
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_inv,
    quat_mul,
)

# ---- env constants (measured on OpenDrawer-ManagerBased-v0) -------------------------------------------------------
POS_GAIN = 0.04  # m of TCP motion per step per unit translation action (steady state)
ROT_GAIN = 0.04  # rad of hand rotation per step per unit rotation action
MAX_POS_ACTION = 0.2  # per-axis spacemouse saturation in the teleop demos
MAX_ROT_ACTION = 0.5
OPEN, CLOSE = 1.0, -1.0
FULL_OPEN_GAP = 0.08  # finger gap (sum of both finger joints) when open

# drawer_top body frame (x out of the cabinet, z up). The cabinet front sits at x = DRAWER_FRONT_X_B - d.
HANDLE_CENTER_B = (0.19, 0.0, 0.0062)  # handle (bar + brackets) centre; the bar's front face is at x = 0.2025
DRAWER_FRONT_X_B = 0.1775  # outer face of the drawer front panel
# Handle grasp geometry. The fingertips sit FINGER_REACH from the TCP along the hand's y axis (the closing axis);
# with the hand tilted down by t the upper fingertip leads the TCP toward the cabinet by FINGER_REACH * sin(t). The
# grasp keeps that tip frame out of FACE_CLEAR_X_B (finger thickness + margin off the drawer face), puts the closing
# line through the bar (crossing the bar's centre height inside BAR_LINE_X_RANGE) and keeps the upper finger above
# the bar (TCP at most HANDLE_DROP_MAX below the crossing, along the closing axis). Teleop demos end with the TCP
# 1.2 cm in front of and 1.2 cm below the handle centre, which is this geometry at their 14 deg median tilt.
FINGER_REACH = 0.04
FACE_CLEAR_X_B = 0.1905
BAR_LINE_X_RANGE = (0.194, 0.2)
HANDLE_DROP_MAX = 0.024
DRAWER_RIM_Z_B = 0.0398  # top of the drawer walls
CHARGER_REST_Z_B = -0.0132  # charger root height when lying on the drawer floor
PLACE_FRONT_X_B = 0.115  # farthest-out charger centre that clears the drawer front wall
PLACE_CABINET_MARGIN = (
    0.11  # charger centre this far out of the cabinet front: the hand is ~20 cm wide across its fingers
)
# Horizontal distance from the robot base beyond which a hand pointing down runs the elbow straight (q4 at its
# upper limit); far cabinets put the back of the open drawer past it, so the placement slides toward the robot.
PLACE_MAX_RADIUS = 0.57
PUSH_OVERSHOOT = 0.03  # push target this far past the closed front face (contact stops the hand first)

# charger root frame: 4.2 x 5.5 x 2.7 cm, long axis along y, geometric centre 1.24 cm along +y from the root
CHARGER_CENTER_C = (0.0, 0.0124, 0.0)

# stage thresholds (the success term: drawer opened > 0.18, closed < 0.01)
D_MIN_PLACE = 0.18  # drawer must be open this far before the charger can go in
D_CLOSED = 0.003
HANDLE_GAP_RANGE = (0.004, 0.035)  # finger gap when closed on the 1.5 cm handle bar
CHARGER_GAP_RANGE = (0.025, 0.07)  # finger gap when closed across the 4.2 cm charger
Q7_MARGIN = 0.3  # rad kept from the wrist joint limits when choosing a hand yaw

# ---- sampling ranges (teleop demos in comments) -------------------------------------------------------------------
# Handle: approach 6-27 deg below horizontal (median 14), yaw +-8 deg, fingers within ~9 deg of vertical, TCP
# within +-2 cm along the 11 cm bar. Near-level approaches pin the wrist-pitch joint (teleop demos sit at q6 = 3.71
# of 3.75), steep ones leave no room between the leading fingertip and the drawer face (see FACE_CLEAR_X_B); the
# tilt actually reached is what the grasp point is computed from. The IK drifts the wrist-roll joint q7 toward its
# upper limit on the way to the handle, and roll about the approach axis adds to q7 one for one, so the roll range
# leans negative. handle_line_u / handle_drop_u pick the grasp inside the feasible window (see _handle_grasp_point).
HANDLE_PITCH_DEG_RANGE = (12.0, 20.0)  # steeper pinches wedge a finger off the bar when pulling (slipped at 21 deg)
# Yaw is measured from the direction the arm points (robot base -> handle), not from straight into the cabinet: a
# level approach drives the wrist through its singularity (q6 = pi, joints 5 and 7 aligned), where the one rotation
# the wrist cannot make is a yaw against the arm's direction. The IK makes up any such yaw with a large q5/q7
# swing, which flipped the wrist in 3 of 12 episodes (yaw 6-14 deg off the arm's direction). Teleop demos stay
# within +-7.5 deg of it.
HANDLE_YAW_DEG_MAX = 3.0
HANDLE_ROLL_DEG_RANGE = (-5.0, 4.0)
HANDLE_LATERAL_MAX = 0.03
HANDLE_STANDOFF_RANGE = (0.05, 0.12)
# Pull: drawer open 0.25-0.31 at release (median 0.28), pulled at ~3 mm/step (0 -> 0.18 in ~65 steps); faster pulls
# slip off the bar. Near cabinets fold the elbow into its limit first; the pull then stalls and stops wherever it
# got (>= 0.18 is enough).
OPEN_TARGET_RANGE = (0.24, 0.32)
PULL_SPEED_RANGE = (0.05, 0.11)
# Let go: back off just enough to clear the bar, then up while pitching the hand down. Backing straight out with a
# level hand folds the elbow into its limit (teleop demos hit q4 = -3.07 here too).
HANDLE_RETREAT_RANGE = (0.03, 0.06)
HANDLE_RETREAT_UP_RANGE = (0.05, 0.12)
HANDLE_RETREAT_PITCH_DEG_RANGE = (20.0, 50.0)
# Charger: top-down within ~15 deg, hand yaw ~90 deg off the charger yaw (fingers across its 4.2 cm side), within
# ~3 cm of its centre.
CHARGER_TILT_DEG_MAX = 8.0
CHARGER_YAW_NOISE_DEG = 12.0
CHARGER_LONG_OFFSET_MAX = 0.012
CHARGER_GRASP_Z_RANGE = (0.0, 0.006)  # TCP above the charger's centre height
PREGRASP_HEIGHT_RANGE = (0.05, 0.12)
# Carry: released just above the drawer rim (charger centre 3 cm over it, median).
CARRY_CLEARANCE_RANGE = (0.05, 0.10)  # charger centre above the drawer rim while carrying
LIFT_LATERAL_MAX = 0.04
CARRY_YAW_DEG_MAX = 30.0
# Hand tipped forward (a pitch in the arm's plane) while carrying and lowering the charger. Lowering into the back of
# a far drawer with the hand straight down runs q6 toward 0, the wrist's other singularity, where the IK swung the
# elbow ~50 deg for a 5 cm move and the open hand then could not rise out of the drawer. Teleop charger grasps are
# tilted up to ~15 deg.
CARRY_TILT_DEG_RANGE = (5.0, 20.0)
PLACE_Y_MAX = 0.10
RELEASE_HEIGHT_RANGE = (0.015, 0.06)  # charger centre above its resting height at release
PLACE_RETREAT_RANGE = (0.06, 0.12)  # hand above the drawer rim after release
# Close: hand pointing down, tilted 0-55 deg toward the cabinet, gripper open (all 100 teleop demos). The closed
# drawer front of a far cabinet is ~0.8 m out, reachable only with the hand tilted well forward (its wrist then sits
# behind the TCP). The tilt is a pitch in the arm's own plane, so the wrist never has to yaw near its singularity.
PUSH_TILT_DEG_RANGE = (25.0, 50.0)  # tilt reached as the drawer shuts; the push starts at PUSH_TILT_START_DEG
# Starting the push already tilted folds the elbow (the open drawer's front is close to the robot) and the IK lifts
# the arm instead: the hand starts nearly vertical and leans forward as the front moves away, like a person would.
PUSH_TILT_START_DEG = 8.0
PUSH_Y_MAX = 0.10
# TCP height on the drawer front, below the handle bar (bar z -0.001..0.014) as in the teleop demos (-0.015 median).
# Pushing at the bar's height let the open fingers slip over it into the gap behind it and jam short of closed.
PUSH_Z_RANGE = (-0.028, -0.014)
PUSH_STANDOFF_RANGE = (0.04, 0.09)
PUSH_SPEED_RANGE = (0.08, 0.2)
PUSH_YAW_NOISE_DEG = 15.0
PUSH_BACKOFF_RANGE = (0.03, 0.07)
# Motion: action-norm caps (teleop p50 0.11, p90 0.18) and P gain (fraction of the error closed per step; the
# arm lags ~1 step, >0.35 starts to overshoot).
SPEED_TRANSIT_RANGE = (0.14, 0.26)
SPEED_CAREFUL_RANGE = (0.06, 0.14)
ROT_SPEED_RANGE = (0.35, 0.5)
GAIN_RANGE = (0.2, 0.35)
VIA_PROB = 0.5  # per transit: go through a jittered midpoint first
VIA_LATERAL_MAX = 0.04
VIA_UP_MAX = 0.06
DWELL_MAX = 4  # extra stationary steps after a gripper change (teleop: median 1, p95 7)

# ---- DAgger-expert overrides (expert=True only; demos are unaffected) --------------------------------------------
# Speeds of the "fast" expert (set_expert_mode("fast"), the earlier DAgger expert): it moved at once and at the top
# of the teleop range and never paused beyond what the fingers need. The default "demo" expert uses the strategy's
# sampled values instead. Set to None to use the sampled value in "fast" mode too.
EXPERT_SPEED_TRANSIT = 0.28
EXPERT_SPEED_CAREFUL = 0.16
EXPERT_PULL_SPEED = 0.11  # top of the pull range: at 0.2 (8 mm/step) the fingers slid off the bar
EXPERT_PUSH_SPEED = 0.2
EXPERT_ROT_SPEED = 0.7  # rotation-vector norm; each axis stays clipped to the teleop's 0.5 (teleop max norm 0.64)
EXPERT_GAIN = 0.3
EXPERT_SKIP_PREGRASP_IF_ALIGNED = True
# "demo": the expert moves like the demos (the EXPERT_* speeds above, zero dwell, no via points and partial gripper
# openings apply only in "fast"). See set_expert_mode.
EXPERT_DEMO_MOTION = True
# In-place corrections (expert only, both modes): the expert corrects from where the policy left the hand instead of
# backing out to the start of the stage, and keeps the policy's grasp / release / push point when it lies inside the
# range the demos sample it from. With EXPERT_IN_PLACE = False the expert behaves like the earlier DAgger expert
# (only the handle check with the demos' pre-grasp tolerances, 2 cm and 20 deg, and the charger hover descent).
EXPERT_IN_PLACE = True
# Handle: go straight for the grasp when the TCP is between the bar and EXPERT_HANDLE_NEAR_ALONG in front of it,
# within EXPERT_HANDLE_NEAR_LATERAL of the approach line and within EXPERT_HANDLE_NEAR_ROT_DEG of the grasp
# orientation; the grasp keeps the TCP's offset along the bar if that is within HANDLE_LATERAL_MAX.
EXPERT_HANDLE_NEAR_ALONG = 0.15
EXPERT_HANDLE_NEAR_LATERAL = 0.05
EXPERT_HANDLE_NEAR_ROT_DEG = 30.0
# Place: holding the charger above the drawer rim, over the area the demos release it in (PLACE_* ranges, plus this
# margin): lower it from where it is instead of carrying it up and over to a sampled release point.
EXPERT_PLACE_MARGIN = 0.01
# Push: the hand already at the drawer front (between the face and the farthest push standoff, within the push
# point ranges plus this margin): push from there, at the hand's own point along the face and height.
EXPERT_PUSH_MARGIN = 0.02
# Rotation is capped at the teleop's 1.15 deg/step, so a 16-step window buys ~18 deg of turn: the expert grasps the
# charger with the orientation nearest the one it already has instead of a sampled one -- its current hand yaw if
# that is within EXPERT_GRASP_YAW_TOL_DEG of across-the-4.2-cm-side (4.2 cos25 + 5.5 sin25 = 6.1 cm still fits the
# 8 cm opening), and its current tilt up to EXPERT_GRASP_TILT_MAX_DEG. When it is over the charger (within
# EXPERT_HOVER_LATERAL_MAX, at least EXPERT_DESCEND_FROM - 1 cm above it) it goes straight to EXPERT_DESCEND_FROM
# above the grasp point, turning on the way, then descends. Measured before this: 0/9 hand-backs in the pick stage
# had the charger in hand (the expert was still turning toward a sampled yaw).
EXPERT_GRASP_YAW_TOL_DEG = 25.0
EXPERT_GRASP_TILT_MAX_DEG = 12.0
EXPERT_HOVER_LATERAL_MAX = 0.12
EXPERT_DESCEND_FROM = 0.035
EXPERT_GRASP_ROT_TOL = 0.12  # rad left unturned when the expert closes on the charger (the grasp tolerates ~25 deg)
# A hand left this far from its target orientation by a joint limit means the arm is in a configuration the task
# cannot be done cleanly from: the plan fails rather than recovering (teleop demos never recover either).
STUCK_FAIL_DEG = 30.0
# Same for a move that stalls this far short of its target: the arm lurched into a singularity or caught on the
# cabinet. The short back-off from the handle is exempt (the folded elbow stops it, harmlessly).
STALL_FAIL_M = 0.05


@dataclass(frozen=True)
class StrategyParams:
    seed: int
    handle_pitch_deg: float
    handle_yaw_deg: float
    handle_roll_deg: float
    handle_lateral: float
    handle_line_u: float  # 0..1: where the closing line crosses the bar, within the feasible window
    handle_drop_u: float  # 0..1: how far below that crossing the TCP sits, within the feasible window
    handle_standoff: float
    open_target: float
    pull_speed: float
    handle_retreat: float
    handle_retreat_up: float
    handle_retreat_pitch_deg: float
    charger_tilt_x_deg: float
    charger_tilt_y_deg: float
    charger_yaw_noise_deg: float
    charger_long_offset: float
    charger_grasp_z: float
    pregrasp_height: float
    carry_clearance: float
    lift_dx: float
    lift_dy: float
    carry_yaw_deg: float
    carry_tilt_deg: float
    place_u: float  # 0..1 across the free depth of the open drawer, cabinet side -> front wall
    place_y: float
    release_height: float
    place_retreat: float
    push_tilt_deg: float
    push_y: float
    push_z: float
    push_standoff: float
    push_speed: float
    push_yaw_noise_deg: float
    push_backoff: float
    speed_transit: float
    speed_careful: float
    rot_speed: float
    gain: float
    dwell_steps: int

    @classmethod
    def sample(cls, seed: int) -> StrategyParams:
        rng = np.random.default_rng(int(seed))

        def u(lo_hi):
            return float(rng.uniform(*lo_hi))

        def pm(m):
            return float(rng.uniform(-m, m))

        return cls(
            seed=int(seed),
            handle_pitch_deg=u(HANDLE_PITCH_DEG_RANGE),
            handle_yaw_deg=pm(HANDLE_YAW_DEG_MAX),
            handle_roll_deg=u(HANDLE_ROLL_DEG_RANGE),
            handle_lateral=pm(HANDLE_LATERAL_MAX),
            handle_line_u=float(rng.uniform(0.0, 1.0)),
            handle_drop_u=float(rng.uniform(0.0, 1.0)),
            handle_standoff=u(HANDLE_STANDOFF_RANGE),
            open_target=u(OPEN_TARGET_RANGE),
            pull_speed=u(PULL_SPEED_RANGE),
            handle_retreat=u(HANDLE_RETREAT_RANGE),
            handle_retreat_up=u(HANDLE_RETREAT_UP_RANGE),
            handle_retreat_pitch_deg=u(HANDLE_RETREAT_PITCH_DEG_RANGE),
            charger_tilt_x_deg=pm(CHARGER_TILT_DEG_MAX),
            charger_tilt_y_deg=pm(CHARGER_TILT_DEG_MAX),
            charger_yaw_noise_deg=pm(CHARGER_YAW_NOISE_DEG),
            charger_long_offset=pm(CHARGER_LONG_OFFSET_MAX),
            charger_grasp_z=u(CHARGER_GRASP_Z_RANGE),
            pregrasp_height=u(PREGRASP_HEIGHT_RANGE),
            carry_clearance=u(CARRY_CLEARANCE_RANGE),
            lift_dx=pm(LIFT_LATERAL_MAX),
            lift_dy=pm(LIFT_LATERAL_MAX),
            carry_yaw_deg=pm(CARRY_YAW_DEG_MAX),
            carry_tilt_deg=u(CARRY_TILT_DEG_RANGE),
            place_u=float(rng.uniform(0.0, 1.0)),
            place_y=pm(PLACE_Y_MAX),
            release_height=u(RELEASE_HEIGHT_RANGE),
            place_retreat=u(PLACE_RETREAT_RANGE),
            push_tilt_deg=u(PUSH_TILT_DEG_RANGE),
            push_y=pm(PUSH_Y_MAX),
            push_z=u(PUSH_Z_RANGE),
            push_standoff=u(PUSH_STANDOFF_RANGE),
            push_speed=u(PUSH_SPEED_RANGE),
            push_yaw_noise_deg=pm(PUSH_YAW_NOISE_DEG),
            push_backoff=u(PUSH_BACKOFF_RANGE),
            speed_transit=u(SPEED_TRANSIT_RANGE),
            speed_careful=u(SPEED_CAREFUL_RANGE),
            rot_speed=u(ROT_SPEED_RANGE),
            gain=u(GAIN_RANGE),
            dwell_steps=int(rng.integers(0, DWELL_MAX + 1)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---- strategy RNG stream (independent of the env's scene seed) ----------------------------------------------------

_strategy_rng = np.random.default_rng()  # unseeded: fresh entropy per process
used_strategies: list[dict] = []  # every StrategyParams executed since the last take_used_strategies()


def seed_strategy_rng(seed: int | None) -> int:
    """Make the sequence of strategies reproducible for this process (``--strategy_seed`` on the scripts).
    ``None`` picks a random seed. Returns the seed actually used, so callers can print it."""
    global _strategy_rng
    if seed is None:
        seed = secrets.randbits(31)
    _strategy_rng = np.random.default_rng(int(seed))
    return int(seed)


def next_strategy_seed() -> int:
    return int(_strategy_rng.integers(2**31))


def take_used_strategies() -> list[dict]:
    """Drain the strategies executed since the last call (one per demo, one per DAgger takeover)."""
    out = list(used_strategies)
    used_strategies.clear()
    return out


def pick_strategy(strategy: StrategyParams | None = None) -> StrategyParams:
    params = strategy if strategy is not None else StrategyParams.sample(next_strategy_seed())
    used_strategies.append(params.to_dict())
    return params


def set_expert_mode(mode: str) -> None:
    """How a DAgger takeover moves: "demo" (the strategy's demo speeds, pauses and via points) or "fast" (the earlier
    DAgger expert: top-of-range speeds, no pauses or via points). In-place corrections apply in both."""
    global EXPERT_DEMO_MOTION
    if mode not in ("demo", "fast"):
        raise ValueError(f"expert mode must be demo or fast, got {mode!r}")
    EXPERT_DEMO_MOTION = mode == "demo"


def describe_expert_overrides() -> str:
    in_place = f"in-place corrections {'on' if EXPERT_IN_PLACE else 'off'}"
    if EXPERT_DEMO_MOTION:
        return f"demo motion (sampled speeds, dwell and via points); {in_place}"
    return (
        f"fast motion: speed_transit={EXPERT_SPEED_TRANSIT} speed_careful={EXPERT_SPEED_CAREFUL} "
        f"pull_speed={EXPERT_PULL_SPEED} push_speed={EXPERT_PUSH_SPEED} rot_speed={EXPERT_ROT_SPEED} "
        f"gain={EXPERT_GAIN} dwell=0 via=off; {in_place}"
    )


# ---- small math helpers -------------------------------------------------------------------------------------------


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v).clamp_min(1e-9)


def _about_world_z(quat: torch.Tensor, angle: float) -> torch.Tensor:
    z = torch.zeros(1, device=quat.device)
    return quat_mul(quat_from_euler_xyz(z, z, z + angle), quat.unsqueeze(0))[0]


def _down_quat(yaw: float, tilt_x: float = 0.0, tilt_y: float = 0.0, device="cpu") -> torch.Tensor:
    """Hand pointing down (z_ee = -z) with its x axis at ``yaw``, then tilted by small world-frame angles."""
    z = torch.zeros(1, device=device)
    down = quat_from_euler_xyz(z + math.pi, z, z + yaw)
    tilt = quat_from_euler_xyz(z + tilt_x, z + tilt_y, z)
    return quat_mul(tilt, down)[0]


def _azimuth(p: torch.Tensor) -> float:
    """Direction of a point seen from the robot base (at the env origin), about world z."""
    return math.atan2(float(p[1]), float(p[0]))


class PlanFailed(Exception):
    """A missed or slipped grasp, a stage that did not complete, or a hand stuck far from where it was sent."""


# ---- planner ------------------------------------------------------------------------------------------------------

Action = torch.Tensor
Plan = Generator[Action, None, bool]
Target = torch.Tensor | Callable[[], torch.Tensor]


class OpenDrawerPlanner:
    """Closed-loop, resumable planner for one episode (or one DAgger takeover)."""

    STAGE_IDS = {
        "open_drawer": 0,
        "pull_drawer": 1,
        "release_handle": 2,
        "put_down_charger": 3,
        "pick_charger": 4,
        "place_charger": 5,
        "close_drawer": 6,
        "done": 7,
    }

    def __init__(self, env, params: StrategyParams, expert: bool = False, trace: bool | None = None):
        self.env = env
        self.p = params
        self.expert = expert
        self.trace_enabled = bool(os.environ.get("OPEN_DRAWER_PLANNER_DEBUG")) if trace is None else trace
        self.device = env.device
        self.rng = np.random.default_rng(params.seed + 1)  # via-point jitter, reproducible from the strategy
        self.stage = "init"
        self.steps = 0
        self.failure: str | None = None  # why the plan stopped, if it failed

        self._fast = expert and not EXPERT_DEMO_MOTION  # see set_expert_mode

        def pick(override, sampled):
            return override if (self._fast and override is not None) else sampled

        self.speed_transit = pick(EXPERT_SPEED_TRANSIT, params.speed_transit)
        self.speed_careful = pick(EXPERT_SPEED_CAREFUL, params.speed_careful)
        self.pull_speed = pick(EXPERT_PULL_SPEED, params.pull_speed)
        self.push_speed = pick(EXPERT_PUSH_SPEED, params.push_speed)
        self.rot_speed = pick(EXPERT_ROT_SPEED, params.rot_speed)
        self.gain = pick(EXPERT_GAIN, params.gain)
        self.dwell = 0 if self._fast else params.dwell_steps

        # Look the TCP up by name: the FrameTransformer keeps the two panda_hand frames (ee_tcp, gripper) in a set, so
        # which one lands at index 0 -- the env's "ee_pose" -- changes from process to process.
        self._tcp_index = env.scene["ee_frame"].data.target_frame_names.index("ee_tcp")
        robot = env.scene["robot"]
        self._robot = robot
        self._finger_ids = [i for i, n in enumerate(robot.joint_names) if "finger" in n]
        self._q7_id = robot.joint_names.index("panda_joint7")
        self._q7_limits = robot.data.soft_joint_pos_limits[0, self._q7_id].tolist()
        cabinet = env.scene["cabinet"]
        self._cabinet = cabinet
        self._drawer_body = cabinet.body_names.index("drawer_top")
        self._drawer_joint = cabinet.joint_names.index("drawer_top_joint")
        self.gripper = OPEN if self.finger_gap() > FULL_OPEN_GAP - 0.01 else CLOSE

    # -- live state -------------------------------------------------------------------------------------------------
    def _origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[0]

    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee = self.env.scene["ee_frame"]
        i = self._tcp_index
        return (ee.data.target_pos_w[0, i] - self._origin()).clone(), ee.data.target_quat_w[0, i].clone()

    def finger_gap(self) -> float:
        return float(self._robot.data.joint_pos[0, self._finger_ids].sum())

    def q7(self) -> float:
        return float(self._robot.data.joint_pos[0, self._q7_id])

    def drawer_open(self) -> float:
        return float(self._cabinet.data.joint_pos[0, self._drawer_joint])

    def drawer_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._cabinet.data.body_pos_w[0, self._drawer_body] - self._origin()
        return pos.clone(), self._cabinet.data.body_quat_w[0, self._drawer_body].clone()

    def to_world(self, p_b) -> torch.Tensor:
        pos, quat = self.drawer_frame()
        p_b = torch.as_tensor(p_b, device=self.device, dtype=torch.float32)
        return pos + quat_apply(quat.unsqueeze(0), p_b.unsqueeze(0))[0]

    def to_drawer(self, p_w: torch.Tensor) -> torch.Tensor:
        pos, quat = self.drawer_frame()
        return quat_apply(quat_inv(quat.unsqueeze(0)), (p_w - pos).unsqueeze(0))[0]

    def open_dir(self) -> torch.Tensor:
        """World direction the drawer slides when opening (out of the cabinet)."""
        _, quat = self.drawer_frame()
        return quat_apply(quat.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=self.device))[0]

    def closed_to_world(self, p_b) -> torch.Tensor:
        """A drawer-frame point as it will be once the drawer is closed (fixed w.r.t. the cabinet)."""
        return self.to_world(p_b) - self.drawer_open() * self.open_dir()

    def rim_z(self) -> float:
        return float(self.to_world((0.0, 0.0, DRAWER_RIM_Z_B))[2])

    def charger_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        charger = self.env.scene["charger"]
        return (charger.data.root_pos_w[0] - self._origin()).clone(), charger.data.root_quat_w[0].clone()

    def charger_center(self) -> torch.Tensor:
        pos, quat = self.charger_pose()
        offset = torch.tensor([CHARGER_CENTER_C], device=self.device)
        return pos + quat_apply(quat.unsqueeze(0), offset)[0]

    def latches(self) -> dict[str, bool]:
        stages = getattr(self.env, "_open_drawer_success_stages", None)
        names = ("drawer_opened", "charger_picked", "charger_placed", "drawer_closed")
        # Right after a reset the latches still hold the previous episode until the success term runs again.
        if stages is None or int(self.env.episode_length_buf[0]) == 0:
            return dict.fromkeys(names, False)
        return {name: bool(stages[name][0]) for name in names}

    def charger_start_z(self) -> float:
        start = getattr(self.env, "_open_drawer_charger_start_z", None)
        if start is None or int(self.env.episode_length_buf[0]) == 0:
            return float(self.charger_pose()[0][2])
        return float(start[0])

    def handle_center(self) -> torch.Tensor:
        return self.to_world(HANDLE_CENTER_B)

    def holding_handle(self) -> bool:
        gap = self.finger_gap()
        tcp, _ = self.tcp_pose()
        rel = self.to_drawer(tcp) - torch.tensor(HANDLE_CENTER_B, device=self.device)
        # the grasp puts the TCP up to ~2.3 cm below the bar (see _handle_grasp_point) and it sags while pulling
        near = abs(float(rel[0])) < 0.04 and abs(float(rel[1])) < 0.07 and abs(float(rel[2])) < 0.045
        return near and HANDLE_GAP_RANGE[0] < gap < HANDLE_GAP_RANGE[1]

    def _handle_slipping(self) -> bool:
        """Fingers starting to slide off the bar: the pinch narrows or the TCP creeps out past the bar's front
        (clean pulls hold a 1.6-2.1 cm gap with the TCP at x = 0.195-0.204 in the drawer frame)."""
        tcp, _ = self.tcp_pose()
        return self.finger_gap() < 0.014 or float(self.to_drawer(tcp)[0]) > HANDLE_CENTER_B[0] + 0.017

    def holding_charger(self) -> bool:
        gap = self.finger_gap()
        tcp, _ = self.tcp_pose()
        near = torch.linalg.norm(tcp - self.charger_center()).item() < 0.035
        return near and CHARGER_GAP_RANGE[0] < gap < CHARGER_GAP_RANGE[1]

    def charger_in_drawer(self) -> bool:
        c = self.to_drawer(self.charger_center())
        return bool(-0.1675 < c[0] < DRAWER_FRONT_X_B and abs(c[1]) < 0.24 and -0.03 < c[2] < DRAWER_RIM_Z_B)

    def charger_reachable(self) -> bool:
        c = self.charger_center()
        # on the table or in the drawer (not on the floor) and within the arm's reach from its base at the origin
        return bool(c[2] > self.charger_start_z() - 0.08 and torch.linalg.norm(c[:2]).item() < 0.8)

    def hand_yaw(self) -> float:
        _, q = self.tcp_pose()
        x_axis = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=self.device))[0]
        return math.atan2(float(x_axis[1]), float(x_axis[0]))

    def _q7_after(self, yaw_delta: float, az_delta: float = 0.0) -> float:
        """Predicted wrist-roll joint after turning the (roughly downward) hand by ``yaw_delta`` about world z while
        the arm swings ``az_delta`` about the base: hand yaw = q1 - q7 + const, and q1 follows the arm's azimuth.
        Moving the hand from the drawer to the charger (~50 deg round the base) without turning it takes q7 past
        its limit from the home pose."""
        return self.q7() + az_delta - yaw_delta

    def _q7_ok(self, yaw_delta: float, az_delta: float = 0.0) -> bool:
        return self._q7_in_range(self._q7_after(yaw_delta, az_delta))

    def _q7_in_range(self, q7: float) -> bool:
        return self._q7_limits[0] + Q7_MARGIN < q7 < self._q7_limits[1] - Q7_MARGIN

    def _pick_yaw(self, candidates: list[float], az_delta: float = 0.0, then_az_delta: float = 0.0) -> float:
        """Among hand yaws, the one needing the least wrist turn that keeps joint 7 inside its limits where the hand
        is going (``az_delta`` round the base) and after a further ``then_az_delta`` swing holding that yaw."""
        current = self.hand_yaw()
        options = sorted(candidates, key=lambda c: abs(_wrap(c - current)))

        def ok(c):
            q7 = self._q7_after(_wrap(c - current), az_delta)
            return self._q7_in_range(q7) and self._q7_in_range(q7 + then_az_delta)

        feasible = [c for c in options if ok(c)]
        return feasible[0] if feasible else options[0]

    # -- control primitives -----------------------------------------------------------------------------------------
    def _trace(self, msg: str) -> None:
        if not self.trace_enabled:
            return
        tcp, _ = self.tcp_pose()
        q = self._robot.data.joint_pos[0, :7]
        lim = self._robot.data.soft_joint_pos_limits[0, :7]
        at_limit = [f"q{j + 1}" for j in range(7) if min(q[j] - lim[j, 0], lim[j, 1] - q[j]) < 0.05]
        print(
            f"[planner{' expert' if self.expert else ''} t={self.steps:4d} {self.stage}] d={self.drawer_open():.3f} "
            f"gap={self.finger_gap() * 100:.1f}cm tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})"
            f"{' limits=' + ','.join(at_limit) if at_limit else ''} {msg}",
            flush=True,
        )

    def _rot_err(self, target_quat: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        dq = quat_mul(target_quat.unsqueeze(0), quat_inv(quat.unsqueeze(0)))
        dq = torch.where(dq[:, 0:1] < 0, -dq, dq)  # shortest arc
        return axis_angle_from_quat(dq)[0]

    def _action(self, dpos: torch.Tensor, drot: torch.Tensor, gripper: float, speed: float) -> Action:
        v = dpos * (self.gain / POS_GAIN)
        n = torch.linalg.norm(v)
        if n > speed:
            v = v * (speed / n)
        w = drot * (self.gain / ROT_GAIN)
        n = torch.linalg.norm(w)
        if n > self.rot_speed:
            w = w * (self.rot_speed / n)
        action = torch.zeros((1, 7), device=self.device)
        action[0, 0:3] = v.clamp(-MAX_POS_ACTION, MAX_POS_ACTION)
        action[0, 3:6] = w.clamp(-MAX_ROT_ACTION, MAX_ROT_ACTION)
        action[0, 6] = gripper
        self.gripper = gripper
        self.steps += 1
        return action

    def servo(
        self,
        target: Target,
        quat: Target | None,
        gripper: float,
        *,
        speed: float | None = None,
        pos_tol: float = 0.004,
        rot_tol: float = 0.05,
        max_steps: int = 250,
        anchor: Callable[[], torch.Tensor] | None = None,
        name: str = "",
        fail_over: float | None = STALL_FAIL_M,
    ) -> Plan:
        """Drive ``anchor`` (default: the TCP) to ``target`` and the hand to ``quat`` (None: keep the current
        orientation). Targets may be callables, re-evaluated every step. Returns True once within tolerance.

        The IK action cannot see joint limits, so a commanded orientation can be unreachable (the wrist joints sit
        at their limits around the handle). When the rotation stops improving the orientation target is frozen at
        the current one and only the translation continues -- what the teleop operator does. When nothing improves
        any more the move gives up early: it returns False if it got within ``fail_over`` of the target (None: any
        distance) and fails the plan otherwise (see STALL_FAIL_M, STUCK_FAIL_DEG)."""
        speed = self.speed_transit if speed is None else speed
        dpos_n = drot_n = float("nan")
        best_pos = best_rot = float("inf")
        pos_idle = rot_idle = 0
        rot_history: list[float] = []
        frozen: torch.Tensor | None = None
        for _ in range(max_steps):
            pos, q = self.tcp_pose()
            tp = target() if callable(target) else target
            dpos = tp - (anchor() if anchor is not None else pos)
            if quat is None:
                drot = torch.zeros(3, device=self.device)
            else:
                drot = self._rot_err(frozen if frozen is not None else (quat() if callable(quat) else quat), q)
            dpos_n, drot_n = torch.linalg.norm(dpos).item(), torch.linalg.norm(drot).item()
            if dpos_n < pos_tol and drot_n < rot_tol:
                return True
            pos_idle = 0 if dpos_n < best_pos - 0.001 else pos_idle + 1
            rot_idle = 0 if drot_n < best_rot - 0.01 else rot_idle + 1
            best_pos, best_rot = min(best_pos, dpos_n), min(best_rot, drot_n)
            rot_history.append(drot_n)
            # a wrist at its limit either stops turning or creeps; 15 steps should turn ~17 deg, under 3 deg is stuck
            creeping = len(rot_history) > 15 and rot_history[-16] - drot_n < 0.05
            if quat is not None and frozen is None and drot_n >= rot_tol and (rot_idle >= 12 or creeping):
                if drot_n > math.radians(STUCK_FAIL_DEG):
                    raise PlanFailed(f"{name}: hand stuck {math.degrees(drot_n):.0f} deg from its target orientation")
                self._trace(
                    f"{name}: rotation stuck {math.degrees(drot_n):.1f} deg off (joint limit), "
                    "keeping the current orientation"
                )
                frozen = q
                continue
            if pos_idle >= 25 and (rot_idle >= 25 or drot_n < rot_tol):
                break
            yield self._action(dpos, drot, gripper, speed)
        if fail_over is not None and dpos_n > fail_over:
            raise PlanFailed(f"{name}: arm stuck {dpos_n * 100:.0f} cm short of its target")
        self._trace(f"{name}: stopped short, residual {dpos_n * 1e3:.1f} mm / {math.degrees(drot_n):.1f} deg")
        return False

    def hold(self, steps: int, gripper: float | None = None) -> Plan:
        """Hold the current pose closed loop (resisting contact) for ``steps`` steps."""
        gripper = self.gripper if gripper is None else gripper
        pos0, q0 = self.tcp_pose()
        for _ in range(steps):
            pos, q = self.tcp_pose()
            yield self._action(pos0 - pos, self._rot_err(q0, q), gripper, self.speed_careful)
        return True

    def actuate(self, gripper: float, *, until_gap: float | None = None, max_steps: int = 25) -> Plan:
        """Command the gripper while holding the pose, until the fingers stop moving (or open past ``until_gap``),
        then dwell ``self.dwell`` extra steps."""
        pos0, q0 = self.tcp_pose()
        prev, still = self.finger_gap(), 0
        for i in range(max_steps):
            pos, q = self.tcp_pose()
            yield self._action(pos0 - pos, self._rot_err(q0, q), gripper, self.speed_careful)
            gap = self.finger_gap()
            still = still + 1 if abs(gap - prev) < 0.0004 else 0
            prev = gap
            if until_gap is not None and gap >= until_gap:
                break
            if i >= 2 and still >= 2:
                break
        yield from self.hold(self.dwell, gripper)
        return True

    def transit(self, target: Target, quat: Target | None, gripper: float, *, anchor=None, name: str = "") -> Plan:
        """Free-space move, optionally through a jittered midpoint (demos, and the expert in demo motion)."""
        if not self._fast and self.rng.random() < VIA_PROB:
            start = anchor() if anchor is not None else self.tcp_pose()[0]
            end = target() if callable(target) else target
            via = 0.5 * (start + end)
            via[0] += float(self.rng.uniform(-VIA_LATERAL_MAX, VIA_LATERAL_MAX))
            via[1] += float(self.rng.uniform(-VIA_LATERAL_MAX, VIA_LATERAL_MAX))
            via[2] = max(float(via[2]), min(float(start[2]), float(end[2]))) + float(self.rng.uniform(0.0, VIA_UP_MAX))
            yield from self.servo(via, quat, gripper, pos_tol=0.02, rot_tol=math.pi, anchor=anchor, name=f"{name} via")
        return (yield from self.servo(target, quat, gripper, rot_tol=0.08, anchor=anchor, name=name))

    # -- stage machine ----------------------------------------------------------------------------------------------
    def next_stage(self) -> str:
        st = self.latches()
        d = self.drawer_open()
        holding_charger = self.holding_charger()
        holding_handle = self.holding_handle()
        if st["drawer_closed"]:
            return "done"
        if st["charger_placed"] or (st["charger_picked"] and self.charger_in_drawer() and not holding_charger):
            return "close_drawer"
        if st["drawer_opened"]:
            if holding_handle:
                return "release_handle"
            if d < D_MIN_PLACE:
                return "put_down_charger" if holding_charger else "open_drawer"
            return "place_charger" if holding_charger else "pick_charger"
        if holding_charger:
            return "put_down_charger"
        if holding_handle:
            return "pull_drawer"
        return "open_drawer"

    def actions(self) -> Iterator[Action]:
        """Yield one action per env step until the task is done (then hold forever) or the plan fails.

        Strict, like the teleop demos: every stage runs once. A missed or slipped grasp, a stage that does not
        complete (it would come round again) or a hand stuck far from its target ends the plan, with the reason in
        ``self.failure``; there are no retries. A DAgger takeover starts a fresh plan from wherever the policy left
        the scene, so the policy's own mistakes are still corrected."""
        attempts: Counter[str] = Counter()
        try:
            while True:
                self.stage = self.next_stage()
                if self.stage == "done":
                    yield from self._finish()
                    while True:
                        yield from self.hold(1)
                attempts[self.stage] += 1
                if attempts[self.stage] > 1:
                    raise PlanFailed(f"{self.stage} did not complete")
                if self.stage in ("pick_charger", "put_down_charger") and not self.charger_reachable():
                    raise PlanFailed("charger out of reach")
                self._trace("start")
                yield from getattr(self, f"_{self.stage}")()
        except PlanFailed as failure:
            self.failure = str(failure)
            self._trace(f"FAILED: {failure}")

    # -- stages -----------------------------------------------------------------------------------------------------
    def _handle_grasp_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World approach axis (into the cabinet, tilted down) and hand orientation for the handle grasp: fingers
        close vertically, one above and one below the bar."""
        p = self.p
        az = _azimuth(self.handle_center())  # the direction the arm points at the handle (see HANDLE_YAW_DEG_MAX)
        a0 = torch.tensor([math.cos(az), math.sin(az), 0.0], device=self.device)
        yaw, pitch, roll = (math.radians(v) for v in (p.handle_yaw_deg, p.handle_pitch_deg, p.handle_roll_deg))
        a = torch.stack(
            [a0[0] * math.cos(yaw) - a0[1] * math.sin(yaw), a0[0] * math.sin(yaw) + a0[1] * math.cos(yaw), a0[2] * 0.0]
        )
        a = _unit(a * math.cos(pitch) + torch.tensor([0.0, 0.0, -math.sin(pitch)], device=self.device))
        x = _unit(torch.linalg.cross(a, torch.tensor([0.0, 0.0, 1.0], device=self.device)))  # along the bar
        y = torch.linalg.cross(a, x)
        x, y = x * math.cos(roll) + y * math.sin(roll), -x * math.sin(roll) + y * math.cos(roll)
        quat = quat_from_matrix(torch.stack([x, y, a], dim=1).unsqueeze(0))[0]
        # The gripper is symmetric: rolled 180 deg about the approach axis it grips the bar the same way with the
        # fingers swapped. Take whichever is nearer the hand's current roll (a hand the IK already rolled far round
        # at the wrist limit otherwise has to turn ~180 deg back).
        flipped = quat_from_matrix(torch.stack([-x, -y, a], dim=1).unsqueeze(0))[0]
        _, q = self.tcp_pose()
        if torch.linalg.norm(self._rot_err(flipped, q)) < torch.linalg.norm(self._rot_err(quat, q)):
            quat = flipped
        return a, quat

    def _handle_grasp_point(self, quat: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Drawer-frame TCP target for closing on the bar with the hand at ``quat``, and the hand's tilt [deg].

        The closing line (hand y axis through the TCP) crosses the bar's centre height at x_line; the TCP sits s
        below that crossing along the line. s >= s_lo keeps the leading upper fingertip out of FACE_CLEAR_X_B
        (a steeper tilt needs the TCP lower), s <= HANDLE_DROP_MAX keeps the upper finger above the bar."""
        p = self.p
        _, drawer_q = self.drawer_frame()
        y_w = quat_apply(quat.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=self.device))
        y_b = quat_apply(quat_inv(drawer_q.unsqueeze(0)), y_w)[0]
        if y_b[2] > 0:  # the fingers are symmetric: take the closing axis pointing down
            y_b = -y_b
        sin_t = max(float(y_b[0]), 1e-3)  # hand tilted down -> the lower finger trails out of the cabinet
        room = BAR_LINE_X_RANGE[1] - FACE_CLEAR_X_B
        s_lo = min(max(0.0, FINGER_REACH - room / sin_t), HANDLE_DROP_MAX)
        s = s_lo + p.handle_drop_u * (HANDLE_DROP_MAX - s_lo)
        x_lo = min(max(BAR_LINE_X_RANGE[0], FACE_CLEAR_X_B + (FINGER_REACH - s) * sin_t), BAR_LINE_X_RANGE[1])
        x_line = x_lo + p.handle_line_u * (BAR_LINE_X_RANGE[1] - x_lo)
        crossing = torch.tensor([x_line, HANDLE_CENTER_B[1] + p.handle_lateral, HANDLE_CENTER_B[2]], device=self.device)
        return crossing + s * y_b, math.degrees(math.asin(min(sin_t, 1.0)))

    def _open_drawer(self) -> Plan:
        p = self.p
        a, q_grasp = self._handle_grasp_frame()
        nominal_b, _ = self._handle_grasp_point(q_grasp)

        def pre():
            return self.to_world(nominal_b) - a * p.handle_standoff

        tcp, q = self.tcp_pose()
        rel = self.to_world(nominal_b) - tcp
        along = float(rel @ a)
        lateral = torch.linalg.norm(rel - along * a).item()
        rot = float(torch.linalg.norm(self._rot_err(q_grasp, q)))
        lateral_max, rot_max = (EXPERT_HANDLE_NEAR_LATERAL, EXPERT_HANDLE_NEAR_ROT_DEG) if EXPERT_IN_PLACE else (0.02, 20.0)
        near = -0.005 < along < EXPERT_HANDLE_NEAR_ALONG and lateral < lateral_max and rot < math.radians(rot_max)
        if self.expert and EXPERT_SKIP_PREGRASP_IF_ALIGNED and near:
            self._trace(f"already near the handle ({along * 100:.1f} cm out, {lateral * 100:.1f} cm off the line, "
                        f"{math.degrees(rot):.0f} deg), going straight for the grasp")
            offset = float(self.to_drawer(tcp)[1]) - HANDLE_CENTER_B[1]
            if EXPERT_IN_PLACE and abs(offset) <= HANDLE_LATERAL_MAX:
                self.p = replace(self.p, handle_lateral=offset)  # grasp the bar where the hand already is
            if self.finger_gap() < FULL_OPEN_GAP - 0.02 and along < 0.04:
                # closed fingers this close would hit the bar: back off while opening
                yield from self.servo(
                    lambda: self.to_world(nominal_b) - a * 0.05, q, OPEN, speed=self.speed_careful, name="back off"
                )
        else:
            yield from self.transit(pre, q_grasp, OPEN, name="handle pre-grasp")
        if self.finger_gap() < FULL_OPEN_GAP - 0.02:
            yield from self.actuate(OPEN)
        # The orientation actually reached (the wrist may have stopped at a joint limit) sets the grasp point, and
        # the approach holds it: a pure translation onto the bar.
        _, q_now = self.tcp_pose()
        grasp_b, tilt = self._handle_grasp_point(q_now)
        self._trace(
            f"handle grasp at {tilt:.0f} deg tilt, TCP {[round(v, 4) for v in grasp_b.tolist()]} in the drawer frame"
        )
        yield from self.servo(
            lambda: self.to_world(grasp_b),
            q_now,
            OPEN,
            speed=self.speed_careful,
            pos_tol=0.003,
            rot_tol=0.05,
            max_steps=150,
            name="handle approach",
        )
        yield from self.actuate(CLOSE)
        if not self.holding_handle():
            raise PlanFailed("handle grasp missed")
        return (yield from self._pull_drawer())

    def _pull_drawer(self) -> Plan:
        p = self.p
        target = max(p.open_target, D_MIN_PLACE + 0.04, self._reach_opening())
        pos0, _ = self.tcp_pose()
        q0, approach = self._grasp_axis()
        heading = _wrap(math.atan2(float(approach[1]), float(approach[0])) - _azimuth(pos0))  # vs the arm's direction
        best, stall = self.drawer_open(), 0
        while self.drawer_open() < target - 0.004:
            pos, q = self.tcp_pose()
            goal = pos + self.open_dir() * (target - self.drawer_open())
            # Only the hand's heading is controlled: pulling straight out swings the arm round the base (~6 deg over
            # the pull) and the hand turns with it, so the wrist never yaws against the arm near its singularity
            # (that flipped the wrist mid-pull). Pitch and roll are left free: holding them against the wrist limit
            # stalled the pull at d = 0.10, where the teleop operator simply lets the hand tip a little.
            _, a_now = self._grasp_axis()
            turn = _wrap(_azimuth(pos) + heading - math.atan2(float(a_now[1]), float(a_now[0])))
            drot = torch.tensor([0.0, 0.0, turn], device=self.device)
            yield self._action(goal - pos, drot, CLOSE, self.pull_speed)
            d = self.drawer_open()
            if d > best + 0.0005:
                best, stall = d, 0
            else:
                stall += 1
            if not self.holding_handle():
                raise PlanFailed(f"handle slipped at d={d:.3f}")
            if d >= D_MIN_PLACE + 0.05 and self._handle_slipping():
                # far enough: let go cleanly instead of letting the fingers skid off the bar (slipped at 0.28-0.29)
                self._trace(f"fingers starting to slide off the bar at d={d:.3f}, letting go")
                break
            if stall >= 20:  # near cabinets fold the elbow into its limit first; >= D_MIN_PLACE is enough
                if d < D_MIN_PLACE:
                    raise PlanFailed(f"pull stalled at d={d:.3f}")
                self._trace(f"pull stopped at d={d:.3f} (elbow at its limit)")
                break
        yield from self._let_go(
            back=approach,
            distance=p.handle_retreat,
            up=p.handle_retreat_up,
            down_to_deg=p.handle_pitch_deg + p.handle_retreat_pitch_deg,
        )
        return self.drawer_open() >= D_MIN_PLACE

    def _reach_opening(self) -> float:
        """How far the drawer must be open for the front of its placement region to be within PLACE_MAX_RADIUS:
        opening it further brings the exposed part toward the robot. Far cabinets need ~0.3, which is still inside
        the teleop range; left at 0.26 the arm had to straighten to reach in and got stuck."""
        front = self.to_world((PLACE_FRONT_X_B, 0.0, 0.0))
        out = self.open_dir()  # the drawer slides toward the robot along this (horizontal) direction
        # smallest extra opening e with |front_xy + e * out_xy| <= PLACE_MAX_RADIUS
        b = float(front[:2] @ out[:2])
        c = float(front[:2] @ front[:2]) - PLACE_MAX_RADIUS**2
        extra = max(0.0, -b - math.sqrt(max(b * b - c, 0.0)))
        return min(self.drawer_open() + extra, 0.34)

    def _release_handle(self) -> Plan:
        p = self.p
        _, approach = self._grasp_axis()
        yield from self._let_go(
            back=approach,
            distance=p.handle_retreat,
            up=p.handle_retreat_up,
            down_to_deg=p.handle_pitch_deg + p.handle_retreat_pitch_deg,
        )
        return True

    def _grasp_axis(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, q = self.tcp_pose()
        z_axis = quat_apply(q.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=self.device))[0]
        return q, z_axis

    def _let_go(self, back: torch.Tensor, distance: float, up: float = 0.0, down_to_deg: float = 0.0) -> Plan:
        """Open the gripper, move back along ``back`` (the approach axis), then up while pitching the hand down until
        it points ``down_to_deg`` below horizontal (a level hand pulled back toward the base folds the elbow into its
        limit). The target is absolute: the hand may already have tipped while pulling."""
        # The expert starts backing off as soon as the fingers clear a 1.5 cm bar; demos wait for them to stop.
        yield from self.actuate(OPEN, until_gap=0.04 if self._fast else None)
        pos, q = self.tcp_pose()
        yield from self.servo(
            pos - back * distance,
            q,
            OPEN,
            speed=self.speed_careful,
            pos_tol=0.01,
            rot_tol=math.pi,
            name="retreat",
            fail_over=None,
        )
        if up > 0.005:
            pos, q = self.tcp_pose()
            level = back.clone()
            level[2] = 0.0
            _, a_now = self._grasp_axis()
            pitch = math.radians(down_to_deg) - math.asin(max(-1.0, min(1.0, -float(a_now[2]))))
            if pitch > 0.0 and torch.linalg.norm(level) > 0.3:
                axis = _unit(torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], device=self.device), level))
                turn = quat_from_angle_axis(torch.tensor([pitch], device=self.device), axis.unsqueeze(0))
                q = quat_mul(turn, q.unsqueeze(0))[0]
            yield from self.servo(
                pos + torch.tensor([0.0, 0.0, up], device=self.device),
                q,
                OPEN,
                pos_tol=0.01,
                rot_tol=0.15,
                name="retreat up",
            )
        return True

    def _charger_grasp_quat(self) -> torch.Tensor:
        p = self.p
        _, cq = self.charger_pose()
        long_axis = quat_apply(cq.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=self.device))[0]
        psi_long = math.atan2(float(long_axis[1]), float(long_axis[0]))
        # Hand x along the charger's long axis -> fingers close across its 4.2 cm side. Both yaws put the fingers on
        # the same faces; take the one the wrist reaches with the least turn, then add the strategy's jitter.
        # The arm swings round the base to the charger and back to the drawer; joint 7 has to stay in range for both.
        az_now, az_charger = _azimuth(self.tcp_pose()[0]), _azimuth(self.charger_center())
        to_charger, to_drawer = _wrap(az_charger - az_now), _wrap(self._drawer_azimuth() - az_charger)
        if self.expert:
            return self._nearest_grasp_quat([psi_long, psi_long + math.pi], to_charger, to_drawer)
        yaw = self._pick_yaw([psi_long, psi_long + math.pi], to_charger, to_drawer)
        jittered = yaw + math.radians(p.charger_yaw_noise_deg)
        q7 = self._q7_after(_wrap(jittered - self.hand_yaw()), to_charger)
        if self._q7_in_range(q7) and self._q7_in_range(q7 + to_drawer):
            yaw = jittered
        return _down_quat(yaw, math.radians(p.charger_tilt_x_deg), math.radians(p.charger_tilt_y_deg), self.device)

    def _drawer_azimuth(self) -> float:
        return _azimuth(self.to_world((PLACE_FRONT_X_B - 0.06, 0.0, 0.0)))

    def _nearest_grasp_quat(self, ideal_yaws: list[float], az_delta: float, then_az_delta: float) -> torch.Tensor:
        """Top-down grasp orientation nearest the current hand orientation (see EXPERT_GRASP_YAW_TOL_DEG)."""
        _, q = self.tcp_pose()
        z = matrix_from_quat(q.unsqueeze(0))[0][:, 2]
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        tilt = math.acos(max(-1.0, min(1.0, float(z @ down))))
        max_tilt = math.radians(EXPERT_GRASP_TILT_MAX_DEG)
        if tilt > max_tilt:  # swing the approach axis toward straight down until it is max_tilt off
            axis = torch.linalg.cross(z, down)
            if torch.linalg.norm(axis) < 1e-6:
                z = down
            else:
                turn = quat_from_angle_axis(
                    torch.tensor([tilt - max_tilt], device=self.device), _unit(axis).unsqueeze(0)
                )
                z = quat_apply(turn, z.unsqueeze(0))[0]
        current, tol = self.hand_yaw(), math.radians(EXPERT_GRASP_YAW_TOL_DEG)
        options = []
        for ideal in ideal_yaws:
            off = _wrap(current - ideal)
            target = ideal + max(-tol, min(tol, off))
            options.append((abs(_wrap(target - current)), target))
        options.sort()
        feasible = [
            t
            for _, t in options
            if self._q7_ok(_wrap(t - current), az_delta)
            and self._q7_in_range(self._q7_after(_wrap(t - current), az_delta) + then_az_delta)
        ]
        yaw = feasible[0] if feasible else options[0][1]
        x = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=self.device)
        x = _unit(x - (x @ z) * z)
        return quat_from_matrix(torch.stack([x, torch.linalg.cross(z, x), z], dim=1).unsqueeze(0))[0]

    def _pick_charger(self) -> Plan:
        p = self.p
        q_grasp = self._charger_grasp_quat()
        _, cq = self.charger_pose()
        offset = quat_apply(cq.unsqueeze(0), torch.tensor([[0.0, p.charger_long_offset, 0.0]], device=self.device))[0]
        offset[2] = p.charger_grasp_z

        def grasp():
            return self.charger_center() + offset

        up = torch.tensor([0.0, 0.0, p.pregrasp_height], device=self.device)
        tcp, q = self.tcp_pose()
        rel = tcp - grasp()
        lateral, height = torch.linalg.norm(rel[:2]).item(), float(rel[2])
        # right above it, or over it high enough that the fingertips clear the charger on the way in
        hovering = (lateral < 0.03 and 0.0 < height < 0.10) or (
            lateral < EXPERT_HOVER_LATERAL_MAX and EXPERT_DESCEND_FROM - 0.01 < height < 0.15
        )
        if self.expert and EXPERT_SKIP_PREGRASP_IF_ALIGNED and hovering:
            self._trace(f"over the charger ({lateral * 100:.1f} cm off, {height * 100:.1f} cm up), going straight in")
            if lateral >= 0.03:
                over = torch.tensor([0.0, 0.0, EXPERT_DESCEND_FROM], device=self.device)
                yield from self.servo(
                    lambda: grasp() + over, q_grasp, OPEN, pos_tol=0.008, rot_tol=0.15, name="over charger"
                )
        else:
            if self.finger_gap() < FULL_OPEN_GAP - 0.02 and self.charger_in_drawer():
                # re-pick from the drawer: get above the rim before opening/turning the hand
                yield from self.servo(
                    tcp + torch.tensor([0.0, 0.0, 0.06], device=self.device),
                    None,
                    self.gripper,
                    pos_tol=0.01,
                    name="clear",
                )
            yield from self.transit(lambda: grasp() + up, q_grasp, OPEN, name="charger pre-grasp")
        if self.finger_gap() < FULL_OPEN_GAP - 0.02:
            yield from self.actuate(OPEN)
        rot_tol = EXPERT_GRASP_ROT_TOL if self.expert else 0.05
        yield from self.servo(
            grasp,
            q_grasp,
            OPEN,
            speed=self.speed_careful,
            pos_tol=0.003,
            rot_tol=rot_tol,
            max_steps=150,
            name="charger descend",
        )
        yield from self.actuate(CLOSE)
        if not self.holding_charger():
            raise PlanFailed("charger grasp missed")
        # Lift: straight up past the drawer rim, with a lateral detour, keeping the grasp orientation.
        pos, q = self.tcp_pose()
        lift = pos.clone()
        lift[0] += p.lift_dx
        lift[1] += p.lift_dy
        radius = torch.linalg.norm(lift[:2]).item()
        if radius > PLACE_MAX_RADIUS:  # a charger at the table's far corner is picked with the arm straight: come back
            lift[:2] *= PLACE_MAX_RADIUS / radius
        lift[2] = max(float(pos[2]) + 0.05, self.rim_z() + p.carry_clearance)
        yield from self.servo(lift, q, CLOSE, pos_tol=0.01, rot_tol=0.08, name="lift")
        if not self.holding_charger():
            raise PlanFailed("charger dropped during the lift")
        return True  # next_stage() routes a held charger to place_charger

    def _place_charger(self) -> Plan:
        p = self.p
        d = self.drawer_open()
        x_lo = DRAWER_FRONT_X_B - d + PLACE_CABINET_MARGIN
        x = x_lo + p.place_u * max(PLACE_FRONT_X_B - x_lo, 0.0)
        # keep the release point within reach (see PLACE_MAX_RADIUS): slide it toward the robot, never past the front
        while (
            x < PLACE_FRONT_X_B and torch.linalg.norm(self.to_world((x, p.place_y, 0.0))[:2]).item() > PLACE_MAX_RADIUS
        ):
            x += 0.01
        target_b = (x, p.place_y, CHARGER_REST_Z_B + p.release_height)
        charger = self.charger_center()
        charger_b = self.to_drawer(charger)
        m = EXPERT_PLACE_MARGIN
        over_spot = (
            self.expert
            and EXPERT_IN_PLACE
            and float(charger[2]) > self.rim_z()
            and x_lo - m <= float(charger_b[0]) <= PLACE_FRONT_X_B + m
            and abs(float(charger_b[1])) <= PLACE_Y_MAX + m
        )
        if over_spot:  # see EXPERT_PLACE_MARGIN: release it where it already is
            target_b = (
                min(max(float(charger_b[0]), x_lo), PLACE_FRONT_X_B),
                min(max(float(charger_b[1]), -PLACE_Y_MAX), PLACE_Y_MAX),
                CHARGER_REST_Z_B + p.release_height,
            )

        def target():
            return self.to_world(target_b)

        def above():
            t = target()
            t[2] = self.rim_z() + p.carry_clearance
            return t

        tcp, q = self.tcp_pose()
        swing = _wrap(_azimuth(target()) - _azimuth(tcp))
        yaw_delta = math.radians(p.carry_yaw_deg)
        for option in (yaw_delta, -yaw_delta, 0.0, swing):  # last resort: turn the hand with the arm (q7 unchanged)
            if self._q7_ok(option, swing):
                yaw_delta = option
                break
        level = _about_world_z(q, yaw_delta).unsqueeze(0)
        tip = torch.tensor([math.radians(p.carry_tilt_deg)], device=self.device)

        def q_carry():
            # tipped forward about the horizontal axis across the arm's direction (see CARRY_TILT_DEG_RANGE)
            az = _azimuth(self.tcp_pose()[0])
            forward = torch.tensor([math.cos(az), math.sin(az), 0.0], device=self.device)
            axis = _unit(torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], device=self.device)))
            return quat_mul(quat_from_angle_axis(tip, axis.unsqueeze(0)), level)[0]

        pos, _ = self.tcp_pose()
        if over_spot:
            self._trace("charger already over the drawer's release area: lowering it from here")
        else:
            if float(pos[2]) < self.rim_z() + 0.03:
                # still low (e.g. the policy is holding it near the table): lift before crossing to the drawer
                yield from self.servo(
                    pos + torch.tensor([0.0, 0.0, self.rim_z() + p.carry_clearance - float(pos[2])], device=self.device),
                    None,
                    CLOSE,
                    pos_tol=0.01,
                    name="lift",
                )
            yield from self.transit(above, q_carry, CLOSE, anchor=self.charger_center, name="carry")
        yield from self.servo(
            target,
            q_carry,
            CLOSE,
            speed=self.speed_careful,
            pos_tol=0.006,
            rot_tol=0.1,
            anchor=self.charger_center,
            max_steps=150,
            name="lower",
        )
        yield from self.actuate(OPEN, until_gap=0.06 if self._fast else None)
        pos, _ = self.tcp_pose()
        up = torch.tensor([0.0, 0.0, max(self.rim_z() + p.place_retreat - float(pos[2]), 0.02)], device=self.device)
        yield from self.servo(pos + up, None, OPEN, pos_tol=0.01, name="retreat up")
        return True

    def _put_down_charger(self) -> Plan:
        """Holding the charger but the drawer is not open (enough): set it back down where it is and let go."""
        pos = self.charger_center()
        pos[2] = self.charger_start_z() + 0.01
        yield from self.servo(
            pos,
            None,
            CLOSE,
            speed=self.speed_careful,
            pos_tol=0.006,
            anchor=self.charger_center,
            max_steps=150,
            name="set down",
        )
        yield from self.actuate(OPEN)
        tcp, _ = self.tcp_pose()
        yield from self.servo(
            tcp + torch.tensor([0.0, 0.0, 0.08], device=self.device), None, OPEN, pos_tol=0.01, name="retreat up"
        )
        return True

    def _close_drawer(self) -> Plan:
        p = self.p
        if self.finger_gap() < FULL_OPEN_GAP - 0.02 and (self.holding_charger() or self.holding_handle()):
            yield from self.actuate(OPEN)
        gripper = OPEN
        push_y, push_z = p.push_y, p.push_z
        tcp_b = self.to_drawer(self.tcp_pose()[0])
        m = EXPERT_PUSH_MARGIN
        at_front = (
            self.expert
            and EXPERT_IN_PLACE
            and DRAWER_FRONT_X_B < float(tcp_b[0]) < DRAWER_FRONT_X_B + PUSH_STANDOFF_RANGE[1] + m
            and abs(float(tcp_b[1])) <= PUSH_Y_MAX + m
            and PUSH_Z_RANGE[0] - m <= float(tcp_b[2]) <= PUSH_Z_RANGE[1] + m
        )
        if at_front:  # see EXPERT_PUSH_MARGIN: push where the hand already is
            push_y = min(max(float(tcp_b[1]), -PUSH_Y_MAX), PUSH_Y_MAX)
            push_z = min(max(float(tcp_b[2]), PUSH_Z_RANGE[0]), PUSH_Z_RANGE[1])
        # Hand down with the fingers side by side along the drawer face, tilted forward. With the fingers one in
        # front of the other the near one hit the protruding handle bar while descending below it. Of the two such
        # yaws, the one joint 7 reaches with the least turn where the push happens (plus jitter it can take).
        face = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], device=self.device), self.open_dir())
        psi_face = math.atan2(float(face[0]), -float(face[1]))  # top-down hand: y axis = (sin psi, -cos psi, 0)
        swing = _wrap(_azimuth(self.to_world((DRAWER_FRONT_X_B, push_y, 0.0))) - _azimuth(self.tcp_pose()[0]))
        yaw = self._pick_yaw([psi_face, psi_face + math.pi], swing)
        jitter = math.radians(p.push_yaw_noise_deg)
        if self._q7_ok(_wrap(yaw + jitter - self.hand_yaw()), swing):
            yaw += jitter
        down = _down_quat(yaw, device=self.device).unsqueeze(0)
        d_start = max(self.drawer_open(), 1e-3)

        def q_push():
            # Tilt about the horizontal axis across the arm's current direction (a pure pitch in the arm's plane,
            # which the wrist can make even at its singularity) so the fingertips lead toward the cabinet, leaning
            # further forward as the drawer front moves away (see PUSH_TILT_START_DEG).
            az = _azimuth(self.tcp_pose()[0])
            forward = torch.tensor([math.cos(az), math.sin(az), 0.0], device=self.device)
            axis = _unit(torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], device=self.device)))
            closed = min(max(1.0 - self.drawer_open() / d_start, 0.0), 1.0)
            tilt_deg = PUSH_TILT_START_DEG + closed * (p.push_tilt_deg - PUSH_TILT_START_DEG)
            tilt = torch.tensor([math.radians(tilt_deg)], device=self.device)
            return quat_mul(quat_from_angle_axis(tilt, axis.unsqueeze(0)), down)[0]

        high_b = (DRAWER_FRONT_X_B + p.push_standoff, push_y, max(push_z + 0.05, DRAWER_RIM_Z_B + 0.05))
        pre_b = (DRAWER_FRONT_X_B + p.push_standoff, push_y, push_z)
        if at_front:
            self._trace("hand already at the drawer front: pushing from here")
        else:
            if float(tcp_b[0]) < DRAWER_FRONT_X_B + 0.03 and float(tcp_b[2]) < DRAWER_RIM_Z_B + 0.04:
                # hand is over/inside the drawer: straight up first
                up = self.to_world((float(tcp_b[0]), float(tcp_b[1]), DRAWER_RIM_Z_B + 0.06))
                yield from self.servo(up, None, gripper, pos_tol=0.01, name="clear drawer")
            yield from self.transit(lambda: self.to_world(high_b), q_push, gripper, name="push approach")
            yield from self.servo(
                lambda: self.to_world(pre_b),
                q_push,
                gripper,
                speed=self.speed_careful,
                pos_tol=0.006,
                rot_tol=0.08,
                name="push descend",
            )

        def push_target():
            return self.closed_to_world((DRAWER_FRONT_X_B - PUSH_OVERSHOOT, push_y, push_z))

        best, stall = self.drawer_open(), 0
        for _ in range(300):
            if self.drawer_open() < D_CLOSED:
                break
            pos, q = self.tcp_pose()
            yield self._action(push_target() - pos, self._rot_err(q_push(), q), gripper, self.push_speed)
            d = self.drawer_open()
            if d < best - 0.0005:
                best, stall = d, 0
            else:
                stall += 1
            if stall >= 25:
                self._trace("push stalled")
                break
        # a few more steps of pressure so the success term sees it shut, then back off
        yield from self.hold(2, gripper)
        if self.drawer_open() >= 0.01:
            raise PlanFailed(f"drawer did not close (d={self.drawer_open():.3f})")
        pos, q = self.tcp_pose()
        yield from self.servo(
            pos + self.open_dir() * p.push_backoff,
            q,
            gripper,
            speed=self.speed_careful,
            pos_tol=0.008,
            rot_tol=math.pi,
            name="back off",
        )
        return True

    def _finish(self) -> Plan:
        tcp, _ = self.tcp_pose()
        if float(self.to_drawer(tcp)[0]) < DRAWER_FRONT_X_B + 0.03:
            yield from self.servo(
                tcp + self.open_dir() * self.p.push_backoff,
                None,
                self.gripper,
                speed=self.speed_careful,
                pos_tol=0.008,
                name="back off",
            )
        return True
