"""The motion-planning expert for PlugCharger: collects the demos and takes over during DAgger.

Stages: reach above the charger, descend, grasp, close, lift to a waypoint, transit to an approach pose in front
of the socket, align onto the socket axis, insert with a straight screw move, then hold the gripper closed for 30
steps (no release: past ~1.5 mm clearance, opening the gripper flings the charger out of the hole).

Each demo samples a ``StrategyParams`` so the demos are not one narrow funnel:

  - grasp: tilt angle (5-25 deg) and offset along the charger base's long axis (+-8 mm)
  - a lift waypoint (random height and lateral detour) between grasp and approach
  - the approach pose: standoff distance, plus a lateral offset and a pitch/yaw off the socket axis, corrected by
    an align move onto the true axis before the straight insert
  - joint velocity/acceleration limits, sampled separately for the free-space and the align+insert stages
  - screw vs RRTConnect for the transit stages, each falling back to the other when it finds no path

Insert itself is always a straight screw move. Sampling ranges are the module constants below.

The strategy is independent of the scene: the episode seed only drives the env's charger/socket randomization,
while the strategy comes from a separate module-level RNG stream (``seed_strategy_rng`` / ``next_strategy_seed``).
So a DAgger takeover on a scene the policy already has a demo of yields a different trajectory, like a human.
Every draw is reproducible: the collection and DAgger scripts seed the stream, and each executed
``StrategyParams`` is appended to ``used_strategies`` (drained per episode by ``take_used_strategies`` and written
into the episode's JSON entry), so a demo can be replayed with ``solve(..., strategy=StrategyParams(**d))``
(RRT paths excepted: mplib doesn't expose its RNG).

Entry points: ``solve`` (a demo from a fresh reset) and ``resume_and_solve`` (the DAgger expert: it inspects the
live state, skips reach/grasp/lift if the policy already holds the charger, and unwinds when the DAgger gate hands
control back to the policy, see ``interruptible_planner.py``).
"""

from __future__ import annotations

import os
import secrets
from dataclasses import asdict, dataclass

import numpy as np
import sapien
import trimesh
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from vig_dagger.expert.interruptible_planner import AbortedError, InterruptiblePandaArmMotionPlanningSolver

FINGER_LENGTH = 0.025
REACH_HEIGHT = 0.05  # reach pose sits this far back along the approach axis (stock)
PAUSE_STEPS = 30  # closed-gripper hold at the insert pose (stock pause tail)

# ---- sampling ranges (stock values in comments) -----------------------------
GRASP_TILT_DEG_RANGE = (5.0, 25.0)  # stock: fixed 15
# Along the grasp frame's x axis = charger base's long (peg) axis, 4cm total.
# +-8mm keeps the ~2cm finger pad fully on the base.
GRASP_OFFSET_MAX = 0.008  # stock: 0
# Off by default: calibration (40 seeds) showed flipped grasps fail ~20% of
# the time and run 300-450 steps -- the mirrored wrist puts the align pose
# near joint limits, so screw planning fails and the RRT fallback takes a
# long detour. Unflipped grasps succeeded 100%. The code path is kept so it
# can be re-enabled if the step budget/limits change.
GRASP_FLIP_PROB = 0.0  # stock: never
STANDOFF_RANGE = (0.03, 0.08)  # stock: 0.05
# Always lift: calibration showed the un-lifted direct grasp->pre-insert screw
# fails with steep tilts (~19 deg dragged the charger on seed 27; fine with a
# lift). A 2cm lift is close enough to the stock direct transit that that
# mode is still covered.
LIFT_PROB = 1.0  # stock: no lift, straight from grasp to pre-insert
LIFT_HEIGHT_RANGE = (0.02, 0.10)
LIFT_LATERAL_MAX = 0.05  # world-xy detour, each axis uniform in +-this
# Joint vel/acc limit scale, sampled separately for the free-space stages
# (reach, descend, lift, transit to the approach pose) and the careful stages
# (align onto the socket axis + insert), so "quick to get there, slow going
# in" and the reverse both occur. Stock: 0.5 everywhere. Transit never
# slower than stock (episode-step budget); insert may be.
SPEED_TRANSIT_RANGE = (0.5, 0.9)
SPEED_INSERT_RANGE = (0.4, 0.85)
# Off-axis approach: the pose the arm transits to before inserting is no
# longer exactly on the socket axis. It is displaced laterally (goal-frame y/z,
# perpendicular to the insertion axis) and rotated (pitch/yaw about those
# axes); a screw move then corrects onto the true axis at the standoff
# distance before the straight insert. Gives the policy demos of "arrive
# slightly off, correct, then insert" -- stock: always perfectly on-axis.
APPROACH_LATERAL_MAX = 0.015  # m, each of y/z uniform in +-this
APPROACH_ANGLE_DEG_MAX = 8.0  # deg, each of pitch/yaw uniform in +-this
RRT_PROB = 0.3  # stock: always screw
RRT_MAX_STEP_RATIO = 2.0  # take an RRT path only if <= this x the screw path's steps (see _transit)

# ---- DAgger-expert overrides (resume_and_solve only; demos are unaffected) ----
# Calibrated in earlier experiments whose gate handed control back after 16
# consecutive calm expert steps (this repo's gate: dagger.return_count steps
# below the return threshold, see vig_dagger/gate.py); the reasoning holds for
# any short hand-back window. EXPERT_ATOMIC_GRASP is set from the experiment
# config (dagger.atomic_grasp).
# When the planner stands in for a human during a DAgger takeover it competes
# with auto-return: control goes back to the policy after
# auto_return_below_count (16) consecutive expert steps with a sub-threshold
# score, and a barely-moving arm scores low. Measured on the p99 collections,
# takeovers spent ~30% of their steps stationary (6-step gripper re-assert,
# 5 refine steps) and never left mplib's from-rest acceleration ramp (median
# peak 3-5 mm/step vs ~15 in a demo transit); 41-47% of hand-backs happened
# while still accelerating. A human expert moves at once and at speed, so the
# expert here does too: no stationary steps, and higher velocity/acceleration
# limits so a correction fits inside the 16-step window. Set any of these to
# None to fall back to the demo behaviour (sampled speeds, 6 / 5 steps).
EXPERT_SPEED_TRANSIT = 1.5  # calibrated 20/20 at 1.5 (2.0 starts to fail)
EXPERT_SPEED_INSERT = 1.0
# Extra acceleration on top of the speed scale was tested and rejected: x3 ->
# 6/20 takeovers succeed, x1.5 -> 17/20, x1 -> 20/20 (the grasped charger
# slips / the insert overshoots). Keep acceleration tied to the speed scale.
EXPERT_ACC_SCALE = 1.0
EXPERT_REGRIP_STEPS = 0  # steps spent re-asserting the closed gripper at takeover (demo path: 6); 0 = set the planner's gripper state without stepping
EXPERT_REFINE_STEPS = 0  # stationary refine steps after the align move (demo path: 5)
# The grasp (reach -> descend -> close -> lift) takes 25-45 steps even at the
# expert speed, while auto-return hands control back after 16 calm readings:
# measured on the p99 collections, 45-62% of hand-backs happened before the
# grasp and ~30% during the close. With EXPERT_ATOMIC_GRASP the expert treats
# the grasp as one unit: an auto-return signal received inside it is latched
# and honoured only once the charger is grasped and lifted (see
# InterruptiblePandaArmMotionPlanningSolver.defer_abort). The auto-return
# decision itself (threshold, 16-step count) is untouched -- only the moment
# the expert lets go changes, like a human who finishes the grasp first.
# Off: hand-back stays strict, exactly like the human-teleop recorder in the
# open-drawer setting (the 16th low reading hands control to the policy on the
# next step, whatever the expert was doing). Instead the grasp is made fast
# enough to finish inside the 16-step window (see EXPERT_HOVER_LATERAL_MAX):
# measured 29/30 grasped at a strict step-16 hand-back vs 0/30 for the old
# expert. Set True to defer the hand-back until the grasp unit completes.
EXPERT_ATOMIC_GRASP = False
# When the policy is already hovering over the charger (within 2 cm laterally,
# 0.5-8 cm above it), skip the detour to the sampled reach pose and descend
# straight to the grasp pose; the screw move handles the reorientation.
EXPERT_SKIP_REACH_IF_ABOVE = True
# m: "already over the charger" if the TCP is within this laterally (grasp
# frame) and 0.5-8 cm above. 6 cm: direct descend from up to 6 cm off still
# grasped 29/30 in the harness; the reach detour beyond 4 cm dropped it to 22/30.
EXPERT_HOVER_LATERAL_MAX = 0.06
# Acceleration scale for the pre-grasp moves only (nothing in the hand yet, so
# the slip problem behind EXPERT_ACC_SCALE=1 doesn't apply).
EXPERT_PREGRASP_ACC_SCALE = 2.0
# Taking over with the charger already held near the socket, the demo stage
# sequence would transit back out to the (off-axis) approach pose at the
# sampled standoff. With EXPERT_CORRECT_IN_PLACE the expert instead corrects
# from where the plug is: once the plug is within the standoff of the socket it
# skips the approach and aligns onto the socket axis at the plug's current
# distance -- backing off only to SOCKET_FACE_DEPTH + ALIGN_CLEARANCE when the
# pins are at or past the face but not in the holes (off the axis by more than
# IN_HOLE_OFFSET or tilted by more than IN_HOLE_TILT_DEG), so they don't scrape
# along the face -- then inserts. Taking over right after a grasp (plug held,
# less than LIFT_DONE_HEIGHT above its resting height), it first does the demo
# lift to the strategy's waypoint, as a demo would.
EXPERT_CORRECT_IN_PLACE = True
SOCKET_FACE_DEPTH = 0.016  # m: plug-to-goal distance along the socket axis at which the pins reach the face
ALIGN_CLEARANCE = 0.005  # m in front of the face where pins that are not in the holes are re-aligned
IN_HOLE_OFFSET = 0.004  # m off the socket axis (the holes' clearance per side)
IN_HOLE_TILT_DEG = 5.0
LIFT_DONE_HEIGHT = 0.02  # m above the charger's resting height
# The random end-effector move at the start of every DAgger takeover (variant random_move; random_move.py), before
# the expert replans. None = off: the expert starts at once. The DAgger worker sets it from dagger.random_move:
# dict(radius=m, rot_deg=deg, max_tries=n, margin=m, tol=m). The move runs at the transit speed of the strategy
# drawn for the takeover (EXPERT_SPEED_TRANSIT in the "fast" expert mode) with the gripper command held, polls
# ``on_move_step`` (not ``on_step``: no query score) and is logged in ``used_moves``.
RANDOM_MOVE: dict | None = None


def set_expert_mode(mode: str) -> None:
    """How the DAgger expert moves after a takeover.

    "demo": like the planner demos -- the strategy's sampled speeds, acceleration tied to speed, the 5 refine steps
    after aligning -- so DAgger data looks like the training demos. "fast": the earlier DAgger expert (transit speed
    1.5, insert 1.0, no refine steps, faster pre-grasp acceleration), tuned to finish corrections inside a short
    hand-back window. Both re-assert the grasp without stationary steps, skip the reach detour when already over the
    charger, and correct in place near the socket (EXPERT_CORRECT_IN_PLACE).
    """
    global EXPERT_SPEED_TRANSIT, EXPERT_SPEED_INSERT, EXPERT_ACC_SCALE, EXPERT_REFINE_STEPS, EXPERT_PREGRASP_ACC_SCALE
    if mode == "demo":
        EXPERT_SPEED_TRANSIT = EXPERT_SPEED_INSERT = EXPERT_ACC_SCALE = EXPERT_REFINE_STEPS = None
        EXPERT_PREGRASP_ACC_SCALE = 1.0
    elif mode == "fast":
        EXPERT_SPEED_TRANSIT, EXPERT_SPEED_INSERT, EXPERT_ACC_SCALE = 1.5, 1.0, 1.0
        EXPERT_REFINE_STEPS, EXPERT_PREGRASP_ACC_SCALE = 0, 2.0
    else:
        raise ValueError(f"expert mode must be demo or fast, got {mode!r}")
# Ranges above were calibrated so demos land at ~160-290 steps, inside the
# env's 300-step limit (the deterministic pause solver runs 186-208 at speed
# 0.5). collect_plug_charger_demos.py rejects any demo that still overruns.


@dataclass(frozen=True)
class StrategyParams:
    seed: int
    grasp_tilt_deg: float
    grasp_offset: float
    grasp_flip: bool
    standoff: float
    lift: bool
    lift_height: float
    lift_dx: float
    lift_dy: float
    speed_transit: float
    speed_insert: float
    approach_dy: float  # m, goal-frame lateral offsets of the off-axis approach pose
    approach_dz: float
    approach_pitch_deg: float  # deg, goal-frame rotations of the off-axis approach pose
    approach_yaw_deg: float
    transit_planner: str  # "screw" | "rrt"

    @classmethod
    def sample(cls, seed: int) -> "StrategyParams":
        rng = np.random.default_rng(int(seed))
        return cls(
            seed=int(seed),
            grasp_tilt_deg=float(rng.uniform(*GRASP_TILT_DEG_RANGE)),
            grasp_offset=float(rng.uniform(-GRASP_OFFSET_MAX, GRASP_OFFSET_MAX)),
            grasp_flip=bool(rng.random() < GRASP_FLIP_PROB),
            standoff=float(rng.uniform(*STANDOFF_RANGE)),
            lift=bool(rng.random() < LIFT_PROB),
            lift_height=float(rng.uniform(*LIFT_HEIGHT_RANGE)),
            lift_dx=float(rng.uniform(-LIFT_LATERAL_MAX, LIFT_LATERAL_MAX)),
            lift_dy=float(rng.uniform(-LIFT_LATERAL_MAX, LIFT_LATERAL_MAX)),
            speed_transit=float(rng.uniform(*SPEED_TRANSIT_RANGE)),
            speed_insert=float(rng.uniform(*SPEED_INSERT_RANGE)),
            approach_dy=float(rng.uniform(-APPROACH_LATERAL_MAX, APPROACH_LATERAL_MAX)),
            approach_dz=float(rng.uniform(-APPROACH_LATERAL_MAX, APPROACH_LATERAL_MAX)),
            approach_pitch_deg=float(rng.uniform(-APPROACH_ANGLE_DEG_MAX, APPROACH_ANGLE_DEG_MAX)),
            approach_yaw_deg=float(rng.uniform(-APPROACH_ANGLE_DEG_MAX, APPROACH_ANGLE_DEG_MAX)),
            transit_planner="rrt" if rng.random() < RRT_PROB else "screw",
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---- strategy RNG stream (independent of the env's episode seed) --------------

_strategy_rng = np.random.default_rng()  # unseeded: fresh entropy per process
used_strategies: list[dict] = []  # every StrategyParams executed since the last take_used_strategies()
used_moves: list[dict] = []  # one entry per takeover with RANDOM_MOVE set, since the last take_used_moves()


def seed_strategy_rng(seed: int | None) -> int:
    """Make the sequence of strategies drawn by ``solve``/``resume_and_solve``
    reproducible for this process (the scripts expose this as --strategy-seed).
    ``None`` picks a random seed. Returns the seed actually used, so callers
    can print it."""
    global _strategy_rng
    if seed is None:
        seed = secrets.randbits(31)
    _strategy_rng = np.random.default_rng(int(seed))
    return int(seed)


def next_strategy_seed() -> int:
    return int(_strategy_rng.integers(2**31))


def take_used_moves() -> list[dict]:
    """Drain the random moves made since the last call (one per takeover while RANDOM_MOVE is set)."""
    out = list(used_moves)
    used_moves.clear()
    return out


def take_used_strategies() -> list[dict]:
    """Drain the strategies executed since the last call (one per ``solve``,
    one per ``resume_and_solve`` takeover). Empty for envs that don't route
    through this solver, so recorders can log ``None`` in that case."""
    out = list(used_strategies)
    used_strategies.clear()
    return out


def _pick(strategy: StrategyParams | None) -> StrategyParams:
    params = strategy if strategy is not None else StrategyParams.sample(next_strategy_seed())
    used_strategies.append(params.to_dict())
    return params


# ---- stages ------------------------------------------------------------------


def _grasp_pose(base_env, params: StrategyParams) -> sapien.Pose:
    # Same OBB grasp as the stock solver; reads the charger's *live* pose so it
    # is correct whether the episode is fresh or the charger was nudged by an
    # earlier policy rollout.
    obb = trimesh.primitives.Box(
        extents=np.array(base_env._base_size) * 2,
        transform=base_env.charger_base_pose.sp.to_transformation_matrix(),
    )
    approaching = np.array([0, 0, -1])
    target_closing = base_env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    grasp_pose = base_env.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])
    # Offset along the base's long axis in the untilted grasp frame, then the
    # tilt about the closing axis (stock applies a fixed 15 deg here), then an
    # optional 180 deg flip about the (tilted) approach axis -- same approach
    # line, fingers swapped sides.
    grasp_pose = grasp_pose * sapien.Pose([params.grasp_offset, 0, 0])
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, np.deg2rad(params.grasp_tilt_deg), 0))
    if params.grasp_flip:
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, 0, np.pi))
    return grasp_pose


def _failed(res) -> bool:
    # planner move_* methods return -1 on planning failure, else a step tuple.
    return isinstance(res, int) and res == -1


_DEBUG = bool(os.environ.get("DIVERSE_EXPERT_DEBUG"))


def _trace(planner, base_env, msg: str) -> None:
    """Stage-level trace of the expert, enabled with DIVERSE_EXPERT_DEBUG=1."""
    if not _DEBUG:
        return
    tcp = base_env.agent.tcp.pose.sp.p
    print(
        f"[expert t={int(base_env.elapsed_steps[0]):3d} planner_steps={planner.elapsed_steps:3d} grip={'C' if planner.gripper_state < 0 else 'o'} "
        f"aborted={int(getattr(planner, 'aborted', False))} defer={int(getattr(planner, 'defer_abort', False))} tcp_z={tcp[2]*100:4.1f}cm] {msg}",
        flush=True,
    )


def _transit(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose, params: StrategyParams):
    """Free-space move (no contact involved), planned with the strategy's
    planner but falling back to the other one if it can't find a path.

    The fallback matters in both directions: screw (IK from the current
    qpos) reliably fails to reach a *flipped* grasp, while RRTConnect gets
    there every time; conversely RRT occasionally returns no path. When both
    succeed and the strategy asked for RRT, the RRT path is only taken if it
    isn't much longer than the screw one -- unsimplified joint-space detours
    can otherwise blow the episode-step budget.
    """
    screw = planner.move_to_pose_with_screw(pose, dry_run=True)
    want_rrt = params.transit_planner == "rrt" or _failed(screw)
    rrt = planner.move_to_pose_with_RRTConnect(pose, dry_run=True) if want_rrt else -1
    if _failed(screw) and _failed(rrt):
        _trace(planner, planner.base_env, f"transit: BOTH planners failed for target p={np.round(pose.p, 3)} -> no motion")
        return -1
    if _DEBUG:
        _trace(planner, planner.base_env, f"transit: screw={'ok' if not _failed(screw) else 'FAIL'} rrt={'ok' if not _failed(rrt) else ('FAIL' if want_rrt else 'n/a')} target p={np.round(pose.p, 3)}")
    if _failed(rrt):
        path = screw
    elif _failed(screw):
        path = rrt
    else:
        path = rrt if len(rrt["position"]) <= RRT_MAX_STEP_RATIO * len(screw["position"]) else screw
    return planner.follow_path(path)


def _init_speed(planner: PandaArmMotionPlanningSolver) -> None:
    """Remember mplib's unscaled joint limits so ``_set_speed`` can rescale
    them per stage. The planner must have been built with limits scale 1.0."""
    planner._base_vel_limits = np.array(planner.planner.joint_vel_limits, dtype=float)
    planner._base_acc_limits = np.array(planner.planner.joint_acc_limits, dtype=float)


def _set_speed(planner: PandaArmMotionPlanningSolver, scale: float, acc_scale: float = 1.0) -> None:
    """Scale mplib's joint velocity limits by ``scale`` and its acceleration
    limits by ``scale * acc_scale`` (demos: acc_scale 1, i.e. both together)."""
    planner.planner.joint_vel_limits = planner._base_vel_limits * scale
    planner.planner.joint_acc_limits = planner._base_acc_limits * scale * acc_scale


def _run_stages(planner: PandaArmMotionPlanningSolver, base_env, params: StrategyParams, *, do_grasp: bool, expert: bool = False):
    # No-op on the plain planner; raises AbortedError on the interruptible one
    # once DAgger auto-return has latched.
    check = getattr(planner, "check_aborted", lambda: None)
    # DAgger-expert overrides (see EXPERT_* above); demos use the sampled values.
    speed_transit = EXPERT_SPEED_TRANSIT if (expert and EXPERT_SPEED_TRANSIT is not None) else params.speed_transit
    speed_insert = EXPERT_SPEED_INSERT if (expert and EXPERT_SPEED_INSERT is not None) else params.speed_insert
    acc_scale = EXPERT_ACC_SCALE if (expert and EXPERT_ACC_SCALE is not None) else 1.0
    regrip_steps = EXPERT_REGRIP_STEPS if (expert and EXPERT_REGRIP_STEPS is not None) else 6
    refine_steps = EXPERT_REFINE_STEPS if (expert and EXPERT_REFINE_STEPS is not None) else 5
    _init_speed(planner)
    _set_speed(planner, speed_transit, acc_scale)
    _trace(planner, base_env, f"START do_grasp={do_grasp} expert={expert} speeds={speed_transit}/{speed_insert} acc={acc_scale} regrip={regrip_steps} refine={refine_steps}")

    if do_grasp:
        grasp_pose = _grasp_pose(base_env, params)
        reach_pose = grasp_pose * sapien.Pose([0, 0, -REACH_HEIGHT])
        atomic = expert and EXPERT_ATOMIC_GRASP and hasattr(planner, "defer_abort")
        if atomic:
            planner.defer_abort = True  # see EXPERT_ATOMIC_GRASP: finish the grasp before honouring an auto-return
        if expert:
            _set_speed(planner, speed_transit, acc_scale * EXPERT_PREGRASP_ACC_SCALE)
        # Already over (or level with) the charger, typical for a takeover at
        # the grasp? Then don't detour to the sampled reach pose. Judged in the
        # world frame relative to the charger *base* centre: lateral distance
        # within EXPERT_HOVER_LATERAL_MAX and height from -1 cm (already at
        # grasp level, e.g. fingers open beside the charger) to +8 cm. (An
        # earlier version judged this in the tilted grasp frame, which
        # rejected level-but-offset starts and sent the expert 5 cm up first.)
        tcp_p = base_env.agent.tcp.pose.sp.p
        base_p = base_env.charger_base_pose.sp.p
        lateral = float(np.linalg.norm(tcp_p[:2] - base_p[:2]))
        height = float(tcp_p[2] - base_p[2])
        hovering = lateral < EXPERT_HOVER_LATERAL_MAX and -0.01 < height < 0.08
        _trace(planner, base_env, f"grasp unit: hovering={hovering} (lateral {lateral*100:.1f}cm, height {height*100:.1f}cm) atomic={atomic}")
        if not (expert and EXPERT_SKIP_REACH_IF_ABOVE and hovering):
            _transit(planner, reach_pose, params)
            _trace(planner, base_env, "reach done")
            check()
        r = planner.move_to_pose_with_screw(grasp_pose)
        _trace(planner, base_env, f"descend done ({'FAILED plan' if _failed(r) else 'ok'})")
        check()
        planner.close_gripper()
        _trace(planner, base_env, f"gripper closed; is_grasping={bool(base_env.agent.is_grasping(base_env.charger)[0])}")
        if params.lift:
            # World-frame offset from the grasp pose: straight up plus a
            # lateral detour, so the transit to the receptacle doesn't always
            # start from the same place or head there in a straight line.
            lift_pose = sapien.Pose([params.lift_dx, params.lift_dy, params.lift_height]) * grasp_pose
            _transit(planner, lift_pose, params)
        if expert:
            _set_speed(planner, speed_transit, acc_scale)
        _trace(planner, base_env, f"lift done; charger_z={float(base_env.charger.pose.p[0, 2])*100:.1f}cm is_grasping={bool(base_env.agent.is_grasping(base_env.charger)[0])}")
        if atomic:
            planner.defer_abort = False  # grasp complete and lifted: a deferred auto-return is honoured here
        check()
    else:
        # Already holding the charger. A freshly constructed planner assumes
        # the gripper is OPEN regardless of sim state and follow_path re-issues
        # gripper_state on every move, so re-assert CLOSED before any motion.
        # The DAgger expert does this without spending env steps (the sim's
        # gripper is already closed on the charger; the next move's action
        # carries the CLOSED command), so no stationary steps at takeover.
        if regrip_steps > 0:
            planner.close_gripper(t=regrip_steps)
        else:
            planner.gripper_state = planner.CLOSED

        if expert and EXPERT_CORRECT_IN_PLACE and params.lift:
            # Taken over right after the grasp: do the demo lift first (see EXPERT_CORRECT_IN_PLACE).
            lifted = float(base_env.charger.pose.sp.p[2] - base_env._base_size[2])
            if lifted < LIFT_DONE_HEIGHT:
                tcp = base_env.agent.tcp.pose.sp
                _transit(planner, sapien.Pose([params.lift_dx, params.lift_dy, max(params.lift_height - lifted, 0.0)]) * tcp, params)
                _trace(planner, base_env, f"lift from {lifted * 100:.1f} cm done")
                check()

    # Align/insert poses are computed from the *live* charger/tcp/goal poses,
    # not accumulated from grasp_pose, so they're right for any grasp.
    tcp_in_charger = base_env.charger.pose.sp.inv() * base_env.agent.tcp.pose.sp
    goal = base_env.goal_pose.sp
    insert_pose = goal * tcp_in_charger
    standoff = params.standoff
    rel = goal.inv() * base_env.charger.pose.sp  # plug in the goal (seated) frame: x along the socket axis
    depth, off_axis = float(-rel.p[0]), float(np.hypot(rel.p[1], rel.p[2]))
    if not do_grasp and expert and EXPERT_CORRECT_IN_PLACE and depth < standoff:
        # Held within the standoff of the socket: no transit back out to the approach pose.
        tilt = float(np.degrees(np.arccos(np.clip(rel.to_transformation_matrix()[0, 0], -1.0, 1.0))))
        in_holes = depth < SOCKET_FACE_DEPTH and off_axis < IN_HOLE_OFFSET and tilt < IN_HOLE_TILT_DEG
        standoff = max(depth, 0.0) if in_holes else max(depth, SOCKET_FACE_DEPTH + ALIGN_CLEARANCE)
        _trace(planner, base_env, f"near the socket (depth {depth * 1000:.0f} mm, off axis {off_axis * 1000:.1f} mm, "
                                  f"tilt {tilt:.1f} deg): align at {standoff * 1000:.0f} mm, no approach")
    else:
        # Off-axis approach pose: same standoff along the socket axis (goal x),
        # but displaced sideways and rotated in the goal frame. The charger is
        # still `standoff` away from the receptacle face, so nothing touches.
        approach_pose = (
            goal
            * sapien.Pose(
                [-standoff, params.approach_dy, params.approach_dz],
                euler2quat(0.0, np.deg2rad(params.approach_pitch_deg), np.deg2rad(params.approach_yaw_deg)),
            )
            * tcp_in_charger
        )
        _transit(planner, approach_pose, params)
        _trace(planner, base_env, "approach done")
        check()
    pre_insert_pose = goal * sapien.Pose([-standoff, 0.0, 0.0]) * tcp_in_charger

    # Align: correct from the off-axis pose onto the true socket axis, at the
    # careful speed, then refine (stock does the same refine_steps=5 here;
    # the DAgger expert skips the stationary refine).
    _set_speed(planner, speed_insert, acc_scale)
    planner.move_to_pose_with_screw(pre_insert_pose)
    check()
    if refine_steps > 0:
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=refine_steps)
    check()

    # Insert: always a straight screw move -- this is the contact phase and
    # the one part the strategy leaves alone.
    planner.move_to_pose_with_screw(insert_pose)
    _trace(planner, base_env, "insert done")
    check()

    # Pause (hold the gripper CLOSED at the insert pose -- no release, no
    # retreat; see plug_charger_largehole_allcams.py for why).
    return planner.close_gripper(t=PAUSE_STEPS)


# ---- entry points --------------------------------------------------------------


def solve(env, seed=None, debug=False, vis=False, strategy: StrategyParams | None = None):
    """Drop-in for ``plug_charger_pause_solver.solve`` (motion-planning collection).

    ``seed`` only seeds the scene; the strategy is drawn from the module's
    strategy RNG (see module docstring) unless ``strategy`` is given explicitly
    (e.g. to replay a logged one).
    """
    env.reset(seed=seed)
    base_env = env.unwrapped
    assert base_env.control_mode in ("pd_joint_pos", "pd_joint_pos_vel"), base_env.control_mode
    params = _pick(strategy)

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=1.0,  # unscaled; _run_stages rescales per stage (see _set_speed)
        joint_acc_limits=1.0,
    )
    res = _run_stages(planner, base_env, params, do_grasp=True)
    planner.close()
    return res


def _random_move(planner, base_env, params: StrategyParams, on_move_step, gripper_cmd: float | None) -> dict:
    """The RANDOM_MOVE move at a takeover (see random_move.py). Returns its log entry."""
    from vig_dagger.expert import random_move

    rng = np.random.default_rng(next_strategy_seed())
    grasped = bool(base_env.agent.is_grasping(base_env.charger)[0])
    # Plan at the expert's transit speed (mplib times the path at planning), then restore the unscaled limits so
    # _run_stages starts from the same planner state as without the move.
    speed = EXPERT_SPEED_TRANSIT if EXPERT_SPEED_TRANSIT is not None else params.speed_transit
    acc_scale = EXPERT_ACC_SCALE if EXPERT_ACC_SCALE is not None else 1.0
    _init_speed(planner)
    _set_speed(planner, speed, acc_scale)
    cfg = dict(RANDOM_MOVE)
    path, info = random_move.plan_random_move(
        planner, base_env, rng=rng, grasped=grasped, radius=float(cfg["radius"]), rot_deg=float(cfg.get("rot_deg", 5.0)),
        max_tries=int(cfg.get("max_tries", 10000)), margin=float(cfg.get("margin", 0.004)), tol=float(cfg.get("tol", 0.0005)))
    info["speed"] = round(float(speed), 4)
    info["start_step"] = int(base_env.elapsed_steps[0])
    if path is not None:
        saved = planner._on_step
        planner._on_step = on_move_step
        # Hold the gripper where the policy left it (a fresh planner would command OPEN).
        planner.gripper_state = float(gripper_cmd) if gripper_cmd is not None else (planner.CLOSED if grasped else planner.OPEN)
        planner.follow_path(path)
        planner._on_step = saved
        planner.gripper_state = planner.OPEN  # fresh-planner default; _run_stages re-asserts CLOSED if still grasped
    planner.planner.joint_vel_limits = planner._base_vel_limits
    planner.planner.joint_acc_limits = planner._base_acc_limits
    info["end_step"] = int(base_env.elapsed_steps[0])
    info["grasped_after"] = bool(base_env.agent.is_grasping(base_env.charger)[0])
    info["tcp_end"] = np.asarray(base_env.agent.tcp.pose.sp.p, dtype=float).round(5).tolist()
    used_moves.append(info)
    _trace(planner, base_env, f"random move: {info}")
    return info


def resume_and_solve(env, debug: bool = False, vis: bool = False, on_step=None, strategy: StrategyParams | None = None,
                     on_move_step=None, move_gripper: float | None = None, on_move_end=None):
    """Drop-in for ``plug_charger_pause_resumable_solver.resume_and_solve`` (DAgger expert).

    ``env`` must be the *wrapped* env so the planner's steps get recorded.
    Returns the planner's step result (last element has ``["success"]``) or
    ``None`` if ``on_step`` aborted the remaining stages (DAgger auto-return).

    Each takeover draws a fresh strategy from the module's strategy RNG (see
    module docstring) -- like a human taking over, it is unrelated to the
    scene and to any earlier takeover in the same episode -- unless
    ``strategy`` is given explicitly. The lift waypoint only happens when this
    call performs the grasp -- if the policy already holds the charger we go
    straight to align, as the pause solver does.

    With ``RANDOM_MOVE`` set and ``on_move_step`` given, the gripper first makes the random move (``_random_move``):
    ``on_move_step`` is polled after each of its steps (True stops it, e.g. at the episode's step limit; the call
    then returns None), the gripper command is held at ``move_gripper``, and ``on_move_end`` is called once the move
    is over (also when no valid target was found), before the expert's first step.
    """
    base_env = env.unwrapped
    assert base_env.control_mode in ("pd_joint_pos", "pd_joint_pos_vel"), base_env.control_mode
    params = _pick(strategy)

    planner = InterruptiblePandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=1.0,  # unscaled; _run_stages rescales per stage (see _set_speed)
        joint_acc_limits=1.0,
        on_step=on_step,
    )
    if RANDOM_MOVE is not None and on_move_step is not None:
        _random_move(planner, base_env, params, on_move_step, move_gripper)
        if on_move_end is not None:
            on_move_end()
        if planner.aborted:  # on_move_step stopped the move at the episode's step limit
            planner.close()
            return None
    # NOTE: is_grasping() reads contact forces from the *last physics step*, so
    # it is stale right after env.reset() (it can still report the previous
    # episode's grasp). Fine for DAgger, where the policy always steps before a
    # takeover; but don't call this straight after a reset in tests -- it will
    # take the already-grasped branch and skip reach/grasp. Read after the
    # random move (if any): that is the state the expert starts from.
    is_grasped = bool(base_env.agent.is_grasping(base_env.charger)[0])
    try:
        res = _run_stages(planner, base_env, params, do_grasp=not is_grasped, expert=True)
    except AbortedError:
        _trace(planner, base_env, "ABORT honoured -> handing control back to the policy")
        planner.close()
        return None
    planner.close()
    return res
