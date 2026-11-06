"""The random end-effector move at a DAgger takeover (variant ``random_move``), before the expert replans.

At every takeover (``planner.resume_and_solve`` with ``planner.RANDOM_MOVE`` set) the gripper tip (TCP) first moves
to a random pose: a position drawn uniformly inside a ball of ``radius`` around its current position, and its
orientation turned by an angle drawn uniformly in [0, ``rot_deg``] about a uniformly random axis through the TCP.
The gripper command is held at the policy's last command (a held charger moves with the hand). The move is a screw
motion at the transit speed of the strategy the expert drew for this takeover; no query score is computed on its
steps, but they are recorded as part of the episode (driver code 2, ``recording.MOVE``).

A target is rejected and redrawn (up to ``max_tries``; 0.6 ms per geometric check, so 10,000 tries bound the search
at ~6 s -- a plug with its pins in the hole has ~0.05 % valid draws, all near-axial pull-outs) when the screw plan
fails (IK / joint limits / self-collision) or when, anywhere along the straight screw path, the moving geometry would
enter an obstacle. The moving geometry is what moves rigidly with the TCP while only the arm joints change: the
hand, both fingers, and the charger when it is grasped (their actual collision shapes, read from the sim).
Obstacles: the receptacle's wall boxes, the table (z = 0), and the charger when it is not grasped. "Enter" is judged
against the start: a point may not get more than ``tol`` deeper into an obstacle grown by ``margin`` than it
already is, so a plug resting with its pins in the hole (already within the margin of the walls) can still be
pulled straight out, while a sideways move that would push the pins into the walls is rejected. If no target is
valid the move is skipped for that takeover (logged with ``skipped: true``).

Ported unchanged from the earlier PlugCharger experiments (maniskill ``scripts/random_eef_perturbation.py``,
2026-11-04); ``tests/vig_dagger/test_random_move.py`` checks the geometry.
"""

from __future__ import annotations

import numpy as np
import sapien
from transforms3d.quaternions import axangle2quat, qmult

MOVING_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
PATH_SAMPLES = 12  # poses checked along the straight screw path, start and end included


def _mat(pose: sapien.Pose) -> np.ndarray:
    return pose.to_transformation_matrix()


def _box_surface(half: np.ndarray, n: int = 4) -> np.ndarray:
    """n x n grid on every face of a box with half sizes ``half`` (corners and edges included)."""
    g = np.linspace(-1.0, 1.0, n)
    u, v = np.meshgrid(g, g)
    faces = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        for sign in (-1.0, 1.0):
            p = np.zeros((u.size, 3))
            p[:, ax] = sign
            p[:, a] = u.ravel()
            p[:, b] = v.ravel()
            faces.append(p)
    return np.unique(np.concatenate(faces), axis=0) * np.asarray(half, dtype=float)


def _collision_component(entity):
    for c in entity.get_components():
        if hasattr(c, "get_collision_shapes"):
            return c
    return None


def _shapes(component, entity_pose: sapien.Pose):
    """World-frame (points, boxes) of a component's collision shapes. boxes: list of (T_world_box, half)."""
    pts, boxes = [], []
    for s in component.get_collision_shapes():
        T = _mat(entity_pose) @ _mat(s.get_local_pose())
        if hasattr(s, "get_half_size"):
            half = np.asarray(s.get_half_size(), dtype=float)
            local = _box_surface(half)
            boxes.append((T, half))
        elif hasattr(s, "get_vertices"):
            local = np.asarray(s.get_vertices(), dtype=float) * np.asarray(s.get_scale(), dtype=float)
        else:
            continue
        pts.append(local @ T[:3, :3].T + T[:3, 3])
    return (np.concatenate(pts) if pts else np.zeros((0, 3))), boxes


def scene_geometry(base_env, grasped: bool) -> dict:
    """Moving points/boxes (hand, fingers, held charger) and obstacles (receptacle, free charger), world frame."""
    mov_pts, mov_boxes = [], []
    for name in MOVING_LINKS:
        link = base_env.agent.robot.links_map[name]
        p, b = _shapes(link._objs[0], link.pose.sp)
        mov_pts.append(p)
        mov_boxes += b
    obs_boxes, obs_pts = [], []
    rec_p, rec_b = _shapes(_collision_component(base_env.receptacle._objs[0]), base_env.receptacle.pose.sp)
    obs_boxes += rec_b
    obs_pts.append(rec_p)
    ch_p, ch_b = _shapes(_collision_component(base_env.charger._objs[0]), base_env.charger.pose.sp)
    if grasped:
        mov_pts.append(ch_p)
        mov_boxes += ch_b
    else:
        obs_boxes += ch_b
        obs_pts.append(ch_p)
    return dict(mov_pts=np.concatenate(mov_pts), mov_boxes=mov_boxes, obs_boxes=obs_boxes, obs_pts=np.concatenate(obs_pts))


def _depth_in_box(pts: np.ndarray, T: np.ndarray, half: np.ndarray, margin: float) -> np.ndarray:
    """How deep each point is inside the box grown by ``margin`` (0 outside)."""
    local = (pts - T[:3, 3]) @ T[:3, :3]
    d = (half + margin) - np.abs(local)
    return np.where((d > 0).all(axis=1), d.min(axis=1), 0.0)


def _rigid(pts: np.ndarray, center: np.ndarray, R: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (pts - center) @ R.T + center + d


def _rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def path_is_clear(geom: dict, center: np.ndarray, d: np.ndarray, axis: np.ndarray, angle: float,
                  margin: float, tol: float, table_z: float = 0.0) -> tuple[bool, str]:
    """Check the straight screw path from the current pose to (center + d, rotated by angle about axis)."""
    mp, op = geom["mov_pts"], geom["obs_pts"]
    # Depth at the start, per point and obstacle (the reference a point may not go deeper than).
    d0_obs = [_depth_in_box(mp, T, h, margin) for T, h in geom["obs_boxes"]]
    d0_tab = np.maximum(0.0, table_z + margin - mp[:, 2])
    d0_rev = [_depth_in_box(op, T, h, margin) for T, h in geom["mov_boxes"]]
    for s in np.linspace(0.0, 1.0, PATH_SAMPLES)[1:]:
        R, ds = _rotmat(axis, s * angle), s * d
        q = _rigid(mp, center, R, ds)
        if (np.maximum(0.0, table_z + margin - q[:, 2]) > d0_tab + tol).any():
            return False, "table"
        for (T, h), dep0 in zip(geom["obs_boxes"], d0_obs):
            if (_depth_in_box(q, T, h, margin) > dep0 + tol).any():
                return False, "obstacle"
        # Obstacle points entering the moving boxes (catches thin edges between the moving points).
        for (T, h), dep0 in zip(geom["mov_boxes"], d0_rev):
            Ts = T.copy()
            Ts[:3, :3] = R @ T[:3, :3]
            Ts[:3, 3] = (T[:3, 3] - center) @ R.T + center + ds
            if (_depth_in_box(op, Ts, h, margin) > dep0 + tol).any():
                return False, "obstacle"
    return True, "ok"


def sample_offset(rng: np.random.Generator, radius: float, rot_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform point in the ball, uniform axis, angle uniform in [0, rot_deg]."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    d = direction * radius * rng.random() ** (1.0 / 3.0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rot_deg) * rng.random()
    return d, axis, float(angle)


def plan_random_move(planner, base_env, *, rng: np.random.Generator, grasped: bool, radius: float, rot_deg: float,
                     max_tries: int = 10000, margin: float = 0.004, tol: float = 0.0005):
    """Draw targets until one is reachable and collision-free. Returns (screw path or None, info dict).

    The path is planned with the planner's current joint limits, so set the speed first.
    """
    tcp = base_env.agent.tcp.pose.sp
    center = np.asarray(tcp.p, dtype=float)
    geom = scene_geometry(base_env, grasped)
    reasons: dict[str, int] = {}
    for k in range(1, max_tries + 1):
        d, axis, angle = sample_offset(rng, radius, rot_deg)
        ok, why = path_is_clear(geom, center, d, axis, angle, margin, tol)
        if ok:
            q = qmult(axangle2quat(axis, angle), np.asarray(tcp.q, dtype=float))
            target = sapien.Pose(center + d, q)
            path = planner.move_to_pose_with_screw(target, dry_run=True)
            if isinstance(path, int) and path == -1:
                ok, why = False, "plan"
        reasons[why] = reasons.get(why, 0) + 1
        if ok:
            return path, dict(skipped=False, tries=k, rejects={r: n for r, n in reasons.items() if r != "ok"},
                              grasped=bool(grasped), tcp_start=center.round(5).tolist(), offset=d.round(5).tolist(),
                              distance=round(float(np.linalg.norm(d)), 5), rot_axis=axis.round(4).tolist(),
                              rot_deg=round(float(np.rad2deg(angle)), 3), path_steps=int(len(path["position"])))
    return None, dict(skipped=True, tries=max_tries, rejects=reasons, grasped=bool(grasped), tcp_start=center.round(5).tolist())
