"""PlugCharger task stages and failure flags per step, from simulator state only (for analysis, never as a trigger).

Positions are in the socket frame: x along the insertion axis, 0 = seated, the pins reach the socket face at
x = -16 mm; "off axis" = distance from the axis in y-z.

  0 Reach      gripper more than 60 mm from the plug, plug not held
  1 Grasp      gripper within 60 mm of the plug on the table, until the plug is held
  2 Lift       plug held, less than 20 mm above its resting height
  3 Transport  plug held and lifted, not yet near the socket
  4 Align      plug held at the socket: from 90 mm in front of the face to past it (x >= -90 mm), within 60 mm of the
               axis, not inserted (the approach, and a plug stuck at the socket: pressed beside the hole, or past the
               face but tilted or off the axis)
  5 Insert     pins past the face (x > -13 mm), within 8 mm of the axis and tilted at most 11.5 deg from it
  6 Seated     the task's success (within 5 mm and 0.2 rad of the goal)

Flags: dropped (held for 3+ steps, then not, not seated), misaligned (pins at the face, x -19..-13 mm, more than 6 mm
off the axis), tilted (at or past the face, not seated, tilted more than 11.5 deg), off_center (the plug held 5 mm or
more from the demo grasp, 15 steps after the first grasp; marked on every held step), stalled (not seated and
neither plug nor gripper moved 2 mm over the last 30 steps).

Raw labels are smoothed: a run shorter than 3 steps takes the label before it. Stages can go back (a regrasp after a
drop); episode summaries use the furthest stage reached.
"""

from __future__ import annotations

import numpy as np

STAGES = ["Reach", "Grasp", "Lift", "Transport", "Align", "Insert", "Seated"]
FLAGS = {"dropped": "Dropped the plug", "misaligned": "Misaligned at the socket face", "tilted": "Tilted at the socket face",
         "off_center": "Off-center grasp", "stalled": "Stalled"}
REF_GRASP_MM = np.array([21.0, 0.0, 5.0])  # the demos' grasp: plug position in the TCP frame


def quat_to_matrix(q):  # wxyz -> (3, 3); q may be (..., 4)
    q = np.asarray(q, np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def smooth(raw: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Runs shorter than min_run take the label before them (the first and the last run are kept)."""
    out = raw.copy()
    start = 0
    for t in range(1, len(raw) + 1):
        if t == len(raw) or raw[t] != raw[start]:
            if t - start < min_run and start > 0 and t < len(raw):
                out[start:t] = out[start - 1]
            start = t
    return out


def moved(p: np.ndarray, window: int) -> np.ndarray:
    """Per step, how far p moved over the last `window` steps (0 for the first steps)."""
    d = np.zeros(len(p))
    d[window:] = np.linalg.norm(p[window:] - p[:-window], axis=-1)
    return d


def runs_after(cond: np.ndarray, length: int) -> np.ndarray:
    """True from the step where cond has held for `length` steps in a row, while it keeps holding."""
    out, run = np.zeros(len(cond), bool), 0
    for t, c in enumerate(cond):
        run = run + 1 if c else 0
        out[t] = run >= length
    return out


def label(state: dict[str, np.ndarray], success: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stage index (into STAGES) and flag masks per observation of one episode.

    Args:
        state: ``TaskStateLog.arrays()``: goal_pose (7,), charger_pose / tcp_pose (N, 7), held, dist, angle (N,).
        success: Whether the episode succeeded (its last observation then counts as seated).
    """
    gp = np.asarray(state["goal_pose"], np.float64)
    Rg = quat_to_matrix(gp[3:7])
    plug = np.asarray(state["charger_pose"], np.float64)
    tcp = np.asarray(state["tcp_pose"], np.float64)
    rel = (plug[:, :3] - gp[:3]) @ Rg * 1000  # socket frame, mm
    x, off = rel[:, 0], np.hypot(rel[:, 1], rel[:, 2])
    # angle between the plug's insertion axis and the socket's (roll about the axis does not count)
    tilt = np.degrees(np.arccos(np.clip(quat_to_matrix(plug[:, 3:7])[:, :, 0] @ Rg[:, 0], -1.0, 1.0)))
    held = np.asarray(state["held"], bool).copy()
    held[0] = False
    seated = (np.asarray(state["dist"]) <= 0.005) & (np.asarray(state["angle"]) <= 0.2)
    lift = (plug[:, 2] - plug[0, 2]) * 1000
    near_plug = np.linalg.norm(tcp[:, :3] - plug[:, :3], axis=1) * 1000 < 60
    raw = np.zeros(len(x), np.int64)
    raw[near_plug & ~held] = 1
    raw[held & (lift < 20)] = 2
    raw[held & (lift >= 20)] = 3
    raw[held & (x >= -90) & (off <= 60)] = 4
    raw[(x >= -13) & (off < 8) & (tilt <= 11.5)] = 5
    raw[seated] = 6
    stage = smooth(raw)
    if success:
        stage[-1] = 6
    was_held = np.maximum.accumulate(runs_after(held, 3))
    flags = {
        "dropped": was_held & ~held & ~seated,
        "misaligned": (x >= -19) & (x <= -13) & (off > 6),
        "tilted": (x >= -19) & (tilt > 11.5) & ~seated,
        "stalled": ~seated & (moved(plug[:, :3], 30) < 0.002) & (moved(tcp[:, :3], 30) < 0.002) & (np.arange(len(x)) >= 30),
    }
    off_center = np.zeros(len(x), bool)
    if held.any():
        t = min(int(np.argmax(held)) + 15, len(held) - 1)
        in_tcp = quat_to_matrix(tcp[t, 3:7]).T @ (plug[t, :3] - tcp[t, :3]) * 1000
        if np.linalg.norm(in_tcp - REF_GRASP_MM) >= 5:
            off_center = held.copy()
    flags["off_center"] = off_center
    return stage, flags


def summary(stage: np.ndarray, flags: dict[str, np.ndarray], success: bool) -> dict:
    """Furthest stage reached, the stage a failed episode ended in, and which flags were ever raised."""
    furthest = int(stage.max())
    return {
        "furthest_stage": STAGES[furthest],
        "final_stage": STAGES[int(stage[-1])],
        "failed_at": None if success else STAGES[furthest],
        **{f"flag_{k}": bool(v.any()) for k, v in flags.items()},
    }
