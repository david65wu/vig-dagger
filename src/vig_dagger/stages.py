"""OpenDrawer task stages and failure flags per step, from simulator state only (for analysis, never as a trigger).

Ordered by the task's four success latches (drawer opened at 0.18 m, charger picked = lifted 4 cm, charger placed in
the drawer, drawer closed):

   0 Reach handle      drawer not yet opened, gripper more than 12 cm from the handle
   1 Handle approach   within 12 cm of the handle (until more than 15 cm away again)
   2 Grasp handle      within 5 cm of the handle (until more than 7 cm away again), until holding it
   3 Pull              holding the handle (before or after the drawer counts as opened)
   4 Release           drawer opened, handle let go, gripper still within 10 cm of it, until it first moves away
   5 Reach charger     gripper more than 12 cm from the charger
   6 Charger approach  within 12 cm of the charger (until more than 15 cm away again)
   7 Grasp charger     within 4 cm of the charger (until more than 6 cm away again), until holding it
   8 Lift              holding the charger, not yet counted as picked
   9 Carry             charger picked and held, not over the open drawer
  10 Lower, release    charger over the open drawer (held or just released), not yet counted as placed
  11 Retreat           charger placed, gripper still within 10 cm of it
  12 Push approach     charger placed, moving to the drawer front
  13 Push              from the drawer's first closing movement after the placement until it is closed
  14 Hold              drawer closed (the success hold)

Flags: missed handle / missed charger (fingers shut on nothing within 6 cm of it), handle slip (stopped holding the
handle after 5+ steps before the drawer counts as opened), charger knocked (moved 2 cm or more in 10 steps while not
held, before it counts as picked), charger dropped (held for 5+ steps, then not, outside the drawer), stalled
(gripper moved less than 2 cm over the last 50 steps, drawer not closed).

Raw labels are smoothed: a run shorter than 3 steps takes the label before it. Stages can go back (a regrasp);
episode summaries use the furthest stage reached.
"""

from __future__ import annotations

import numpy as np

STAGES = ["Reach handle", "Handle approach", "Grasp handle", "Pull", "Release", "Reach charger", "Charger approach",
          "Grasp charger", "Lift", "Carry", "Lower, release", "Retreat", "Push approach", "Push", "Hold"]
FLAGS = {"missed_handle": "Missed the handle", "handle_slip": "Handle slipped", "missed_charger": "Missed the charger",
         "charger_knocked": "Knocked the charger", "charger_dropped": "Dropped the charger", "stalled": "Stalled"}
LATCHES = ("drawer_opened", "charger_picked", "charger_placed", "drawer_closed")
# Per-step task state the workers log (TaskStateLog in isaac.py) and the labels read.
STATE_KEYS = ("goal_pose", "drawer_open", "tcp_pose", "charger_pose", "qpos", "holding_handle", "holding_charger",
              "stages")
HANDLE_CENTER_B = np.array([0.19, 0.0, 0.0062])  # handle centre in the drawer body frame (planner.HANDLE_CENTER_B)
DRAWER_X_B = (-0.1675, 0.1775)  # drawer interior along its axis, body frame
DRAWER_Y_B = 0.24


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


def band(d: np.ndarray, enter: float, leave: float) -> np.ndarray:
    """True from where d drops below enter until it rises above leave (hysteresis)."""
    out, on = np.zeros(len(d), bool), False
    for t, v in enumerate(d):
        on = v < (leave if on else enter)
        out[t] = on
    return out


def label(state: dict[str, np.ndarray], success: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stage index (into STAGES) and flag masks per observation of one episode.

    Args:
        state: ``TaskStateLog.arrays()``: goal_pose (7: handle centre at the start + drawer orientation, wxyz),
            drawer_open (N), tcp_pose / charger_pose (N, 7), qpos (N, 9), holding_handle / holding_charger (N),
            stages (N, 4: the success latches).
        success: Whether the episode succeeded (its last observation then counts as the hold stage).
    """
    goal = np.asarray(state["goal_pose"], np.float64)
    Rd = quat_to_matrix(goal[3:7])
    open_dir = Rd[:, 0]
    opening = np.asarray(state["drawer_open"], np.float64)
    handle = goal[:3] + opening[:, None] * open_dir
    drawer_origin = handle - Rd @ HANDLE_CENTER_B
    tcp = np.asarray(state["tcp_pose"], np.float64)[:, :3]
    charger = np.asarray(state["charger_pose"], np.float64)[:, :3]
    gap = np.asarray(state["qpos"], np.float64)[:, 7:9].sum(axis=1)
    hold_h, hold_c = np.asarray(state["holding_handle"], bool), np.asarray(state["holding_charger"], bool)
    latch = np.asarray(state["stages"], bool)
    opened, picked, placed, closed = (latch[:, i] for i in range(4))
    d_handle = np.linalg.norm(tcp - handle, axis=1)
    d_charger = np.linalg.norm(tcp - charger, axis=1)
    c_in = (charger - drawer_origin) @ Rd  # charger in the drawer body frame
    over_drawer = (c_in[:, 0] > DRAWER_X_B[0]) & (c_in[:, 0] < DRAWER_X_B[1]) & (np.abs(c_in[:, 1]) < DRAWER_Y_B)
    T = len(tcp)
    closing = np.zeros(T, bool)
    closing[5:] = opening[5:] - opening[:-5] <= -0.002
    near_h, grasp_h = band(d_handle, 0.12, 0.15), band(d_handle, 0.05, 0.07)
    near_c, grasp_c = band(d_charger, 0.12, 0.15), band(d_charger, 0.04, 0.06)
    raw = np.zeros(T, np.int64)
    left_handle = pushing = False  # Release ends once the gripper moves away; Push holds until closed
    for t in range(T):
        if closed[t]:
            s = 14
        elif placed[t]:
            pushing = pushing or bool(closing[t])
            s = 13 if pushing else (11 if d_charger[t] < 0.10 else 12)
        elif picked[t]:
            s = 10 if over_drawer[t] else (9 if hold_c[t] else (7 if grasp_c[t] else 6 if near_c[t] else 5))
        elif opened[t]:
            if hold_h[t]:
                s = 3
            elif hold_c[t]:
                s = 8
            elif grasp_c[t]:
                s = 7
            elif near_c[t]:
                s = 6
            elif d_handle[t] < 0.10 and not left_handle:
                s = 4
            else:
                s = 5
            left_handle = left_handle or s in (5, 6, 7, 8)
        else:
            s = 3 if hold_h[t] else (2 if grasp_h[t] else 1 if near_h[t] else 0)
        raw[t] = s
    stage = smooth(raw)
    if success:
        stage[-1] = 14
    was_h = np.maximum.accumulate(runs_after(hold_h, 5))
    was_c = np.maximum.accumulate(runs_after(hold_c, 5))
    knocked = np.zeros(T, bool)
    knocked[10:] = np.linalg.norm(charger[10:, :2] - charger[:-10, :2], axis=1) >= 0.02
    flags = {
        "missed_handle": (d_handle < 0.06) & (gap < 0.004) & ~hold_h & ~opened,
        "handle_slip": was_h & ~hold_h & ~opened,
        "missed_charger": (d_charger < 0.06) & (gap < 0.02) & ~hold_c & ~picked,
        "charger_knocked": knocked & ~hold_c & ~picked,
        "charger_dropped": was_c & ~hold_c & ~over_drawer & ~placed,
        "stalled": (moved(tcp, 50) < 0.02) & (np.arange(T) >= 50) & ~closed,
    }
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
