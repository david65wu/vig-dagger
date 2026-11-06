"""Recording episodes in ManiSkill's trajectory format, plus the per-step task state the analysis needs.

``SplitRecordEpisode``: ManiSkill's ``RecordEpisode`` writes to one file chosen at construction, but whether an
episode succeeded is known only at its end. This variant buffers the episode and flushes it to a success or a
failure file afterwards (swapping the file handles ``flush_trajectory`` writes to), and can attach extra groups
(arrays per step) to the written ``traj_<n>`` group.

``TaskStateLog``: a wrapper below the recorder that logs, at the reset observation and after every step, the state
stage labels are computed from (``stages.py``) and who issued the step (``controller``: 0 policy, 1 expert, 2 the
random end-effector move at a takeover).

Trajectory layout (ManiSkill): ``traj_<n>/actions (T, 8)``, ``obs/agent/qpos (T+1, 9)``,
``obs/sensor_data/<camera>/rgb (T+1, H, W, 3)``, ``env_states/...``, ``success (T,)``; the converter
(``dataset.py``) pairs obs[t] with actions[t] and drops the last observation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
from mani_skill import get_commit_info
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers.record import RecordEpisode, clean_trajectories, parse_env_info

POLICY, EXPERT, MOVE = 0, 1, 2  # who issued a step (MOVE: the random move at a takeover, before the expert)


class TaskStateLog(gym.Wrapper):
    """Per-observation task state of a single-env episode, reset at every ``reset``.

    Arrays (index = observation, 0 = after reset): ``held`` (plug grasped; False at index 0, where the contact
    reading is stale), ``charger_pose`` and ``tcp_pose`` (7: position, quaternion wxyz), ``dist`` / ``angle`` (plug to
    goal, m / rad); per step (index = action): ``controller``. ``goal_pose`` is fixed per episode.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.controller = POLICY
        self._rows: dict[str, list] = {}
        self.goal_pose = np.zeros(7, np.float32)

    def _snapshot(self, first: bool) -> None:
        base = self.env.unwrapped
        d, a = base._compute_distance()
        self._rows["held"].append(False if first else bool(base.agent.is_grasping(base.charger)[0]))
        self._rows["charger_pose"].append(base.charger.pose.raw_pose[0].cpu().numpy().astype(np.float32))
        self._rows["tcp_pose"].append(base.agent.tcp.pose.raw_pose[0].cpu().numpy().astype(np.float32))
        self._rows["dist"].append(float(d[0]))
        self._rows["angle"].append(float(a[0]))

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self._rows = {k: [] for k in ("held", "charger_pose", "tcp_pose", "dist", "angle", "controller")}
        self.controller = POLICY
        self.goal_pose = self.env.unwrapped.goal_pose.raw_pose[0].cpu().numpy().astype(np.float32)
        self._snapshot(first=True)
        return out

    def step(self, action):
        out = super().step(action)
        self._rows["controller"].append(self.controller)
        self._snapshot(first=False)
        return out

    @property
    def num_steps(self) -> int:
        return len(self._rows.get("controller", []))

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self._rows.items()}
        out["held"] = out["held"].astype(bool)
        out["controller"] = out["controller"].astype(np.int8)
        out["goal_pose"] = self.goal_pose
        return out


class SplitRecordEpisode(RecordEpisode):
    """RecordEpisode that decides the destination file (success or failure) when the episode is flushed."""

    def __init__(self, env, success_dir, fail_dir, *, trajectory_name=None, save_video=False, **kwargs):
        super().__init__(env, output_dir=success_dir, trajectory_name=trajectory_name, save_video=save_video,
                         save_on_reset=False, **kwargs)
        self._success_state = self._snapshot()
        fail_dir = Path(fail_dir)
        fail_dir.mkdir(parents=True, exist_ok=True)
        traj_name = trajectory_name or time.strftime("%Y%m%d_%H%M%S")
        fail_h5 = h5py.File(fail_dir / f"{traj_name}.h5", "w")
        fail_json = dict(env_info=parse_env_info(self.env), commit_info=get_commit_info(), episodes=[],
                         source_type=kwargs.get("source_type"), source_desc="Unsuccessful attempts, not training data.")
        if fail_json["env_info"] is not None:
            fail_json["env_info"]["max_episode_steps"] = self.max_episode_steps
        self._fail_state = dict(h5_file=fail_h5, json_path=fail_h5.filename.replace(".h5", ".json"), json_data=fail_json,
                                episode_id=-1, video_id=-1, output_dir=fail_dir)

    def _snapshot(self):
        return dict(h5_file=self._h5_file, json_path=self._json_path, json_data=self._json_data,
                    episode_id=self._episode_id, video_id=self._video_id, output_dir=self.output_dir)

    def _apply(self, state):
        self._h5_file, self._json_path, self._json_data = state["h5_file"], state["json_path"], state["json_data"]
        self._episode_id, self._video_id, self.output_dir = state["episode_id"], state["video_id"], state["output_dir"]

    def flush_episode(self, *, success: bool, save: bool, extra_info: dict | None = None,
                      extra_groups: dict[str, dict[str, np.ndarray]] | None = None) -> int | None:
        """Write the buffered episode to the success or failure file (or drop it when save is False).

        Returns the written ``traj_<n>`` index, or None when nothing was written.
        """
        self._apply(self._success_state if success else self._fail_state)
        self.flush_trajectory(save=save)
        index = None
        if save:
            index = self._episode_id
            if extra_info:
                self._json_data["episodes"][-1].update(extra_info)
                dump_json(self._json_path, self._json_data, indent=2)
            group = self._h5_file[f"traj_{index}"]
            for name, arrays in (extra_groups or {}).items():
                sub = group.create_group(name, track_order=True)
                for key, value in arrays.items():
                    sub.create_dataset(key, data=np.asarray(value))
            self._h5_file.flush()
        if self.save_video:
            self.flush_video(save=save)
        if success:
            self._success_state = self._snapshot()
        else:
            self._fail_state = self._snapshot()
        self._apply(self._success_state)
        return index

    def close(self) -> None:
        fail = self._fail_state
        if self.clean_on_close:
            clean_trajectories(fail["h5_file"], fail["json_data"])
        dump_json(fail["json_path"], fail["json_data"], indent=2)
        fail["h5_file"].close()
        super().close()


def write_arrays(path: Path, group: str, arrays: dict[str, Any], attrs: dict[str, Any] | None = None) -> None:
    """Append one group of arrays (plus scalar attributes) to an HDF5 file, replacing a group of the same name."""
    with h5py.File(path, "a") as f:
        if group in f:
            del f[group]
        g = f.create_group(group)
        for key, value in arrays.items():
            g.create_dataset(key, data=np.asarray(value))
        for key, value in (attrs or {}).items():
            g.attrs[key] = value
