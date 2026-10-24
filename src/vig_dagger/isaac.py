"""The Isaac Sim side of OpenDrawer's simulator jobs (collection, DAgger, evaluation), run with ``.venv-isaac``.

A worker runs its body with ``run(body, args)``: it starts the simulator app first (Isaac Lab can only be imported
once the app runs) and makes the process's exit code tell success from failure (closing the app always exits with
0). Then ``make_env()`` builds the task with one environment, CPU physics, and only the policy's cameras (``left``
and ``wrist``; the scene's other three cameras are dropped, which changes nothing else and saves rendering and
storage).

The policy's state is ``ee_pose`` (7, as observed) plus the last two relative joint positions, the training data's
``robot0_eef_pos/quat/gripper_qpos``. ``ee_pose`` is the TCP only if the process runs with ``PYTHONHASHSEED=0`` (Isaac
Lab keeps the hand's frames in a set whose order depends on the hash seed); ``make_env`` refuses to run otherwise.
The job queue sets it for every simulator job.

Success follows the teleop recorder: the env's success term holds for 10 steps in a row, then 14 more steps are
recorded, within ``max_episode_steps`` (1200).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

TASK = "OpenDrawer-ManagerBased-v0"
UNUSED_CAMERAS = ("teleop", "right", "top")
LATCHES = ("drawer_opened", "charger_picked", "charger_placed", "drawer_closed")
NUM_SUCCESS_STEPS, SUCCESS_HOLD_STEPS = 10, 14


def launch():
    """Start Isaac Sim headless with cameras and CPU physics; returns the app (close it at the end)."""
    import os

    from isaaclab.app import AppLauncher

    from vig_dagger.jobs import exit_on_sigterm

    if os.environ.get("TMPDIR"):  # the job's own temporary directory (plan.isaac_job)
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    app = AppLauncher(dict(headless=True, enable_cameras=True, device="cpu")).app
    exit_on_sigterm()  # again: Kit installs its own handlers, and a worker must close its HDF5 files on SIGTERM
    return app


def run(body, args) -> None:
    """Run a worker's ``body(args)`` inside Isaac Sim. Closing the app ends the process with exit code 0 whatever
    happened, so a failure is printed and the process exits with a nonzero code without closing the app (the
    body's own ``finally`` blocks have closed its files by then)."""
    import os
    import sys
    import traceback

    app = launch()
    code = 0
    try:
        body(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exc.code is not None and not isinstance(exc.code, int):
            print(exc.code, flush=True)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if code:
        os._exit(code)
    app.close()


def make_env(record_dir: Path | None = None, record_name: str | None = None):
    """The task as one environment; with ``record_dir`` it records demos for export (``Recorder``).

    Returns:
        (env, success_term): the unwrapped ``ManagerBasedRLEnv`` and the env's success term (removed from the
        terminations, so the episode loop decides when an episode ends).
    """
    import gymnasium as gym

    import vig_dagger.opendrawer_env  # noqa: F401  (registers the task)
    from vig_dagger.opendrawer_env.open_drawer_env_cfg import OpenDrawerEnvCfg

    cfg = OpenDrawerEnvCfg()
    cfg.sim.device = "cpu"
    cfg.scene.num_envs = 1
    cfg.env_name = TASK
    success = cfg.terminations.success
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    cfg.observations.policy.concatenate_terms = False
    for cam in UNUSED_CAMERAS:
        setattr(cfg.scene, f"{cam}_cam", None)
        setattr(cfg.observations.rgb_camera, f"{cam}_cam", None)
    if record_dir is not None:
        _add_recorders(cfg, Path(record_dir), record_name or "demos")
    env = gym.make(TASK, cfg=cfg).unwrapped
    frame = env.scene["ee_frame"].data.target_frame_names[0]
    if frame != "ee_tcp":
        raise SystemExit(f"obs ee_pose is the '{frame}' frame in this process; the policy's state needs ee_tcp. "
                         "Run simulator jobs with PYTHONHASHSEED=0 (the job queue does).")
    return env, success


def reset(env, seed: int):
    """Reset the simulation and the scene to ``seed``; returns the first observation."""
    env.sim.reset()
    if env.recorder_manager is not None:
        env.recorder_manager.reset()
    obs, _ = env.reset(seed=seed)
    return obs


# ---------------------------------------------------------------- policy inputs


def policy_state(obs: dict, device: torch.device | str) -> torch.Tensor:
    """(1, 9): ee_pose (7, as observed) + the last two relative joint positions (the fingers)."""
    ee = obs["pose"]["ee_pose"].reshape(1, -1)
    joint_pos = obs["policy"]["joint_pos"].reshape(1, -1)
    return torch.cat([ee, joint_pos[:, -2:]], dim=-1).to(device=device, dtype=torch.float32)


def camera_images(obs: dict, cameras: list[str] | tuple[str, ...], device: torch.device | str) -> torch.Tensor:
    """(1, n_cams, 3, H, W) undegraded RGB uint8 frames in the policy's camera order."""
    rgb = obs["rgb_camera"]
    return torch.stack([rgb[f"{c}_cam"][0].permute(2, 0, 1) for c in cameras])[None].to(device)


# ---------------------------------------------------------------- success and task state


def latches(env) -> dict[str, bool]:
    stages = getattr(env, "_open_drawer_success_stages", None)
    if stages is None or int(env.episode_length_buf[0]) == 0:
        return dict.fromkeys(LATCHES, False)
    return {name: bool(stages[name][0]) for name in LATCHES}


class SuccessRule:
    """The teleop recorder's success: the success term for NUM_SUCCESS_STEPS steps in a row, then SUCCESS_HOLD_STEPS
    more steps. Call ``update()`` after every step; it returns "success" once the hold is over, else None."""

    def __init__(self, env, term) -> None:
        self.env, self.term = env, term
        self.in_a_row, self.hold_left = 0, None

    def update(self) -> str | None:
        ok = bool(self.term.func(self.env, **self.term.params)[0])  # also keeps the stage latches current
        if self.hold_left is None:
            self.in_a_row = self.in_a_row + 1 if ok else 0
            if self.in_a_row < NUM_SUCCESS_STEPS:
                return None
            self.hold_left = SUCCESS_HOLD_STEPS  # the hold counts down from this step, as in the teleop recorder
        if self.hold_left <= 0:
            return "success"
        self.hold_left -= 1
        return None

    @property
    def concluded(self) -> bool:
        """The success term held long enough; the remaining steps only finish the recording."""
        return self.hold_left is not None


class TaskStateLog:
    """Per-observation task state of one episode for the stage labels (``stages.STATE_KEYS``) and the controller per
    step (0 policy, 1 expert). Call ``reset(env)`` after the env's reset and ``step(env, controller)`` after every
    step, after ``SuccessRule.update`` (which updates the stage latches)."""

    def __init__(self) -> None:
        self.rows: dict[str, list] = {}
        self.goal_pose = np.zeros(7, np.float32)
        self._geo = None

    def _snapshot(self, env) -> None:
        g = self._geo
        tcp_pos, tcp_quat = g.tcp_pose()
        ch_pos, ch_quat = g.charger_pose()
        now = latches(env)
        self.rows["tcp_pose"].append(torch.cat([tcp_pos, tcp_quat]).cpu().numpy().astype(np.float32))
        self.rows["charger_pose"].append(torch.cat([ch_pos, ch_quat]).cpu().numpy().astype(np.float32))
        self.rows["drawer_open"].append(float(g.drawer_open()))
        self.rows["qpos"].append(g._robot.data.joint_pos[0].cpu().numpy().astype(np.float32))
        self.rows["holding_handle"].append(bool(g.holding_handle()))
        self.rows["holding_charger"].append(bool(g.holding_charger()))
        self.rows["stages"].append([now[k] for k in LATCHES])

    def reset(self, env) -> None:
        from vig_dagger.expert.planner import OpenDrawerPlanner, StrategyParams

        self._geo = OpenDrawerPlanner(env, StrategyParams.sample(0), expert=True, trace=False)  # geometry helpers only
        self.rows = {k: [] for k in ("tcp_pose", "charger_pose", "drawer_open", "qpos", "holding_handle",
                                     "holding_charger", "stages", "controller")}
        _, quat = self._geo.drawer_frame()
        self.goal_pose = torch.cat([self._geo.handle_center(), quat]).cpu().numpy().astype(np.float32)
        self._snapshot(env)

    def step(self, env, controller: int) -> None:
        self.rows["controller"].append(int(controller))
        self._snapshot(env)

    @property
    def num_steps(self) -> int:
        return len(self.rows.get("controller", []))

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        out["controller"] = out["controller"].astype(np.int8)
        for key in ("holding_handle", "holding_charger", "stages"):
            out[key] = out[key].astype(bool)
        out["goal_pose"] = self.goal_pose
        return out


# ---------------------------------------------------------------- recording demos


def _add_recorders(cfg, out_dir: Path, name: str) -> None:
    """Record every step like the teleop recorder: actions, the policy / pose / rgb_camera observation groups before
    each step, sim states, and the success latches; ``Recorder.export`` writes a finished episode."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
    from isaaclab.utils import configclass

    class PreStepGroup(RecorderTerm):
        def record_pre_step(self):
            return self.cfg.output_key, self._env.obs_buf[self.cfg.observation_group_name]

    @configclass
    class PreStepGroupCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = PreStepGroup
        observation_group_name: str = "policy"
        output_key: str = "obs"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(out_dir)
    cfg.recorders.dataset_filename = name
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    for group in ("pose", "rgb_camera"):
        setattr(cfg.recorders, f"record_pre_step_{group}_observations",
                PreStepGroupCfg(observation_group_name=group, output_key=f"obs/{group}"))


class Recorder:
    """Exports recorded episodes of an env built with ``make_env(record_dir=...)`` as ``data/demo_<i>`` groups of
    ``<record_dir>/<name>.hdf5``, with attributes (scene seed, strategy, outcome ...) and extra per-step groups."""

    def __init__(self, env) -> None:
        self.env = env

    def step(self) -> None:
        """Record the success latches of the step just taken (call after every step, after ``SuccessRule.update``)."""
        stages = getattr(self.env, "_open_drawer_success_stages", None)
        if stages is not None:
            data = {name: value.to(dtype=torch.bool).unsqueeze(-1) for name, value in stages.items()}
            self.env.recorder_manager.add_to_episodes("success_stages", data)

    def export(self, attrs: dict[str, Any], groups: dict[str, dict[str, np.ndarray]] | None = None) -> int:
        """Write the current episode as the next demo; returns its index."""
        import json

        recorder = self.env.recorder_manager
        recorder.record_pre_reset([0], force_export_or_skip=False)
        recorder.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=self.env.device))
        recorder.export_episodes([0])
        handler = recorder._dataset_file_handler
        index = handler.demo_count - 1
        group = handler._hdf5_data_group[f"demo_{index}"]
        for key, value in attrs.items():
            group.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        for name, arrays in (groups or {}).items():
            sub = group.create_group(name)
            for key, value in arrays.items():
                sub.create_dataset(key, data=np.asarray(value))
        handler.flush()
        return index

    def drop(self) -> None:
        """Forget the current episode (a failed attempt)."""
        from isaaclab.utils.datasets import EpisodeData

        self.env.recorder_manager._episodes[0] = EpisodeData()

    def close(self) -> None:
        handler = getattr(self.env.recorder_manager, "_dataset_file_handler", None)
        if handler is not None:
            handler.flush()
            handler.close()


def write_arrays(path: Path, group: str, arrays: dict[str, Any], attrs: dict[str, Any] | None = None) -> None:
    """Append one group of arrays (plus scalar attributes) to an HDF5 file, replacing a group of the same name."""
    import h5py

    with h5py.File(path, "a") as f:
        if group in f:
            del f[group]
        g = f.create_group(group)
        for key, value in arrays.items():
            g.create_dataset(key, data=np.asarray(value))
        for key, value in (attrs or {}).items():
            g.attrs[key] = value


def self_test(t0: float) -> None:
    """``python -m vig_dagger.isaac`` (run by ``vig-dagger check``): build the task, render the policy's cameras and
    step it; prints one ``ISAAC_CHECK {...}`` line."""
    import json
    import time

    from vig_dagger.task import POLICY_CAMERAS

    env, _ = make_env()
    obs = reset(env, 0)
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(1, 7, device=env.device))
    images = camera_images(obs, POLICY_CAMERAS, "cpu")
    result = dict(cameras={c: list(obs["rgb_camera"][f"{c}_cam"].shape) for c in POLICY_CAMERAS},
                  image_mean=float(images.float().mean()), state=list(policy_state(obs, "cpu").shape),
                  ee_frame=env.scene["ee_frame"].data.target_frame_names[0], torch=torch.__version__,
                  cuda=torch.cuda.is_available(), seconds=round(time.time() - t0, 1))
    print("ISAAC_CHECK " + json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    import time

    run(self_test, time.time())
