"""DAgger collection worker for one round and scorer (part 3).

``python -m vig_dagger.workers.dagger --config <work_dir>/config.json --round <r> --scorer <s> --worker <w> --workers <W>``
reads the collection's ``settings.json`` (driving policies, thresholds, counts, expert mode; written by
``steps.py dagger-prepare``).

Every episode: the round's full-resolution policy drives (16-step chunks, the first 8 actions executed open-loop), and the
query score (VIG or flow loss) is computed on every observation. The gate (``gate.py``) hands control to the expert
after ``take_count`` consecutive scores above the take threshold; the expert (``expert/planner.py``,
``resume_and_solve``) re-plans the rest of the task from the live state, and the score keeps being computed after
every expert step; control returns to the policy after ``return_count`` consecutive scores below the return
threshold, for the rest of the same episode. Without a return the expert finishes the task.

An episode ends at success (the policy's step that achieves it, or the end of the expert's plan), at the step limit
(``dagger.max_episode_steps``, for policy and expert alike), or when the expert's planner fails. Successful episodes are
DAgger demos: their trajectories (policy and expert steps, with the executed actions as labels) go to
``trajectories/worker_<w>_seg<k>.h5``. Every attempt, failed ones included, gets its per-step record in
``attempts/worker_<w>.h5`` (group ``seed_<seed>``: controller per step, score per observation, VIG's gap per level,
task state) and one summary line in ``episodes/worker_<w>.jsonl``.

Workers share the scene-seed sequence: worker w of W takes seeds seed_start + w, + W, + 2W, ... Before each
episode a worker counts the successes of all workers; it stops once they reach ``dagger.num_demos`` (episodes still
running then finish and are kept; the dataset builder uses the first ``dagger.num_demos`` to finish). A restarted worker skips the
seeds it already finished. Each episode reseeds torch and the expert's strategy stream from its scene seed, so a
scene's rollout does not depend on which worker ran it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vig_dagger import config as config_module
from vig_dagger.config import Paths
from vig_dagger.envs import make_env
from vig_dagger.expert import planner
from vig_dagger.gate import TwoThresholdGate
from vig_dagger.policy import PolicySet, camera_images, check_cameras, robot_state
from vig_dagger.recording import EXPERT, POLICY, SplitRecordEpisode, TaskStateLog, write_arrays
from vig_dagger.scoring import build_scorer


def count_successes(out: Path) -> tuple[int, int]:
    """(successes, attempts) over every worker's episode log."""
    successes = attempts = 0
    for path in (out / "episodes").glob("worker_*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                attempts += 1
                successes += bool(json.loads(line)["success"])
    return successes, attempts


class Episode:
    """One DAgger episode's rollout state."""

    def __init__(self, env, log: TaskStateLog, policies: PolicySet, scorer, gate: TwoThresholdGate, args) -> None:
        self.env, self.log, self.policies, self.scorer, self.gate, self.args = env, log, policies, scorer, gate, args
        self.scores: dict[int, float] = {}
        self.gaps: dict[int, list[float]] = {}
        self.seconds: list[float] = []
        self.events: list[tuple[int, str]] = []
        self.pending = None  # the score of the current observation, when the gate has already seen it
        self.hit_limit = False

    def score(self, obs) -> object:
        device = self.policies.device
        result = self.scorer(self.policies, robot_state(obs, device), camera_images(obs, self.policies.cameras, device))
        t = self.log.num_steps
        self.scores[t] = result.value
        if "w1_by_factor" in result.details:
            self.gaps[t] = [result.details["w1_by_factor"][f] for f in self.policies.factors[1:]]
        self.seconds.append(result.details["seconds"])
        return result

    def on_expert_step(self, obs, _reward, _terminated, truncated, _info, _action) -> bool:
        """Called after every expert step inside a planned motion; True stops the expert."""
        if bool(torch.as_tensor(truncated).reshape(-1)[0]):
            self.hit_limit = True
            return True
        result = self.score(obs)
        if self.gate.update(result.value) == "return":
            self.events.append((self.log.num_steps, "return"))
            self.pending = result
            return True
        return False

    def run(self, seed: int) -> tuple[bool, str]:
        # Reset, policy and expert steps all run in inference mode: ManiSkill's controllers keep tensors that a
        # later step outside it could not update in place.
        with torch.inference_mode():
            return self._run(seed)

    def _run(self, seed: int) -> tuple[bool, str]:
        args = self.args
        obs, _ = self.env.reset(seed=seed)
        self.gate.reset()
        self.log.controller = POLICY
        chunk, used = None, 0
        while True:
            if self.pending is not None:  # scored by the expert's last step, which returned control
                result, switch, self.pending = self.pending, None, None
            else:
                result = self.score(obs)
                switch = self.gate.update(result.value)
            if switch == "take":
                self.events.append((self.log.num_steps, "take"))
                self.log.controller = EXPERT
                try:
                    plan = planner.resume_and_solve(self.env, on_step=self.on_expert_step)
                except Exception as exc:
                    print(f"[seed {seed}] the expert's planner raised: {exc}", flush=True)
                    return False, "expert_error"
                finally:
                    self.log.controller = POLICY
                if self.hit_limit or self.log.num_steps > args.max_episode_steps:
                    return False, "time_limit_expert"
                if plan is None:  # control came back to the policy mid-plan
                    obs = self.env.unwrapped.get_obs()
                    chunk = None
                    continue
                if plan == -1:
                    return False, "expert_error"
                success = bool(plan[-1]["success"].item())
                return success, "success_expert" if success else "expert_finished_failed"
            if chunk is None or used >= args.chunk_steps:
                chunk, used = self.policies.denormalize_actions(result.chunks[0:1]), 0
            action = chunk[:, used].to(device=self.env.unwrapped.device, dtype=torch.float32)
            used += 1
            obs, _, terminated, truncated, info = self.env.step(action)
            if bool(info["success"][0]):
                return True, "success_policy"
            if bool(truncated[0]):
                return False, "time_limit_policy"
            if bool(terminated[0]):
                return False, "terminated"

    def arrays(self) -> dict[str, np.ndarray]:
        n_obs = self.log.num_steps + 1
        score = np.full(n_obs, np.nan, np.float32)
        for t, v in self.scores.items():
            if t < n_obs:
                score[t] = v
        out = {"score": score, "controller": np.asarray(self.log.arrays()["controller"], np.int8),
               "take_steps": np.array([t for t, e in self.events if e == "take"], np.int64),
               "return_steps": np.array([t for t, e in self.events if e == "return"], np.int64),
               "score_seconds": np.asarray(self.seconds, np.float32)}
        if self.gaps:
            gaps = np.full((n_obs, len(self.policies.factors) - 1), np.nan, np.float32)
            for t, v in self.gaps.items():
                if t < n_obs:
                    gaps[t] = v
            out["vig_gap"] = gaps
            out["vig_factors"] = np.asarray(self.policies.factors[1:], np.int64)
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="the experiment config (<work_dir>/config.json)")
    p.add_argument("--round", required=True)
    p.add_argument("--scorer", choices=["vig", "flow"], required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    cli = p.parse_args()

    cfg = config_module.load(cli.config)
    out = Paths.of(cfg).dagger(cli.round, cli.scorer)
    s = json.loads((out / "settings.json").read_text())  # written and checked by ``steps.py dagger-prepare``
    args = argparse.Namespace(max_episode_steps=int(s["max_episode_steps"]), chunk_steps=int(s["chunk_steps"]))
    planner.set_expert_mode(s["expert"])
    planner.EXPERT_ATOMIC_GRASP = bool(s["atomic_grasp"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = s["policies"]
    policies = PolicySet.load(spec if isinstance(spec, str) else {int(k): v for k, v in spec.items()}, device)
    check_cameras(policies)
    scorer = build_scorer(cli.scorer, s["scorer_settings"])
    gate = TwoThresholdGate(s["take_threshold"], s["return_threshold"], s["take_count"], s["return_count"])
    print(f"[worker {cli.worker}] round={cli.round} scorer={cli.scorer} take>{gate.take_threshold:.6g} x{gate.take_count} "
          f"return<{gate.return_threshold:.6g} x{gate.return_count} expert={s['expert']} levels={policies.factors}", flush=True)

    name = f"worker_{cli.worker:02d}"
    for sub in ("episodes", "attempts", "trajectories"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes" / f"{name}.jsonl"
    done_seeds = {json.loads(line)["seed"] for line in episodes_path.read_text().splitlines() if line.strip()} \
        if episodes_path.exists() else set()
    segment = len(list((out / "trajectories").glob(f"{name}_seg*.h5")))
    traj_name = f"{name}_seg{segment}"
    log = TaskStateLog(make_env(max_episode_steps=args.max_episode_steps))
    env = SplitRecordEpisode(log, success_dir=str(out / "trajectories"), fail_dir=str(out / "trajectories_failed"),
                             trajectory_name=traj_name, source_type="dagger", record_reward=False,
                             source_desc=f"DAgger ({cli.round} round, {cli.scorer} scorer): policy and motion-planning expert steps")
    seed = int(s["seed_start"]) + cli.worker
    try:
        while True:
            successes, attempts = count_successes(out)
            if successes >= s["target"] or attempts >= s["max_attempts"]:
                print(f"[worker {cli.worker}] stopping: {successes} successes, {attempts} attempts over all workers")
                break
            if seed in done_seeds:
                seed += cli.workers
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed % 2**32)
            planner.seed_strategy_rng((int(s["strategy_seed"]) * 1_000_003 + seed) % 2**31)
            episode = Episode(env, log, policies, scorer, gate, args)
            success, reason = episode.run(seed)
            strategies = planner.take_used_strategies()
            arrays = episode.arrays()
            state = log.arrays()
            traj = env.flush_episode(success=success, save=success or bool(s["save_failed"]),
                                     extra_info={"seed": seed, "reason": reason, "expert_strategies": strategies},
                                     extra_groups={"dagger": arrays, "task_state": state})
            controller = arrays["controller"]
            summary = dict(
                seed=seed, success=success, reason=reason, steps=int(len(controller)),
                expert_steps=int((controller == EXPERT).sum()), takeovers=int(len(arrays["take_steps"])),
                returns=int(len(arrays["return_steps"])),
                first_take_step=int(arrays["take_steps"][0]) if len(arrays["take_steps"]) else None,
                file=f"trajectories/{traj_name}.h5" if success else None, traj=traj if success else None,
                worker=cli.worker, finished_at=time.time(), seconds=round(time.time() - t0, 1),
                score_seconds=float(np.mean(arrays["score_seconds"])) if len(arrays["score_seconds"]) else None)
            write_arrays(out / "attempts" / f"{name}.h5", f"seed_{seed}", {**arrays, **state},
                         {k: v for k, v in summary.items() if v is not None and not isinstance(v, bool)} | {"success": success})
            with open(episodes_path, "a") as fh:
                fh.write(json.dumps(summary) + "\n")
            print(f"[seed {seed}] {'SUCCESS' if success else 'failed'} ({reason}) steps={summary['steps']} "
                  f"expert={summary['expert_steps']} takeovers={summary['takeovers']} returns={summary['returns']} "
                  f"{summary['seconds']}s", flush=True)
            seed += cli.workers
    finally:
        env.close()


if __name__ == "__main__":
    main()
