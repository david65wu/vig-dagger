"""DAgger collection worker for one round and scorer (part 3); runs in the Isaac environment (``.venv-isaac``).

``python -m vig_dagger.workers.dagger --config <work_dir>/config.json --round <r> --scorer <s> --worker <w> --workers <W>``
reads the collection's ``settings.json`` (driving policies, thresholds, counts, expert mode; written by
``steps.py dagger-prepare``).

Every episode: the round's full-resolution policy drives (16-step chunks, the first 8 actions executed open-loop), and
the query score (VIG or flow loss) is computed on every observation, policy- and expert-driven alike. The gate
(``gate.py``) hands control to the expert after ``take_count`` consecutive scores above the take threshold: a new
expert plan (``expert/planner.py``, a fresh strategy, expert mode) looks at the live scene and plans only what is left,
correcting in place when the hand is already near its next target. Control returns to the policy after
``return_count`` consecutive scores below the return threshold, on the very next step, whatever the expert was doing
(its plan is dropped; a later takeover plans afresh). Without a return the expert finishes the task.

An episode succeeds by the teleop recorder's rule (``isaac.SuccessRule``) within ``dagger.max_episode_steps`` (policy
and expert steps alike). It fails at the step limit, or when the expert's plan fails (no retries: a missed or
slipped grasp, a stage that does not complete, the charger out of reach). Successful episodes are DAgger demos: they
go to ``trajectories/worker_<w>_seg<k>.hdf5`` as ``data/demo_<i>`` in the planner demos' format (policy and expert
steps, the executed actions as labels), with the episode's DAgger record and task state as extra groups. Every
attempt, failed ones included, gets its per-step record in ``attempts/worker_<w>.h5`` (group ``seed_<seed>``:
controller per step, score per observation, VIG's gap per level, task state) and one summary line in
``episodes/worker_<w>.jsonl``.

Workers share the scene-seed sequence: worker w of W takes seeds seed_start + w, + W, + 2W, ... Before each
episode a worker counts the successes of all workers; it stops once they reach ``dagger.num_demos`` (episodes still
running then finish and are kept; the dataset builder uses the first ``dagger.num_demos`` to finish). A restarted
worker skips the seeds it already finished and writes a new trajectory segment. Each episode reseeds torch and the
expert's strategy stream from its scene seed, so a scene's rollout does not depend on which worker ran it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vig_dagger import isaac
from vig_dagger.jobs import exit_on_sigterm


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
    """One DAgger episode's rollout and its per-step record."""

    def __init__(self, env, success_term, recorder, policies, scorer, gate, max_steps: int, chunk_steps: int) -> None:
        self.env, self.success_term, self.recorder = env, success_term, recorder
        self.policies, self.scorer, self.gate = policies, scorer, gate
        self.max_steps, self.chunk_steps = max_steps, chunk_steps
        self.log = isaac.TaskStateLog()
        self.scores: list[float] = []
        self.gaps: list[list[float]] = []
        self.seconds: list[float] = []
        self.events: list[tuple[int, str]] = []
        self.strategies: list[dict] = []
        self.expert_failure: str | None = None

    def score(self, obs):
        device = self.policies.device
        result = self.scorer(self.policies, isaac.policy_state(obs, device),
                             isaac.camera_images(obs, self.policies.cameras, device))
        self.scores.append(result.value)
        if "w1_by_factor" in result.details:
            self.gaps.append([result.details["w1_by_factor"][f] for f in self.policies.factors[1:]])
        self.seconds.append(result.details["seconds"])
        return result

    def run(self, seed: int) -> tuple[bool, str]:
        import torch

        from vig_dagger.expert import planner as P
        from vig_dagger.task import EXPERT, POLICY

        env = self.env
        with torch.inference_mode():
            obs = isaac.reset(env, seed)
            self.log.reset(env)
            rule = isaac.SuccessRule(env, self.success_term)
            self.gate.reset()
            plan = expert = chunk = None
            used, steps = 0, 0
            while True:
                result = self.score(obs)
                switch = self.gate.update(result.value)
                if switch == "take":
                    self.events.append((steps, "take"))
                    params = P.pick_strategy()
                    self.strategies.append(params.to_dict())
                    plan = P.OpenDrawerPlanner(env, params, expert=True, trace=False)
                    expert = plan.actions()
                elif switch == "return":
                    self.events.append((steps, "return"))
                    plan = expert = chunk = None
                if expert is not None:
                    try:
                        action = next(expert)
                    except StopIteration:
                        self.expert_failure = plan.failure or plan.stage
                        return False, "expert_failed"
                    controller = EXPERT
                else:
                    if chunk is None or used >= self.chunk_steps:
                        chunk, used = self.policies.denormalize_actions(result.chunks[0:1]), 0
                    action = chunk[:, used]
                    used += 1
                    controller = POLICY
                obs, *_ = env.step(action.reshape(1, -1).to(device=env.device, dtype=torch.float32))
                steps += 1
                status = rule.update()  # first: it also updates the stage latches the records read
                self.recorder.step()
                self.log.step(env, controller)
                who = "expert" if controller == EXPERT else "policy"
                if status == "success":
                    return (True, f"success_{who}") if steps <= self.max_steps else (False, f"time_limit_{who}")
                if not rule.concluded and steps >= self.max_steps:
                    return False, f"time_limit_{who}"

    def arrays(self):
        import numpy as np

        out = {"score": np.asarray(self.scores, np.float32), "controller": self.log.arrays()["controller"],
               "take_steps": np.array([t for t, e in self.events if e == "take"], np.int64),
               "return_steps": np.array([t for t, e in self.events if e == "return"], np.int64),
               "score_seconds": np.asarray(self.seconds, np.float32)}
        if self.gaps:
            out["vig_gap"] = np.asarray(self.gaps, np.float32)
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
    exit_on_sigterm()
    isaac.run(collect, cli)


def collect(cli) -> None:
    import numpy as np
    import torch

    from vig_dagger import config as config_module
    from vig_dagger.config import Paths
    from vig_dagger.expert import planner as P
    from vig_dagger.gate import TwoThresholdGate
    from vig_dagger.policy import PolicySet, check_cameras
    from vig_dagger.scoring import build_scorer
    from vig_dagger.task import EXPERT

    cfg = config_module.load(cli.config)
    out = Paths.of(cfg).dagger(cli.round, cli.scorer)
    s = json.loads((out / "settings.json").read_text())  # written and checked by ``steps.py dagger-prepare``
    P.set_expert_mode(s["expert"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = s["policies"]
    policies = PolicySet.load(spec if isinstance(spec, str) else {int(k): v for k, v in spec.items()}, device)
    check_cameras(policies)
    scorer = build_scorer(cli.scorer, s["scorer_settings"])
    gate = TwoThresholdGate(s["take_threshold"], s["return_threshold"], s["take_count"], s["return_count"])
    print(f"[worker {cli.worker}] round={cli.round} scorer={cli.scorer} take>{gate.take_threshold:.6g} x{gate.take_count} "
          f"return<{gate.return_threshold:.6g} x{gate.return_count} levels={policies.factors}; "
          f"expert: {P.describe_expert_overrides()}", flush=True)

    name = f"worker_{cli.worker:02d}"
    for sub in ("episodes", "attempts", "trajectories"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes" / f"{name}.jsonl"
    done_seeds = {json.loads(line)["seed"] for line in episodes_path.read_text().splitlines() if line.strip()} \
        if episodes_path.exists() else set()
    segment = len(list((out / "trajectories").glob(f"{name}_seg*.hdf5")))  # a restart never reopens a written file
    traj_name = f"{name}_seg{segment}"
    env, success_term = isaac.make_env(record_dir=out / "trajectories", record_name=traj_name)
    recorder = isaac.Recorder(env)
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
            P.seed_strategy_rng((int(s["strategy_seed"]) * 1_000_003 + seed) % 2**31)
            episode = Episode(env, success_term, recorder, policies, scorer, gate, int(s["max_episode_steps"]),
                              int(s["chunk_steps"]))
            success, reason = episode.run(seed)
            P.take_used_strategies()
            arrays = episode.arrays()
            state = episode.log.arrays()
            controller = arrays["controller"]
            summary = dict(
                seed=seed, success=success, reason=reason, expert_failure=episode.expert_failure,
                steps=int(len(controller)), expert_steps=int((controller == EXPERT).sum()),
                takeovers=int(len(arrays["take_steps"])), returns=int(len(arrays["return_steps"])),
                first_take_step=int(arrays["take_steps"][0]) if len(arrays["take_steps"]) else None,
                file=None, traj=None, worker=cli.worker, seconds=round(time.time() - t0, 1),
                score_seconds=float(np.mean(arrays["score_seconds"])) if len(arrays["score_seconds"]) else None)
            if success:
                summary["traj"] = recorder.export(
                    {"scene_seed": seed, "outcome": reason, "source": f"dagger_{cli.round}_{cli.scorer}",
                     "expert_strategies": episode.strategies}, {"dagger": arrays, "task_state": state})
                summary["file"] = f"trajectories/{traj_name}.hdf5"
            else:
                recorder.drop()
            summary["finished_at"] = time.time()
            isaac.write_arrays(out / "attempts" / f"{name}.h5", f"seed_{seed}", {**arrays, **state},
                               {k: v for k, v in summary.items() if v is not None and not isinstance(v, bool)}
                               | {"success": success})
            with open(episodes_path, "a") as fh:
                fh.write(json.dumps(summary) + "\n")
            print(f"[seed {seed}] {'SUCCESS' if success else 'failed'} ({reason}"
                  f"{': ' + episode.expert_failure if episode.expert_failure else ''}) steps={summary['steps']} "
                  f"expert={summary['expert_steps']} takeovers={summary['takeovers']} returns={summary['returns']} "
                  f"{summary['seconds']}s", flush=True)
            seed += cli.workers
    finally:
        recorder.close()
        env.close()


if __name__ == "__main__":
    main()
