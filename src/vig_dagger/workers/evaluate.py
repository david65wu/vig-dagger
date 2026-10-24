"""Paired evaluation of one policy at one resolution level on a range of scene seeds (part 4); runs in the Isaac
environment (``.venv-isaac``).

Every policy is evaluated on the same scenes (seeds), one episode at a time with CPU physics, so results pair up by
seed. The policy samples a 16-step chunk and executes its first --actions-per-step actions, then replans. An episode
succeeds by the teleop recorder's rule (``isaac.SuccessRule``) within --max-episode-steps; torch is reseeded per
scene.

One JSON line per episode is appended to --out (seeds already there are skipped, so a restarted job continues):
success, steps, the success latches reached, the furthest task stage and the stage a failure ended in
(``stages.py``), and the failure flags.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vig_dagger import isaac
from vig_dagger.jobs import exit_on_sigterm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--factor", type=int, default=1, help="resolution level the policy acts at (MoE policies)")
    p.add_argument("--seeds", type=int, nargs=2, required=True, metavar=("FIRST", "COUNT"))
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--actions-per-step", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    exit_on_sigterm()
    isaac.run(evaluate, args)


def evaluate(args) -> None:
    import numpy as np
    import torch

    from vig_dagger import stages
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.task import POLICY_CAMERAS

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = LibraryPolicy(args.checkpoint, device)
    if tuple(policy.cameras) != POLICY_CAMERAS:
        raise ValueError(f"policy cameras {policy.cameras} differ from the task's {POLICY_CAMERAS}")
    if args.factor not in policy.factors:
        raise ValueError(f"{args.checkpoint} has no level {args.factor} (levels {policy.factors})")
    env, success_term = isaac.make_env()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["seed"] for line in args.out.read_text().splitlines() if line.strip()} if args.out.exists() else set()
    first, count = args.seeds
    try:
        for seed in range(first, first + count):
            if seed in done:
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed % 2**32)
            with torch.inference_mode():
                obs = isaac.reset(env, seed)
                log = isaac.TaskStateLog()
                log.reset(env)
                rule = isaac.SuccessRule(env, success_term)
                chunk, used, steps, outcome = None, 0, 0, None
                while outcome is None:
                    if chunk is None or used >= args.actions_per_step:
                        chunk = policy.act(isaac.policy_state(obs, device), isaac.camera_images(obs, policy.cameras, device),
                                           args.factor)[0]
                        used = 0
                    obs, *_ = env.step(chunk[used].reshape(1, -1).to(device=env.device, dtype=torch.float32))
                    used += 1
                    steps += 1
                    status = rule.update()  # first: it also updates the stage latches the log reads
                    log.step(env, 0)
                    if status == "success":
                        outcome = "success" if steps <= args.max_episode_steps else "succeeded after the step limit"
                    elif not rule.concluded and steps >= args.max_episode_steps:
                        outcome = f"step limit ({args.max_episode_steps})"
            success = outcome == "success"
            state = log.arrays()
            stage, flags = stages.label(state, success)
            reached = state["stages"].any(axis=0)
            record = dict(seed=seed, factor=args.factor, success=success, outcome=outcome, steps=steps,
                          latches={k: bool(v) for k, v in zip(stages.LATCHES, reached)}, max_latch=int(reached.sum()),
                          stage_steps={s: int((stage == i).sum()) for i, s in enumerate(stages.STAGES)},
                          seconds=round(time.time() - t0, 1), **stages.summary(stage, flags, success))
            with open(args.out, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"[seed {seed}] {'SUCCESS' if success else 'failed'} ({outcome}) steps={steps} "
                  f"furthest={record['furthest_stage']} {record['seconds']}s", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
