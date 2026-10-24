"""Collect one part of OpenDrawer motion-planner demos (part 1); runs in the Isaac environment (``.venv-isaac``).

Scene seeds start at --start-seed and go up by one per attempt; each attempt draws a fresh planner strategy from the
strategy stream seeded with --strategy-seed. An attempt counts only if it is clean like the teleop demos: the
success rule holds (``isaac.SuccessRule``) within --max-episode-steps, with exactly two grasps (handle, then
charger) and no retries; the planner fails an attempt on a missed or slipped grasp instead of trying again.

Successful demos go to ``<out>/part_<k>.hdf5`` as ``data/demo_<i>`` (actions, the policy / pose / rgb_camera
observations before each step, sim states, success latches, and the per-step task state for the stage labels), with
the scene seed and the strategy as attributes. ``part_<k>.done.json`` marks a finished part; an unfinished part is
collected again from the start.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vig_dagger import isaac
from vig_dagger.jobs import exit_on_sigterm


def attempt(env, success_term, recorder, seed: int, max_steps: int):
    """One planner attempt from scene ``seed``: (success, reason, steps, strategy, task state)."""
    import torch

    from vig_dagger.expert import planner as P

    with torch.inference_mode():
        isaac.reset(env, seed)
        log = isaac.TaskStateLog()
        log.reset(env)
        rule = isaac.SuccessRule(env, success_term)
        params = P.pick_strategy()
        plan = P.OpenDrawerPlanner(env, params, expert=False, trace=False)
        steps, grasps, closed = 0, 0, False
        for action in plan.actions():
            env.step(action)
            steps += 1
            status = rule.update()  # first: it also updates the stage latches the records read
            recorder.step()
            log.step(env, 0)
            grasps += int(not closed and bool(action[0, 6] < 0))
            closed = bool(action[0, 6] < 0)
            if status == "success":
                if steps > max_steps:
                    return False, f"succeeded after the {max_steps}-step limit", steps, params.to_dict(), log.arrays()
                if grasps != 2:  # teleop demos: always one handle grasp and one charger grasp
                    return False, f"not clean: {grasps} grasps", steps, params.to_dict(), log.arrays()
                return True, "success", steps, params.to_dict(), log.arrays()
            if not rule.concluded and steps >= max_steps:
                return False, f"step limit ({max_steps})", steps, params.to_dict(), log.arrays()
        return False, f"planner failed: {plan.failure or plan.stage}", steps, params.to_dict(), log.arrays()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--num-demos", type=int, required=True)
    p.add_argument("--start-seed", type=int, required=True)
    p.add_argument("--strategy-seed", type=int, required=True)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--max-attempts", type=int, default=None, help="default 3 x --num-demos")
    args = p.parse_args()
    exit_on_sigterm()
    isaac.run(collect, args)


def collect(args) -> None:
    from vig_dagger.expert import planner as P

    name = f"part_{args.part:03d}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.hdf5").unlink(missing_ok=True)  # an unfinished part starts over
    env, success_term = isaac.make_env(record_dir=args.out, record_name=name)
    recorder = isaac.Recorder(env)
    P.seed_strategy_rng(args.strategy_seed)
    max_attempts = args.max_attempts or 3 * args.num_demos
    seed, successes, attempts, t0 = args.start_seed, 0, 0, time.time()
    try:
        while successes < args.num_demos and attempts < max_attempts:
            attempts += 1
            t1 = time.time()
            success, reason, steps, strategy, state = attempt(env, success_term, recorder, seed, args.max_episode_steps)
            P.take_used_strategies()
            if success:
                recorder.export({"scene_seed": seed, "strategy": strategy, "outcome": reason, "source": "planner"},
                                {"task_state": state})
                successes += 1
            else:
                recorder.drop()
            print(f"[seed {seed}] {'SUCCESS' if success else 'failed'} ({reason}) after {steps} steps, "
                  f"{time.time() - t1:.0f}s; {successes}/{args.num_demos} demos after {attempts} attempts", flush=True)
            seed += 1
    finally:
        recorder.close()
        env.close()
    done = dict(part=args.part, demos=successes, attempts=attempts, first_seed=args.start_seed, last_seed=seed - 1,
                strategy_seed=args.strategy_seed, seconds=round(time.time() - t0))
    if successes < args.num_demos:
        raise SystemExit(f"only {successes} of {args.num_demos} demos after {attempts} attempts")
    (args.out / f"{name}.done.json").write_text(json.dumps(done, indent=2))
    print(f"done: {done}", flush=True)


if __name__ == "__main__":
    main()
