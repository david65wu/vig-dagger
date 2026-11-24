"""Collect one part of motion-planner demos (part 1, only with ``demos.source: collect``).

Scene seeds start at --start-seed and go up by one per attempt; successful demos go to ``<out>/part_<k>.h5/.json``
(ManiSkill trajectory format, RGB images of every camera at --cam-size, CPU physics), failed attempts are counted
but not saved. A demo that succeeds only after the env's 300-step limit is counted as failed (a policy would be
cut off there). Each episode's planner strategy is written into its JSON entry, and the per-step task state
(``TaskStateLog``) into ``traj_<n>/task_state``. --motion demo (default) moves at the strategy's sampled speeds;
an expert mode (``demos.motion``, e.g. fast_maniskill) moves like that DAgger expert (``planner.solve(expert_motion=
True)``: its fixed speeds, no refine steps, faster pre-grasp acceleration); the strategy (grasp, lift, approach) is
drawn the same way either way, and the JSON entry records the motion. ``part_<k>.done.json`` marks a finished part; an unfinished part
is collected again from the start.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vig_dagger.envs import make_env
from vig_dagger.expert import planner
from vig_dagger.jobs import exit_on_sigterm
from vig_dagger.recording import SplitRecordEpisode, TaskStateLog


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--num-demos", type=int, required=True)
    p.add_argument("--start-seed", type=int, required=True)
    p.add_argument("--strategy-seed", type=int, required=True)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--cam-size", type=int, default=256)
    p.add_argument("--max-attempts", type=int, default=None, help="default 20 x --num-demos")
    p.add_argument("--motion", default="demo", help="demo, or a DAgger expert mode to move like (demos.motion)")
    args = p.parse_args()
    exit_on_sigterm()

    name = f"part_{args.part:03d}"
    expert_motion = args.motion != "demo"
    if expert_motion:
        planner.set_expert_mode(args.motion)
    planner.seed_strategy_rng(args.strategy_seed)
    log = TaskStateLog(make_env(max_episode_steps=args.max_episode_steps, cam_size=args.cam_size))
    env = SplitRecordEpisode(log, success_dir=str(args.out), fail_dir=str(args.out / "failed"), trajectory_name=name,
                             source_type="motionplanning", source_desc="vig_dagger motion-planning expert",
                             record_reward=False)
    max_attempts = args.max_attempts or 20 * args.num_demos
    seed, successes, attempts, t0 = args.start_seed, 0, 0, time.time()
    try:
        while successes < args.num_demos and attempts < max_attempts:
            attempts += 1
            try:
                res = planner.solve(env, seed=seed, expert_motion=expert_motion)
                success = res != -1 and bool(res[-1]["success"].item())
                if success and bool(res[3][0]):  # truncated: succeeded only after the step limit
                    print(f"[seed {seed}] succeeded after the {args.max_episode_steps}-step limit; counted as failed")
                    success = False
            except Exception as exc:
                print(f"[seed {seed}] the planner raised: {exc}")
                success = False
            strategies = planner.take_used_strategies()
            env.flush_episode(success=success, save=success,
                              extra_info=({"strategy": strategies[-1]} if strategies else {}) | {"seed": seed, "motion": args.motion},
                              extra_groups={"task_state": log.arrays()})
            successes += int(success)
            print(f"[seed {seed}] {'success' if success else 'failed'} ({successes}/{args.num_demos} after {attempts} attempts)", flush=True)
            seed += 1
    finally:
        env.close()
    done = dict(part=args.part, demos=successes, attempts=attempts, first_seed=args.start_seed, last_seed=seed - 1,
                strategy_seed=args.strategy_seed, motion=args.motion, seconds=round(time.time() - t0))
    if successes < args.num_demos:
        raise SystemExit(f"only {successes} of {args.num_demos} demos after {attempts} attempts")
    (args.out / f"{name}.done.json").write_text(json.dumps(done, indent=2))
    print(f"done: {done}")


if __name__ == "__main__":
    main()
