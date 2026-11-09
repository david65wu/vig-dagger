"""Paired evaluation of one policy at one resolution level on a range of scene seeds (part 4: the retrained
policies, and the MoEs for comparison).

Every policy is evaluated on the same scenes (seeds), one episode at a time with CPU physics, so results pair up by
seed. The policy samples a 16-step chunk and executes its first --actions-per-step actions open-loop, then
replans; an episode ends at success or at --max-episode-steps. Torch is reseeded per scene.

One JSON line per episode is appended to --out (seeds already there are skipped, so a restarted job continues):
success, steps, the furthest stage reached and the stage a failure ended in (``stages.py``), failure flags, the
plug's closest and final distance to the goal, and its position in the gripper 15 steps after the first grasp.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vig_dagger import stages
from vig_dagger.envs import POLICY_CAMERAS, make_env
from vig_dagger.jobs import exit_on_sigterm
from vig_dagger.policy import LibraryPolicy, camera_images, robot_state
from vig_dagger.recording import TaskStateLog


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--factor", type=int, default=1, help="resolution level the policy acts at (MoE policies)")
    p.add_argument("--seeds", type=int, nargs=2, required=True, metavar=("FIRST", "COUNT"))
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--actions-per-step", type=int, default=8)
    p.add_argument("--cam-size", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    exit_on_sigterm()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = LibraryPolicy(args.checkpoint, device)
    if tuple(policy.cameras) != POLICY_CAMERAS:
        raise ValueError(f"policy cameras {policy.cameras} differ from the task's {POLICY_CAMERAS}")
    factor = args.factor
    if factor not in policy.factors:
        raise ValueError(f"{args.checkpoint} has no level {factor} (levels {policy.factors})")
    log = TaskStateLog(make_env(max_episode_steps=args.max_episode_steps, cam_size=args.cam_size))
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
            success, steps = False, 0
            with torch.inference_mode():
                obs, _ = log.reset(seed=seed)
                while steps < args.max_episode_steps and not success:
                    actions = policy.act(robot_state(obs, device), camera_images(obs, policy.cameras, device), factor)
                    for i in range(min(args.actions_per_step, actions.shape[1])):
                        obs, _, terminated, truncated, info = log.step(actions[:, i].to(device=log.unwrapped.device, dtype=torch.float32))
                        steps += 1
                        success = bool(info["success"][0])
                        if success or bool(terminated[0]) or bool(truncated[0]):
                            break
                    if bool(terminated[0]) or bool(truncated[0]):
                        break
            state = log.arrays()
            stage, flags = stages.label(state, success)
            dist_mm = np.asarray(state["dist"]) * 1000
            held = np.asarray(state["held"], bool)
            record = dict(seed=seed, factor=factor, success=success, steps=steps,
                          min_dist_mm=round(float(dist_mm.min()), 2), final_dist_mm=round(float(dist_mm[-1]), 2),
                          final_angle_deg=round(float(np.degrees(state["angle"][-1])), 2),
                          first_grasp=int(np.argmax(held)) if held.any() else None,
                          stage_steps={s: int((stage == i).sum()) for i, s in enumerate(stages.STAGES)},
                          seconds=round(time.time() - t0, 1), **stages.summary(stage, flags, success))
            with open(args.out, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"[seed {seed}] {'SUCCESS' if success else 'failed'} steps={steps} furthest={record['furthest_stage']} "
                  f"{record['seconds']}s", flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    main()
