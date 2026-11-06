"""Roll out a fixed acting resolution and record raw flow scores at all seven levels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vig_dagger import isaac
from vig_dagger.jobs import exit_on_sigterm
from vig_dagger.resolution_flow import episode_dir, episode_done, query_seed, score_query, write_json


def rollout(args):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.task import POLICY_CAMERAS

    cfg = json.loads(args.config.read_text())
    factors = cfg["factors"]
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    if args.group == "moe":
        policy = LibraryPolicy(cfg["checkpoints"]["moe"], device)
        policies = {f: policy for f in factors}
    else:
        policies = {f: LibraryPolicy(cfg["checkpoints"]["per_resolution"][str(f)], device) for f in factors}
    for f, policy in policies.items():
        assert tuple(policy.cameras) == POLICY_CAMERAS and f in policy.factors
        assert policy.horizon == 16 and policy.action_dim == 7
    env, success_term = isaac.make_env()
    seeds = cfg["scene_seeds"][:1] if args.pilot else cfg["scene_seeds"]
    video = None
    try:
        for seed in seeds:
            if episode_done(cfg, args.group, args.acting_factor, seed):
                continue
            started = time.time()
            dest = episode_dir(cfg, args.group, args.acting_factor, seed)
            dest.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                torch.manual_seed(seed)
                np.random.seed(seed % 2**32)
                obs = isaac.reset(env, seed)
                rule = isaac.SuccessRule(env, success_term)
                log = isaac.TaskStateLog()
                log.reset(env)
                query_steps, flows, chunks, rng_seeds, score_seconds, actions = [], [], [], [], [], []
                step, used, action_chunk, outcome = 0, 0, None, None
                dt = float(env.cfg.sim.dt * env.cfg.decimation)
                video_steps = []
                if cfg.get("record_video"):
                    from vig_dagger.resolution_flow_video import VideoWriter, camera_frame, WIDTH, CAMERA_HEIGHT
                    video = VideoWriter(dest / "cameras.mp4", WIDTH, CAMERA_HEIGHT,
                                        1 / (dt * cfg["video_stride"]))
                    video.write(camera_frame(obs))
                    video_steps.append(0)
                while outcome is None:
                    if action_chunk is None or used >= cfg["actions_per_step"]:
                        tick = time.time()
                        rng = query_seed(seed, step)
                        flow, normalized, action_chunk = score_query(
                            policies, factors, isaac.policy_state(obs, device),
                            isaac.camera_images(obs, POLICY_CAMERAS, device), args.acting_factor,
                            rng, cfg["loss_draws"])
                        query_steps.append(step)
                        flows.append(flow)
                        chunks.append(normalized)
                        rng_seeds.append(rng)
                        score_seconds.append(time.time() - tick)
                        used = 0
                        if len(query_steps) % 10 == 0:
                            write_json(dest / "progress.json", dict(step=step, queries=len(query_steps),
                                       elapsed_seconds=time.time() - started, group=args.group,
                                       acting_factor=args.acting_factor, seed=seed))
                    action = action_chunk[used].reshape(1, -1).to(device=env.device, dtype=torch.float32)
                    obs, *_ = env.step(action)
                    actions.append(action[0].cpu().numpy().copy())
                    step += 1
                    used += 1
                    status = rule.update()
                    log.step(env, 0)
                    if video is not None and step % cfg["video_stride"] == 0:
                        video.write(camera_frame(obs))
                        video_steps.append(step)
                    if status == "success":
                        outcome = "success" if step <= cfg["max_episode_steps"] else "succeeded_after_time_limit"
                    elif step >= cfg["max_episode_steps"] and not rule.concluded:
                        outcome = "time_limit"
                arrays = dict(factors=np.asarray(factors), query_steps=np.asarray(query_steps),
                              query_seconds=np.asarray(query_steps) * dt, flow=np.asarray(flows),
                              normalized_chunks=np.asarray(chunks), rng_seeds=np.asarray(rng_seeds),
                              score_seconds=np.asarray(score_seconds), executed_actions=np.asarray(actions))
                if video is not None:
                    video.finish()
                    video = None
                    arrays["video_steps"] = np.asarray(video_steps)
                arrays.update({f"state_{k}": v for k, v in log.arrays().items()})
                tmp = dest / "trace.npz.partial"
                with tmp.open("wb") as f:
                    np.savez_compressed(f, **arrays)
                tmp.replace(dest / "trace.npz")
                row = dict(protocol_id=cfg["protocol_id"], group=args.group, acting_factor=args.acting_factor,
                           seed=seed, success=outcome == "success", outcome=outcome, steps=step, dt=dt,
                           queries=len(query_steps), simulated_seconds=step * dt,
                           wall_seconds=time.time() - started, mean_query_wall_seconds=float(np.mean(score_seconds)),
                           loss_draws=cfg["loss_draws"], calibration=False, finished_at=time.time())
                if cfg.get("record_video"):
                    row.update(video_fps=1 / (dt * cfg["video_stride"]), video_frames=len(video_steps),
                               video_cameras=["left", "wrist"])
                write_json(dest / "episode.json", row)
                print("EPISODE " + json.dumps(row), flush=True)
    finally:
        if video is not None:
            video.abort()
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--group", choices=("moe", "per_resolution"), required=True)
    p.add_argument("--acting-factor", type=int, required=True)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    exit_on_sigterm()
    isaac.run(rollout, args)


if __name__ == "__main__":
    main()
