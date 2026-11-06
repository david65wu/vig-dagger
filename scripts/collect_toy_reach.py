"""Collect successful narrow-tube planner demos, training data, and review MP4s."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import h5py
import numpy as np


CAMERAS = ("third_person", "wrist")
FPS = 60


def write_video(path, frames, fps=FPS):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-threads", "2", "-preset", "veryfast", "-crf", "18", "-g", "10", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE,
    )
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed for {path}")


def write_tree(group, values):
    for key, value in values.items():
        if isinstance(value, dict):
            write_tree(group.create_group(key), value)
        else:
            group.create_dataset(key, data=value.cpu().numpy())


def collect(args):
    import torch
    from PIL import Image, ImageDraw

    from vig_dagger.toy_reach_env.config import ToyReachEnvCfg
    from vig_dagger.toy_reach_env.environment import ToyReachEnv
    from vig_dagger.toy_reach_env.planner import TubePlan

    sphere = args.task in ('sphere', 'fixed_goal_sphere')
    if sphere:
        from vig_dagger.toy_reach_env.sphere_config import SphereReachEnvCfg
        from vig_dagger.toy_reach_env.sphere_environment import SphereReachEnv
        cfg = SphereReachEnvCfg()
        if args.task == 'fixed_goal_sphere':cfg.endpoint_mode = 'fixed_goal'
        env = SphereReachEnv(cfg)
    else:
        cfg = ToyReachEnvCfg()
        env = ToyReachEnv(cfg)
    task_root = args.out / "dataset" / "toy_reach"
    task_root.mkdir(parents=True)
    (args.out / "mp4s").mkdir()
    settings = dict(
        num_demos=args.num_demos, first_seed=args.seed, point_a=cfg.point_a, point_b=cfg.point_b,
        tube_radius_m=cfg.tube_radius, planner_radius_m=0.008, lookahead_m=0.012,
        max_translation_command_m=0.008, orientation_wxyz=cfg.eef_quaternion,
        action_scale=1.0, fps=FPS, cameras=CAMERAS, image_shape=[256, 256, 3],
        planner="cubic_bezier_cartesian_with_differential_ik_tracking",
        observation_alignment="obs[t] and RGB frame t precede actions[t]; next_state[t] follows actions[t]",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        task_variant=args.task, state_dim=12 if sphere else 9,
    )
    if sphere:
        settings.update(point_a=None, point_b=None, sphere_center=cfg.sphere_center,
                        sphere_radius_m=cfg.sphere_radius, min_endpoint_separation_m=cfg.min_endpoint_separation,
                        state_order='TCP xyz, quaternion wxyz, two finger positions, goal xyz',
                        endpoints='Sampled per demo; saved in each demo record and HDF5 attributes')
        if args.task == 'fixed_goal_sphere':
            settings.update(point_b=cfg.point_b, fixed_goal_direction=cfg.fixed_goal_direction,
                endpoints='B fixed on sphere; A uniform on opposite hemisphere: dot(A-C, B-C) <= 0')
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    summaries, shards = [], []
    try:
        for index in range(args.num_demos):
            seed = args.seed + index
            key = f"demo_{index:03d}"
            if sphere:
                obs, _ = env.reset(seed=seed)
                plan = env.sample_plan(seed+100000)
            else:
                plan = TubePlan.sample(cfg.point_a, cfg.point_b, settings["planner_radius_m"], seed)
                obs, _ = env.reset(seed=seed, options={"eef_position": plan.control_points[0]})
            initial_state = env.scene.get_state(is_relative=True)
            states, next_states, actions, joints, velocities, targets = [], [], [], [], [], []
            images = {c: [] for c in CAMERAS}
            stable = 0
            for step in range(1000):
                state = obs["policy"][0].cpu().numpy().copy()
                target = plan.target(state[:3])
                action = env.action_toward(target, max_translation=settings["max_translation_command_m"])
                # Copy before stepping: sensor/robot tensors are reused by Isaac Lab.
                states.append(state)
                actions.append(action[0].cpu().numpy().copy())
                targets.append(target)
                joints.append(env.scene["robot"].data.joint_pos[0].cpu().numpy().copy())
                velocities.append(env.scene["robot"].data.joint_vel[0].cpu().numpy().copy())
                for camera in CAMERAS:
                    rgb = obs["rgb_camera"][f"{camera}_cam"][0].cpu().numpy().copy()
                    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8 or rgb.std() < 2:
                        raise RuntimeError(f"Invalid camera frame: {key}/{camera}/{step}")
                    images[camera].append(rgb)
                obs, _ = env.step(action)
                next_states.append(obs["policy"][0].cpu().numpy().copy())
                error = float(np.linalg.norm(next_states[-1][:3] - np.asarray(cfg.point_b)))
                stable = stable + 1 if error < 0.0015 else 0
                if stable >= 15:
                    break
            else:
                raise RuntimeError(f"{key} did not reach B in 1000 steps; error={error}")

            state = np.asarray(states, dtype=np.float32)
            next_state = np.asarray(next_states, dtype=np.float32)
            action_array = np.asarray(actions, dtype=np.float32)
            positions = np.concatenate((state[:, :3], next_state[-1:, :3]))
            a, b = np.asarray(cfg.point_a), np.asarray(cfg.point_b)
            progress = np.clip((positions - a) @ (b - a) / np.dot(b - a, b - a), 0, 1)
            distance = np.linalg.norm(positions - a - progress[:, None] * (b - a), axis=1)
            rotation = 2 * np.arccos(np.clip(np.abs(state[:, 3:7] @ np.asarray(cfg.eef_quaternion)), 0, 1))
            if distance.max() > cfg.tube_radius:
                raise RuntimeError(f"{key} left the {cfg.tube_radius} m tube: {distance.max()} m")
            if not all(np.isfinite(x).all() for x in (state, next_state, action_array)):
                raise RuntimeError(f"{key} contains non-finite values")
            if not np.array_equal(state[1:], next_state[:-1]):
                raise RuntimeError(f"{key} has misaligned transition states")
            if rotation.max() > 0.02:
                raise RuntimeError(f"{key} did not hold its orientation")
            if sphere and not (np.allclose(state[:, 9:12], b, atol=1e-7)
                               and np.allclose(next_state[:, 9:12], b, atol=1e-7)):
                raise RuntimeError(f'{key} goal conditioning is misaligned')

            length = len(actions)
            shard_name = f"shard_{index:05d}"
            hdf = Path("shards") / f"{shard_name}.hdf5"
            (task_root / hdf).parent.mkdir(exist_ok=True)
            with h5py.File(task_root / hdf, "x") as f:
                g = f.create_group(f"data/{key}")
                g.attrs.update(length=length, seed=seed, success=True, fps=FPS, quaternion_order="wxyz")
                g.attrs.update(point_a=a, point_b=b, task_variant=args.task)
                g.create_dataset("actions", data=action_array)
                for name, value in {
                    "obs/state": state, "obs/robot0_eef_pos": state[:, :3],
                    "obs/robot0_eef_quat": state[:, 3:7], "obs/robot0_gripper_qpos": state[:, 7:9],
                    "obs/joint_pos": joints, "obs/joint_vel": velocities,
                    "next_state": next_state, "timestamps": np.arange(length) / FPS,
                    "planner/targets": targets, "planner/control_points": plan.control_points,
                    "final_obs/state": next_state[-1],
                }.items():
                    g.create_dataset(name, data=value)
                if sphere:
                    g.create_dataset('obs/goal_xyz', data=state[:, 9:12])
                write_tree(g.create_group("initial_state"), initial_state)
                for camera in CAMERAS:
                    g.create_dataset(f"obs/rgb_camera/{camera}_cam", data=np.asarray(images[camera]),
                                     compression="gzip", compression_opts=1, chunks=(1, 256, 256, 3))
                    g.create_dataset(f"final_obs/rgb_camera/{camera}_cam",
                                     data=obs["rgb_camera"][f"{camera}_cam"][0].cpu().numpy())

            videos = {}
            for camera in CAMERAS:
                video = Path("videos") / camera / f"{shard_name}.mp4"
                write_video(task_root / video, images[camera])
                sidecar = dict(video_backend="ffmpeg", fps=FPS, height=256, width=256, frames=length,
                               cam=camera, shard=shard_name,
                               segments=[dict(demo_key=key, frame_start=0, frame_count=length)])
                (task_root / video.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2) + "\n")
                videos[camera] = dict(mp4=str(video), sidecar=str(video.with_suffix(".json")))

            # Labeled review video: both views at 30 fps, preserving real-time speed.
            review_frames = []
            for t in range(0, length, 2):
                panel = Image.new("RGB", (512, 284))
                for column, camera in enumerate(CAMERAS):
                    panel.paste(Image.fromarray(images[camera][t]), (256 * column, 28))
                ImageDraw.Draw(panel).text((6, 7), f"Demo {index + 1:02d}/{args.num_demos} | third person (left), wrist (right)", fill="white")
                review_frames.append(np.asarray(panel))
            # One terminal frame makes the last observation visible in the review.
            panel = Image.new("RGB", (512, 284))
            for column, camera in enumerate(CAMERAS):
                panel.paste(Image.fromarray(obs["rgb_camera"][f"{camera}_cam"][0].cpu().numpy()), (256 * column, 28))
            ImageDraw.Draw(panel).text((6, 7), f"Demo {index + 1:02d}/{args.num_demos} | reached B ({error * 1000:.2f} mm)", fill="white")
            review_frames.extend([np.asarray(panel)] * 15)
            review_path = args.out / "mp4s" / f"{key}.mp4"
            write_video(review_path, review_frames, fps=30)

            summary = dict(demo=key, display_number=index + 1, seed=seed, steps=length, seconds=length / FPS,
                           success=True, goal_error_m=error, max_line_distance_m=float(distance.max()),
                           max_rotation_error_rad=float(rotation.max()), start_position=state[0, :3].tolist(),
                           control_points=plan.control_points.tolist(), mp4=str(review_path.relative_to(args.out)))
            summary.update(point_a=a.tolist(), point_b=b.tolist())
            summaries.append(summary)
            shards.append(dict(shard_id=index, hdf5=str(hdf), videos=videos,
                               demos=[dict(demo_key=key, length=length, task="Sphere Reach" if sphere else "Toy Reach", source_demo_key=key)]))
            (args.out / "progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print("DEMO_DONE " + json.dumps(summary), flush=True)

        manifest = dict(version=1, task="Sphere Reach" if sphere else "Toy Reach", task_slug="toy_reach",
                        env_id="SphereReachEnv" if sphere else "ToyReachEnv", task_variant=args.task,
                        control_mode="relative_ik", timestamps_hz=FPS, video_fps=FPS,
                        cameras=list(CAMERAS), demos_per_shard=1, shards=shards,
                        created_at_utc=settings["created_at_utc"])
        (task_root / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.out / "dataset" / "dataset_manifest.json").write_text(json.dumps(dict(
            format="robopolicy_small_files", format_version=1, num_tasks=1, num_demos=len(shards),
            tasks=[dict(task="Sphere Reach" if sphere else "Toy Reach", task_slug="toy_reach", manifest="toy_reach/demo_manifest.json")],
            created_at_utc=settings["created_at_utc"],
        ), indent=2) + "\n")
        (args.out / "summary.json").write_text(json.dumps(dict(settings=settings, demos=summaries), indent=2) + "\n")
        print(f"COLLECTION_DONE demos={len(summaries)} output={args.out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument('--task', choices=['line', 'sphere', 'fixed_goal_sphere'], default='line')
    args = parser.parse_args()
    if args.num_demos < 1:
        parser.error("num-demos must be positive")
    if args.out.exists():
        parser.error("use a fresh output directory")
    from isaaclab.app import AppLauncher

    class CollectionLauncher(AppLauncher):
        def _resolve_device_settings(self, launcher_args):
            super()._resolve_device_settings(launcher_args)
            launcher_args["active_gpu"] = args.gpu
            launcher_args["multi_gpu"] = False

    app = CollectionLauncher(dict(headless=True, enable_cameras=True, device="cpu")).app
    try:
        import torch
        torch.set_num_threads(2)
        with torch.inference_mode():
            collect(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
