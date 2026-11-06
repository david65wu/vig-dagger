"""Render and validate the toy reach environment; no demo collection or training."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def preview(args):
    import numpy as np
    from PIL import Image, ImageDraw
    import torch

    from vig_dagger.toy_reach_env import mdp
    from vig_dagger.toy_reach_env.config import ToyReachEnvCfg
    from vig_dagger.toy_reach_env.environment import ToyReachEnv

    cfg = ToyReachEnvCfg()
    cfg.tube_radius = args.tube_radius
    cfg.scene.third_person_cam.height = cfg.scene.third_person_cam.width = args.resolution
    cfg.scene.wrist_cam.height = cfg.scene.wrist_cam.width = args.resolution
    env = ToyReachEnv(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    video = None
    panels = []
    checks = {}

    def frame(obs, label):
        images = []
        for name in ("third_person_cam", "wrist_cam"):
            rgb = obs["rgb_camera"][name][0].cpu().numpy()
            assert rgb.shape == (args.resolution, args.resolution, 3), (name, rgb.shape)
            assert rgb.dtype == np.uint8 and rgb.std() > 2, f"Invalid or blank {name}"
            images.append(Image.fromarray(rgb))
        panel = Image.new("RGB", (args.resolution * 2, args.resolution + 28))
        for i, im in enumerate(images):
            panel.paste(im, (i * args.resolution, 28))
        ImageDraw.Draw(panel).text((6, 7), f"{label} | third person (left), wrist (right)", fill="white")
        return panel

    try:
        obs, _ = env.reset(seed=args.seed)
        first = obs["policy"].clone()
        obs, _ = env.reset(seed=args.seed)
        reset_error = float((obs["policy"] - first).abs().max())
        assert reset_error < 1e-4, f"Seeded reset mismatch: {reset_error}"
        checks["seeded_reset_max_error"] = reset_error
        checks["start_position"] = obs["policy"][0, :3].tolist()
        assert float(mdp.line_distance(env)[0, 0]) <= cfg.tube_radius + 0.001
        start = frame(obs, "A: start")
        start.save(args.out / "start.png")
        panels.append(start)
        video = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{start.width}x{start.height}", "-r", "30", "-i", "-", "-an", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(args.out / "reach.mp4")], stdin=subprocess.PIPE,
        )
        path = []
        midpoint_saved = False
        for step in range(720):
            obs, _ = env.step(env.action_toward(cfg.point_b, max_translation=0.008))
            path.append(obs["policy"][0, :3].tolist())
            if step % 2 == 0:
                video.stdin.write(frame(obs, f"A to B: step {step}").tobytes())
            if not midpoint_saved and float(obs["policy"][0, 1]) >= 0:
                panel = frame(obs, "Midpoint")
                panel.save(args.out / "midpoint.png")
                panels.append(panel)
                midpoint_saved = True
            if float(mdp.goal_distance(env)[0, 0]) < 0.001:
                break
        checks["reach_steps"] = step + 1
        checks["goal_error_m"] = float(mdp.goal_distance(env)[0, 0])
        assert bool(mdp.success(env)[0, 0]), checks
        end = frame(obs, "B: goal reached")
        end.save(args.out / "goal.png")
        panels.append(end)
        for _ in range(30):
            video.stdin.write(end.tobytes())
        video.stdin.close()
        assert video.wait() == 0, "ffmpeg failed"
        video = None
        np.save(args.out / "eef_path.npy", np.asarray(path))
        probes = []
        for offset in (0.05, -0.05):
            position = [0.45 + offset, 0.0, 0.35]
            obs = env.place_eef(position)
            error = float((obs["policy"][0, :3] - torch.tensor(position, device=env.device)).norm())
            assert error < 0.001
            probes.append({"requested": position, "actual": obs["policy"][0, :3].tolist(), "error_m": error})
            panel = frame(obs, f"Probe: {offset:+.2f} m off line")
            panel.save(args.out / f"probe_{offset:+.2f}.png")
            panels.append(panel)
        checks["probes"] = probes
        checks["camera_shape"] = [args.resolution, args.resolution, 3]
        checks["tube_radius_m"] = cfg.tube_radius
        montage = Image.new("RGB", (panels[0].width, panels[0].height * len(panels)))
        for i, panel in enumerate(panels):
            montage.paste(panel, (0, i * panel.height))
        montage.save(args.out / "camera_preview.png")
        (args.out / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
        print("TOY_REACH_VALIDATED " + json.dumps(checks), flush=True)
    except Exception:
        # Keep a view of the physical scene if a motion/observation check fails.
        for _ in range(3):
            env.sim.render()
        for name in ("third_person_cam", "wrist_cam"):
            env.scene[name].update(env.physics_dt, force_recompute=True)
        failed_obs = env.observation_manager.compute(update_history=False)
        frame(failed_obs, "Validation failure").save(args.out / "failure.png")
        raise
    finally:
        if video is not None:
            video.stdin.close()
            video.wait()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tube-radius", type=float, default=0.01)
    args = parser.parse_args()
    if args.resolution < 32 or args.resolution % 2:
        parser.error("resolution must be an even number >=32")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("use a fresh output directory")
    from isaaclab.app import AppLauncher

    class PreviewLauncher(AppLauncher):
        def _resolve_device_settings(self, launcher_args):
            super()._resolve_device_settings(launcher_args)
            launcher_args["active_gpu"] = args.gpu
            launcher_args["multi_gpu"] = False

    app = PreviewLauncher(dict(headless=not args.gui, enable_cameras=True, device="cpu")).app
    try:
        import torch
        with torch.inference_mode():
            preview(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Kit's close() can otherwise mask a failed validation with exit code 0.
    app.close()
