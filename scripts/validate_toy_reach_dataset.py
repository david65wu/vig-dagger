"""Check a completed collection and make a combined MP4, ZIP, and path plot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import zipfile

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.index import read_manifest, validate_entries


def check_video(item):
    path, expected_frames, shape, fps = item
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path),
    ]))["streams"][0]
    assert int(metadata["nb_frames"]) == expected_frames, (path, metadata, expected_frames)
    assert (metadata["height"], metadata["width"]) == shape, (path, metadata)
    numerator, denominator = map(int, metadata["r_frame_rate"].split("/"))
    assert numerator / denominator == fps
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
                    "-f", "null", "-"], check=True, capture_output=True)
    return str(path)


def main(root):
    summary = json.loads((root / "summary.json").read_text())
    settings, demos = summary["settings"], summary["demos"]
    sphere = settings.get('task_variant', 'line') in ('sphere', 'fixed_goal_sphere')
    state_dim = settings.get('state_dim', 9)
    schema = FeatureSchema.model_validate({"features": [
        {"key": "actions", "shape": [7], "paths": ["actions"], "normalization": "minmax", "future": 15},
        {"key": "state", "shape": [state_dim], "paths": ["obs/state"], "normalization": "minmax"},
        {"key": "images", "shape": [3, 256, 256], "source": "video", "dtype": "uint8",
         "cameras": settings["cameras"]},
    ]})
    entries = read_manifest(root / "dataset")
    episodes = validate_entries(entries, schema)
    assert len(episodes) == len(demos) == settings["num_demos"]
    checks, paths, endpoints = [], [], []
    for episode, demo in zip(episodes, demos):
        a = np.asarray(demo.get('point_a', settings['point_a']))
        b = np.asarray(demo.get('point_b', settings['point_b']))
        endpoints.append((a,b))
        assert episode.episode_id == demo["demo"] and demo["success"]
        t = episode.length
        with h5py.File(episode.path) as f:
            g = f[f"data/{episode.episode_id}"]
            state, next_state, actions = g["obs/state"][:], g["next_state"][:], g["actions"][:]
            assert state.shape == next_state.shape == (t, state_dim) and actions.shape == (t, 7)
            assert g['obs/robot0_gripper_qpos'].shape == (t, 2)
            if sphere:
                center = np.asarray(settings['sphere_center'])
                assert np.allclose(np.linalg.norm(np.stack([a,b])-center, axis=1), settings['sphere_radius_m'], atol=1e-7)
                assert np.linalg.norm(b-a) >= settings['min_endpoint_separation_m']
                if settings['task_variant'] == 'fixed_goal_sphere':
                    assert np.allclose(b, settings['point_b'], atol=1e-8)
                    assert np.dot(a-center, b-center) <= 1e-12
                assert np.array_equal(g.attrs['point_a'],a) and np.array_equal(g.attrs['point_b'],b)
                assert np.allclose(state[:,9:12],b,atol=1e-7) and np.allclose(next_state[:,9:12],b,atol=1e-7)
                assert np.array_equal(g['obs/goal_xyz'][:],state[:,9:12])
                assert np.array_equal(np.concatenate([g['obs/robot0_eef_pos'][:],g['obs/robot0_eef_quat'][:],
                    g['obs/robot0_gripper_qpos'][:],g['obs/goal_xyz'][:]],axis=1),state)
            assert all(np.isfinite(value).all() for value in (state, next_state, actions))
            assert np.array_equal(state[1:], next_state[:-1])
            assert np.array_equal(g["initial_state/articulation/robot/joint_position"][0], g["obs/joint_pos"][0])
            assert np.allclose(g["timestamps"][:], np.arange(t) / 60)
            positions = np.concatenate((state[:, :3], next_state[-1:, :3]))
            projection = (positions - a) @ (b - a) / np.dot(b - a, b - a)
            radius = np.linalg.norm(positions - a - projection.clip(0, 1)[:, None] * (b - a), axis=1)
            assert radius.max() <= settings["tube_radius_m"]
            assert np.linalg.norm(next_state[-15:, :3] - b, axis=1).max() < .0015
            paths.append((projection, positions, radius))
            for camera, video in episode.videos.items():
                raw = g[f"obs/rgb_camera/{camera}_cam"]
                assert raw.shape == (t, 256, 256, 3) and raw.dtype == np.uint8
                # The wrist and external cameras should both change during the motion.
                assert np.mean(np.abs(raw[0].astype(float) - raw[-1].astype(float))) > .5
                checks.append((video.path, t, (256, 256), 60))
        checks.append((root / demo["mp4"], (t + 1) // 2 + 15, (284, 512), 30))
    with ThreadPoolExecutor(max_workers=4) as pool:
        validated = list(pool.map(check_video, checks))

    concat = root / "review_playlist.txt"
    concat.write_text("".join(f"file '{demo['mp4']}'\n" for demo in demos))
    combined = root / f"all_{len(demos)}_demos.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", "-movflags", "+faststart", str(combined)], check=True)
    check_video((combined, sum((d["steps"] + 1) // 2 + 15 for d in demos), (284, 512), 30))
    with zipfile.ZipFile(root / "individual_demo_mp4s.zip", "x", compression=zipfile.ZIP_STORED) as archive:
        for demo in demos:
            archive.write(root / demo["mp4"], arcname=Path(demo["mp4"]).name)

    if sphere:
        fig = plt.figure(figsize=(13,5))
        spatial = fig.add_subplot(121, projection='3d')
        deviation = fig.add_subplot(122)
        u,v = np.meshgrid(np.linspace(0,2*np.pi,32),np.linspace(0,np.pi,16))
        center=np.asarray(settings['sphere_center']);r=settings['sphere_radius_m']
        spatial.plot_wireframe(center[0]+r*np.cos(u)*np.sin(v),center[1]+r*np.sin(u)*np.sin(v),
                               center[2]+r*np.cos(v),color='gray',alpha=.15,linewidth=.4)
        for i, ((progress, positions, radius),(a,b)) in enumerate(zip(paths,endpoints)):
            color=plt.get_cmap('tab20')(i)
            spatial.plot(*positions.T,color=color,alpha=.85)
            spatial.scatter(*a,color=color,marker='o',s=15)
            spatial.scatter(*b,color=color,marker='*',s=35)
            deviation.plot(progress,radius*1000,color=color,alpha=.85)
        spatial.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='A: circles; B: stars')
        spatial.set_box_aspect((1,1,1))
        deviation.set(xlabel='Progress from A to B',ylabel='Distance from chord (mm)')
        deviation.axhline(settings['tube_radius_m']*1000,color='black',ls='--',label='Tube radius')
        deviation.grid(alpha=.2);deviation.legend()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        a,b=endpoints[0]
        for i, (progress, positions, radius) in enumerate(paths):
            color = plt.get_cmap("tab20")(i)
            axes[0].plot(progress, (positions[:, 0] - a[0]) * 1000, color=color, alpha=.85)
            axes[1].plot(progress, (positions[:, 2] - a[2]) * 1000, color=color, alpha=.85)
            axes[2].plot(progress, radius * 1000, color=color, alpha=.85)
        for ax, title in zip(axes, ["X deviation from line (mm)", "Z deviation from line (mm)", "Distance from segment (mm)"]):
            ax.set(xlabel="Progress from A to B", ylabel=title)
            ax.grid(alpha=.2)
        for ax in axes[:2]:
            ax.set_ylim(-11, 11)
            ax.axhline(10, color="black", ls="--", alpha=.5)
            ax.axhline(-10, color="black", ls="--", alpha=.5)
        axes[2].set_ylim(0, 11)
        axes[2].axhline(10, color="black", ls="--", label="Tube radius: 10 mm")
        axes[2].legend()
    fig.suptitle(f"{len(demos)} measured planner trajectories — one color per demonstration")
    fig.tight_layout()
    fig.savefig(root / "trajectories.png", dpi=160)
    plt.close(fig)

    result = dict(num_demos=len(demos), total_steps=sum(d["steps"] for d in demos),
                  task_variant=settings.get('task_variant','line'),state_dim=state_dim,
                  videos_decoded=len(validated) + 1,
                  max_goal_error_mm=max(d["goal_error_m"] for d in demos) * 1000,
                  max_line_distance_mm=max(d["max_line_distance_m"] for d in demos) * 1000,
                  dataset_index_and_alignment="passed", initial_states="passed",
                  raw_camera_motion="passed", video_frame_counts_and_decode="passed")
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
