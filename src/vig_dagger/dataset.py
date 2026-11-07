"""Training datasets in robopolicy's small-files format: conversion from ManiSkill trajectories, and combination.

Layout of a dataset root (what ``robopolicy.datasets`` reads):

    dataset_manifest.json                      {"format": "robopolicy_small_files", "tasks": [...]}
    plug_charger/demo_manifest.json            shards, their demos (key, length) and videos
    plug_charger/shards/shard_<i>.hdf5         data/<demo_key>/{actions (T, 8), obs/robot0_joint_pos (T, 7),
                                               obs/robot0_gripper_qpos (T, 2)}
    plug_charger/videos/<camera>/shard_<i>.mp4 + .json (sidecar: frame count, the demo's segment)

One demo per shard. Step t pairs observation t with action t; ManiSkill's final observation (after the last action)
is dropped. Videos are H.264 (libx264, CRF 18, yuv420p, keyframe every 10 frames, 20 fps), like the original
datasets; training therefore sees compressed frames while the simulator feeds raw renders (about 37 dB PSNR apart).

Combining keeps every input's shard files (hard links when possible) and renumbers only the later inputs' shards,
so a demo keeps its identity, and with it its train/validation assignment, in every combined dataset that starts
with the same base dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from vig_dagger.envs import ENV_ID, POLICY_CAMERAS

TASK, SLUG, FPS = "plug charger", "plug_charger", 20.0


def _ffmpeg_writer(path: Path, height: int, width: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-g", "10", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _write_video(path: Path, frames: h5py.Dataset, count: int, chunk: int = 64) -> tuple[int, int]:
    _, height, width, _ = frames.shape
    proc = _ffmpeg_writer(path, height, width)
    for start in range(0, count, chunk):
        proc.stdin.write(np.ascontiguousarray(frames[start:min(count, start + chunk)], dtype=np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err}")
    return height, width


def _write_manifests(root: Path, shards: list[dict], sources: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    (root / SLUG).mkdir(parents=True, exist_ok=True)
    (root / SLUG / "demo_manifest.json").write_text(json.dumps(dict(
        version=1, task=TASK, task_slug=SLUG, env_id=ENV_ID, control_mode="pd_joint_pos", timestamps_hz=FPS,
        video_fps=FPS, demos_per_shard=1, cameras=list(POLICY_CAMERAS), created_at_utc=now, sources=sources,
        shards=shards), indent=1))
    (root / "dataset_manifest.json").write_text(json.dumps(dict(
        format="robopolicy_small_files", format_version=1, created_at_utc=now, num_tasks=1,
        num_demos=sum(len(s["demos"]) for s in shards),
        tasks=[dict(task=TASK, task_slug=SLUG, manifest=f"{SLUG}/demo_manifest.json")]), indent=1))


def convert(sources: list[tuple[Path, list[int] | None]], root: Path, first_shard: int = 0,
            names: list[str] | None = None) -> int:
    """Convert ManiSkill trajectory files into a new dataset at ``root``.

    Args:
        sources: (trajectory .h5, trajectory indices or None for every ``traj_<n>``) in output order.
        root: New dataset root; must not exist yet (the caller converts into a temporary root and renames it).
        first_shard: Shard id of the first demo.
        names: Per source, the name its demos are called by (``<name>__traj_<n>``); default: the file's stem. The
            names are part of each demo's identity, which decides the train/validation split and the data order.

    Returns:
        The number of converted demos.
    """
    if root.exists():
        raise FileExistsError(root)
    shards, index = [], first_shard
    for position, (path, indices) in enumerate(sources):
        stem = names[position] if names else path.stem
        with h5py.File(path, "r") as f:
            keys = sorted((k for k in f if k.startswith("traj_")), key=lambda k: int(k[5:]))
            if indices is not None:
                keys = [f"traj_{i}" for i in indices]
            for key in keys:
                g = f[key]
                length = len(g["actions"])
                demo_key = f"{stem}__{key}"
                shard_name = f"shard_{index:05d}"
                shard_path = root / SLUG / "shards" / f"{shard_name}.hdf5"
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                qpos = g["obs/agent/qpos"][:length]
                with h5py.File(shard_path, "w") as out:
                    demo = out.create_group(f"data/{demo_key}")
                    demo.attrs.update(source_file=path.name, source_traj_key=key, length=length)
                    demo.create_dataset("actions", data=g["actions"][:].astype(np.float32))
                    demo.create_dataset("obs/robot0_joint_pos", data=qpos[:, :7].astype(np.float32))
                    demo.create_dataset("obs/robot0_gripper_qpos", data=qpos[:, -2:].astype(np.float32))
                videos = {}
                for camera in POLICY_CAMERAS:
                    video = Path("videos") / camera / f"{shard_name}.mp4"
                    height, width = _write_video(root / SLUG / video, g[f"obs/sensor_data/{camera}/rgb"], length)
                    sidecar = dict(video_backend="ffmpeg", fps=FPS, height=height, width=width, frames=length, cam=camera,
                                   shard=shard_name, segments=[dict(demo_key=demo_key, frame_start=0, frame_count=length)])
                    (root / SLUG / video.with_suffix(".json")).write_text(json.dumps(sidecar, indent=1))
                    videos[camera] = dict(mp4=str(video), sidecar=str(video.with_suffix(".json")))
                shards.append(dict(shard_id=index, hdf5=f"shards/{shard_name}.hdf5", videos=videos,
                                   demos=[dict(demo_key=demo_key, length=length, task=TASK, source_file=path.name,
                                               source_traj_key=key)]))
                index += 1
    _write_manifests(root, shards, [str(p) for p, _ in sources])
    return len(shards)


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:  # another filesystem
        shutil.copy2(src, dst)


def combine(inputs: list[tuple[Path, int | None]], root: Path) -> int:
    """A new dataset with the first ``count`` (None: all) demos of each input dataset, in order.

    The first input's shard ids are kept; later inputs' shards are numbered after the largest id so far. Files
    are hard-linked (copied across filesystems). Returns the number of demos.
    """
    if root.exists():
        raise FileExistsError(root)
    shards, sources, next_id = [], [], 0
    for position, (source, count) in enumerate(inputs):
        manifest = json.loads((source / SLUG / "demo_manifest.json").read_text())
        selected = manifest["shards"][:count] if count is not None else manifest["shards"]
        if count is not None and len(selected) < count:
            raise ValueError(f"{source} has {len(selected)} demos, {count} requested")
        for shard in selected:
            new_id = shard["shard_id"] if position == 0 else next_id
            entry = json.loads(json.dumps(shard))
            entry["shard_id"] = new_id
            entry["hdf5"] = f"shards/shard_{new_id:05d}.hdf5"
            _link(source / SLUG / shard["hdf5"], root / SLUG / entry["hdf5"])
            for camera, files in shard["videos"].items():
                if camera not in POLICY_CAMERAS:
                    continue
                video = Path("videos") / camera / f"shard_{new_id:05d}.mp4"
                _link(source / SLUG / files["mp4"], root / SLUG / video)
                _link(source / SLUG / files["sidecar"], root / SLUG / video.with_suffix(".json"))
                entry["videos"][camera] = dict(mp4=str(video), sidecar=str(video.with_suffix(".json")))
            entry["videos"] = {c: v for c, v in entry["videos"].items() if c in POLICY_CAMERAS}
            shards.append(entry)
            next_id = max(next_id, new_id + 1)
        sources.append(f"{source} ({len(selected)} demos)")
    _write_manifests(root, shards, sources)
    return len(shards)


def num_demos(root: Path) -> int:
    return len(json.loads((root / SLUG / "demo_manifest.json").read_text())["shards"])


def ready(root: Path) -> bool:
    return (root / "dataset_manifest.json").is_file()


def finalize(tmp: Path, root: Path) -> None:
    """Move a dataset built in ``tmp`` to ``root`` (the manifest's presence marks a complete dataset)."""
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
