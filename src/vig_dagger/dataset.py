"""Training datasets in robopolicy's small-files format: conversion from ManiSkill trajectories, and combination.

Layout of a dataset root (what ``robopolicy.datasets`` reads):

    dataset_manifest.json                      {"format": "robopolicy_small_files", "tasks": [...]}
    plug_charger/demo_manifest.json            shards, their demos (key, length) and videos
    plug_charger/shards/shard_<i>.hdf5         data/<demo_key>/{actions (T, 8), obs/robot0_joint_pos (T, 7),
                                               obs/robot0_gripper_qpos (T, 2)}
    plug_charger/videos/<camera>/shard_<i>.mp4 + .json (sidecar: frame count, the demo's segment)

One episode per shard. Step t pairs observation t with action t; ManiSkill's final observation (after the last
action) is dropped. With ``cut_random_moves`` a DAgger episode (one with ``dagger/controller``, 2 = random move) is cut
at its random moves: its shard holds one demo per piece before, between and after them (``<demo key>_s<start>-<end>``,
episode steps [start, end)), the episode's videos stay whole and each piece's video segment points at its frames.
Training samples need the full 16-step action chunk inside a demo, so the 15 steps before each move leave training too
(their chunk would contain the move), while the expert's recovery after each move stays in; pieces shorter than
``min_length`` are dropped. This is maniskill's split_dagger_dataset_at_random_moves.py, applied while converting. Videos are H.264 (libx264, CRF 18, yuv420p, keyframe every 10 frames, 20 fps), like the original
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


def _runs(keep: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of the True runs."""
    m = np.r_[False, keep.astype(bool), False]
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def convert(sources: list[tuple[Path, list[int] | None]], root: Path, first_shard: int = 0,
            names: list[str] | None = None, cut_random_moves: bool = False, min_length: int = 16) -> int:
    """Convert ManiSkill trajectory files into a new dataset at ``root``.

    Args:
        sources: (trajectory .h5, trajectory indices or None for every ``traj_<n>``) in output order.
        root: New dataset root; must not exist yet (the caller converts into a temporary root and renames it).
        first_shard: Shard id of the first demo.
        names: Per source, the name its demos are called by (``<name>__traj_<n>``); default: the file's stem. The
            names are part of each demo's identity, which decides the train/validation split and the data order.
        cut_random_moves: Cut DAgger episodes at their random moves (see the module docstring).
        min_length: Pieces shorter than this are dropped (16: the action chunk).

    Returns:
        The number of converted episodes (shards).
    """
    if root.exists():
        raise FileExistsError(root)
    shards, index = [], first_shard
    stats = dict(episodes=0, with_moves=0, moves=0, steps=0, move_steps=0, pieces=0, short_pieces=0, short_steps=0)
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
                actions = g["actions"][:].astype(np.float32)
                pieces = [(0, length)]  # [start, end) of each demo in the episode
                if cut_random_moves and "dagger/controller" in g:
                    move = g["dagger/controller"][:length] == 2
                    stats["episodes"] += 1
                    if move.any():
                        stats["with_moves"] += 1
                        stats["moves"] += len(_runs(move))
                        stats["steps"] += length
                        stats["move_steps"] += int(move.sum())
                        runs = _runs(~move)
                        pieces = [(a, b) for a, b in runs if b - a >= min_length]
                        stats["short_pieces"] += len(runs) - len(pieces)
                        stats["short_steps"] += sum(b - a for a, b in runs if b - a < min_length)
                    stats["pieces"] += len(pieces)
                cut = pieces != [(0, length)]
                keys_in = [demo_key if not cut else f"{demo_key}_s{a}-{b}" for a, b in pieces]
                with h5py.File(shard_path, "w") as out:
                    for k, (a, b) in zip(keys_in, pieces):
                        demo = out.create_group(f"data/{k}")
                        demo.attrs.update(source_file=path.name, source_traj_key=key, length=b - a)
                        if cut:
                            demo.attrs.update(source_demo_key=demo_key, segment_start=a, segment_end=b,
                                              random_move_steps_cut=True)
                        demo.create_dataset("actions", data=actions[a:b])
                        demo.create_dataset("obs/robot0_joint_pos", data=qpos[a:b, :7].astype(np.float32))
                        demo.create_dataset("obs/robot0_gripper_qpos", data=qpos[a:b, -2:].astype(np.float32))
                videos = {}
                for camera in POLICY_CAMERAS:
                    video = Path("videos") / camera / f"{shard_name}.mp4"
                    height, width = _write_video(root / SLUG / video, g[f"obs/sensor_data/{camera}/rgb"], length)
                    sidecar = dict(video_backend="ffmpeg", fps=FPS, height=height, width=width, frames=length, cam=camera,
                                   shard=shard_name, segments=[dict(demo_key=k, frame_start=a, frame_count=b - a)
                                                               for k, (a, b) in zip(keys_in, pieces)])
                    (root / SLUG / video.with_suffix(".json")).write_text(json.dumps(sidecar, indent=1))
                    videos[camera] = dict(mp4=str(video), sidecar=str(video.with_suffix(".json")))
                demos = []
                for k, (a, b) in zip(keys_in, pieces):
                    entry = dict(demo_key=k, length=b - a, task=TASK, source_file=path.name, source_traj_key=key)
                    if cut:
                        entry.update(source_demo_key=demo_key, segment_start=a, segment_end=b)
                    demos.append(entry)
                shards.append(dict(shard_id=index, hdf5=f"shards/{shard_name}.hdf5", videos=videos, demos=demos))
                index += 1
    _write_manifests(root, shards, [str(p) for p, _ in sources])
    if cut_random_moves:
        top = json.loads((root / "dataset_manifest.json").read_text())
        top["random_move_steps_cut"] = dict(min_length=int(min_length), **stats)
        (root / "dataset_manifest.json").write_text(json.dumps(top, indent=1))
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
    shards, sources, next_id, cuts = [], [], 0, []
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
        cut = cut_stats(source)
        if cut is not None:
            if count is not None and count != len(manifest["shards"]):
                raise ValueError(f"{source}: combining part of a cut dataset would misreport its random-move counts")
            cuts.append(cut)
    _write_manifests(root, shards, sources)
    if cuts:
        top = json.loads((root / "dataset_manifest.json").read_text())
        top["random_move_steps_cut"] = {k: (cuts[0][k] if k == "min_length" else sum(c[k] for c in cuts)) for k in cuts[0]}
        (root / "dataset_manifest.json").write_text(json.dumps(top, indent=1))
    return len(shards)


def num_demos(root: Path) -> int:
    """Episodes (shards); a cut episode's pieces count once."""
    return len(json.loads((root / SLUG / "demo_manifest.json").read_text())["shards"])


def cut_stats(root: Path) -> dict | None:
    """The random-move cut's counts (``convert(cut_random_moves=True)``), summed over a combined dataset's inputs."""
    return json.loads((root / "dataset_manifest.json").read_text()).get("random_move_steps_cut")


def ready(root: Path) -> bool:
    return (root / "dataset_manifest.json").is_file()


def finalize(tmp: Path, root: Path) -> None:
    """Move a dataset built in ``tmp`` to ``root`` (the manifest's presence marks a complete dataset)."""
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
