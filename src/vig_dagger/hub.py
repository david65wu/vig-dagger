"""Hugging Face Hub transfer of demos, datasets, policies and DAgger collections, and importing existing files.

Repository layout (``hub.dataset_repo``, a dataset repo; ``hub.model_repo``, a model repo):

    dataset repo:  raw/planner/part_<k>.hdf5 + .done.json        planner demos (Isaac Lab demo files)
                   raw/planner_extra/part_<k>.hdf5 + .done.json  extra planner demos (the base_plus_planner baseline)
                   dagger/<experiment>/<round>/<scorer>/ a DAgger collection (trajectories, attempts, episodes)
    model repo:    base/moe/latest.pt                    the 7-level MoE that drives round ``moe``
                   base/thresholds.json, base/demo_scores.npz   its VIG and flow-loss thresholds on demo frames

Checkpoints are uploaded weights-only (``export_checkpoint``: optimizer, scheduler and RNG states removed, about a
third of the size); they still load for inference but cannot resume training.

Uploading needs ``huggingface-cli login`` (or HF_TOKEN) with write access to the repos; downloading public repos
needs nothing.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from vig_dagger.config import Paths


def export_checkpoint(src: Path, dst: Path) -> None:
    """A weights-only copy of a robopolicy checkpoint (loads for inference, cannot resume training)."""
    payload = torch.load(src, map_location="cpu", weights_only=True)
    for key in ("optimizer", "scheduler", "scaler", "rng"):
        payload[key] = {}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(dst)


def _repo(cfg: dict, kind: str) -> str:
    repo = cfg["hub"]["dataset_repo" if kind == "dataset" else "model_repo"]
    if not repo:
        raise SystemExit(f"set hub.{'dataset_repo' if kind == 'dataset' else 'model_repo'} in the config")
    return repo


def _link_tree(src: Path, dst: Path) -> None:
    """Make dst point at src (a relative symlink), replacing an existing link."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} already exists; remove it to replace it with the downloaded copy")
    dst.symlink_to(os.path.relpath(src, dst.parent))


def upload(cfg: dict, what: str) -> None:
    from huggingface_hub import HfApi

    api, paths, revision = HfApi(), Paths.of(cfg), cfg["hub"]["revision"]

    def push(folder: Path, repo: str, repo_type: str, path_in_repo: str, patterns=None) -> None:
        if not folder.exists():
            print(f"[hub] skip {folder} (missing)")
            return
        api.create_repo(repo, repo_type=repo_type, exist_ok=True, private=True)
        print(f"[hub] upload {folder} -> {repo_type} {repo}/{path_in_repo}", flush=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=repo_type, path_in_repo=path_in_repo,
                          revision=revision, allow_patterns=patterns, commit_message=f"upload {path_in_repo}")

    if what == "demos":
        repo = _repo(cfg, "dataset")
        for extra, name in ((False, "planner"), (True, "planner_extra")):
            push(paths.raw_demos(extra).resolve(), repo, "dataset", f"raw/{name}", ["part_*.hdf5", "part_*.done.json"])
    elif what == "policy":
        repo, driver = _repo(cfg, "model"), paths.driver_dir("moe")
        staging = paths.root / "hub_staging" / "base"
        if staging.exists():
            shutil.rmtree(staging)
        export_checkpoint(driver / "latest.pt", staging / "moe" / "latest.pt")
        for extra in ("run_config.json",):
            if (driver / extra).is_file():
                shutil.copy2(driver / extra, staging / "moe" / extra)
        for extra in (paths.thresholds("moe"), paths.demo_scores("moe")):
            if extra.is_file():
                shutil.copy2(extra, staging / extra.name)
        push(staging, repo, "model", "base")
    elif what == "dagger":
        repo = _repo(cfg, "dataset")
        for round_name in cfg["rounds"]:
            for scorer in cfg["dagger"]["scorers"]:
                push(paths.dagger(round_name, scorer), repo, "dataset", f"dagger/{cfg['name']}/{round_name}/{scorer}",
                     ["settings.json", "trajectories/*", "attempts/*", "episodes/*"])
    else:
        raise SystemExit(f"unknown upload target {what!r} (demos, policy or dagger)")


def download(cfg: dict, what: str) -> None:
    """Download into <work_dir>/hub and link the files where the pipeline expects them."""
    from huggingface_hub import snapshot_download

    paths, revision = Paths.of(cfg), cfg["hub"]["revision"]
    if what == "demos":
        repo = _repo(cfg, "dataset")
        local = paths.root / "hub" / "dataset"
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=local,
                          allow_patterns=["raw/planner/*", "raw/planner_extra/*"])  # "*" also matches "/" here
        for extra, name in ((False, "planner"), (True, "planner_extra")):
            if (local / "raw" / name).is_dir():
                _link_tree(local / "raw" / name, paths.raw_demos(extra))
    elif what == "policy":
        repo = _repo(cfg, "model")
        local = paths.root / "hub" / "model"
        snapshot_download(repo, repo_type="model", revision=revision, local_dir=local, allow_patterns=["base/*"])
        driver = paths.driver_dir("moe")
        driver.mkdir(parents=True, exist_ok=True)
        for src in sorted((local / "base" / "moe").iterdir()):
            _link_tree(src, driver / src.name)
        for name in ("thresholds.json", "demo_scores.npz"):
            if (local / "base" / name).is_file():
                _link_tree(local / "base" / name, driver / name)
    else:
        raise SystemExit(f"unknown download target {what!r} (demos or policy)")


def import_demos(cfg: dict, files: list[Path], extra: bool = False) -> None:
    """Link existing planner demo files (Isaac Lab .hdf5, the format ``workers/collect.py`` writes) into the raw demo
    directory as part_<k>, in order."""
    import h5py

    raw = Paths.of(cfg).raw_demos(extra)
    raw.mkdir(parents=True, exist_ok=True)
    for k, src in enumerate(files):
        src = src.resolve()
        dst = raw / f"part_{k:03d}.hdf5"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        with h5py.File(src, "r") as f:
            n = sum(1 for key in f["data"] if key.startswith("demo_"))
        (raw / f"part_{k:03d}.done.json").write_text(json.dumps(dict(part=k, demos=n, imported_from=str(src))))
        print(f"[import] part_{k:03d} <- {src} ({n} demos)")


def import_policy(cfg: dict, checkpoint: Path, thresholds: Path | None = None) -> None:
    """Use an existing MoE checkpoint (weights-only copy) as round ``moe``'s driver, optionally with its thresholds."""
    paths = Paths.of(cfg)
    export_checkpoint(checkpoint, paths.driver_dir("moe") / "latest.pt")
    print(f"[import] {paths.driver_dir('moe') / 'latest.pt'} <- {checkpoint}")
    if thresholds is not None:
        shutil.copy2(thresholds, paths.thresholds("moe"))
        print(f"[import] {paths.thresholds('moe')} <- {thresholds}")
