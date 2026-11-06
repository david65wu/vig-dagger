"""Hugging Face Hub transfer of demos, MoE checkpoints and DAgger collections, and importing existing files.

Repository layout:

    hub.dataset_repo (dataset):  raw/planner/part_<k>.h5 + .json        the planner demos (ManiSkill trajectories)
                                 dagger/<experiment>/<collection>/      a DAgger collection (``upload dagger``)
    moe.download[sd<seed>] (model): base/moe/latest.pt                  a MoE checkpoint (the seed-0 MoE), with its
                                    base/moe/metrics.jsonl, train_config.json   training log and run config

A downloaded MoE brings only its checkpoint, training log and config: its thresholds are recomputed here from demo
frames, with the same code as for the MoEs trained here (the repo's own base/thresholds.json is not used).

Uploaded checkpoints are weights-only (``export_checkpoint``: optimizer, scheduler and RNG states removed, about a
third of the size); they still load for inference but cannot resume training. Uploading needs ``hf auth login`` (or
HF_TOKEN) with write access to the repos; downloading public repos needs nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from vig_dagger.config import Paths, collections, downloaded


def export_checkpoint(src: Path, dst: Path) -> None:
    """A weights-only copy of a robopolicy checkpoint (loads for inference, cannot resume training)."""
    payload = torch.load(src, map_location="cpu", weights_only=True)
    for key in ("optimizer", "scheduler", "scaler", "rng"):
        payload[key] = {}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(dst)


def _dataset_repo(cfg: dict) -> str:
    repo = cfg["hub"]["dataset_repo"]
    if not repo:
        raise SystemExit("set hub.dataset_repo in the config")
    return repo


def _link(src: Path, dst: Path) -> None:
    """Make dst point at src (a relative symlink), replacing an existing link."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} already exists; remove it to replace it with the downloaded copy")
    dst.symlink_to(os.path.relpath(src, dst.parent))


def download_demos(cfg: dict) -> None:
    """The planner demos into <work_dir>/hub/dataset, linked as data/raw/planner."""
    from huggingface_hub import snapshot_download

    paths = Paths.of(cfg)
    local = paths.root / "hub" / "dataset"
    snapshot_download(_dataset_repo(cfg), repo_type="dataset", revision=cfg["hub"]["revision"], local_dir=local,
                      allow_patterns=["raw/planner/*"])  # "*" also matches "/" here
    if not (local / "raw" / "planner").is_dir():
        raise SystemExit(f"{_dataset_repo(cfg)} has no raw/planner/")
    _link(local / "raw" / "planner", paths.raw_demos())


def download_moe(cfg: dict, seed: int) -> None:
    """The MoE of one seed from its repo in moe.download, linked as policies/moe/sd<seed>/latest.pt."""
    from huggingface_hub import snapshot_download

    repo = downloaded(cfg, seed)
    if not repo:
        raise SystemExit(f"moe.download has no repo for seed {seed} (sd{seed}); that seed is trained here")
    paths = Paths.of(cfg)
    local = paths.root / "hub" / "model" / f"sd{seed}"
    snapshot_download(repo, repo_type="model", revision=cfg["hub"]["revision"], local_dir=local,
                      allow_patterns=["base/moe/*"])
    src = local / "base" / "moe"
    if not (src / "latest.pt").is_file():
        raise SystemExit(f"{repo} has no base/moe/latest.pt")
    # The checkpoint, and its training log and config (train_config.json, not run_config.json: that name would make
    # a later training of the same seed resume from it).
    for name in ("latest.pt", "metrics.jsonl", "train_config.json"):
        if (src / name).is_file():
            _link(src / name, paths.moe_dir(seed) / name)
    print(f"[hub] MoE seed {seed} <- {repo}", flush=True)


def upload(cfg: dict, what: str) -> None:
    from huggingface_hub import HfApi

    api, paths, revision = HfApi(), Paths.of(cfg), cfg["hub"]["revision"]

    def push(folder: Path, repo: str, path_in_repo: str, patterns=None) -> None:
        if not folder.exists():
            print(f"[hub] skip {folder} (missing)")
            return
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
        print(f"[hub] upload {folder} -> dataset {repo}/{path_in_repo}", flush=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset", path_in_repo=path_in_repo,
                          revision=revision, allow_patterns=patterns, commit_message=f"upload {path_in_repo}")

    repo = _dataset_repo(cfg)
    if what == "demos":
        push(paths.raw_demos().resolve(), repo, "raw/planner", ["part_*.h5", "part_*.json"])
    elif what == "dagger":
        for c in collections(cfg):
            push(paths.dagger(c), repo, f"dagger/{cfg['name']}/{c.id}",
                 ["settings.json", "trajectories/*", "attempts/*", "episodes/*"])
    else:
        raise SystemExit(f"unknown upload target {what!r} (demos or dagger)")


def import_demos(cfg: dict, files: list[Path]) -> None:
    """Link existing planner trajectory files (with their .json) into the raw demo directory as part_<k>, in order."""
    raw = Paths.of(cfg).raw_demos()
    raw.mkdir(parents=True, exist_ok=True)
    for k, src in enumerate(files):
        src = src.resolve()
        for suffix in (".h5", ".json"):
            dst = raw / f"part_{k:03d}{suffix}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.with_suffix(suffix))
        with open(src.with_suffix(".json")) as fh:
            n = sum(1 for e in json.load(fh)["episodes"] if e.get("success", True))
        (raw / f"part_{k:03d}.done.json").write_text(json.dumps(dict(part=k, demos=n, imported_from=str(src))))
        print(f"[import] part_{k:03d} <- {src} ({n} demos)")


def import_moe(cfg: dict, seed: int, checkpoint: Path) -> None:
    """Use an existing MoE checkpoint (weights-only copy) as the MoE of a seed; its thresholds are computed here."""
    dst = Paths.of(cfg).moe_checkpoint(seed)
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"{dst} already exists")
    export_checkpoint(checkpoint, dst)
    print(f"[import] {dst} <- {checkpoint}")

