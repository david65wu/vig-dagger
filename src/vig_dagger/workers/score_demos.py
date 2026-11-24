"""Score demo frames with VIG (and flow loss), as DAgger does, for one MoE's thresholds (part 2).

The frames are every ``thresholds.frame_stride``-th frame of the first ``thresholds.num_demos`` raw planner demos
(ManiSkill trajectories: joint state and raw RGB renders), scored with the MoE of one seed. In iteration mode
(``--moe round<r>``) they are every ``frame_stride``-th frame of ALL the round-r MoE's training demos: the base
planner demos and the successful DAgger episodes of rounds 1 ... r-1 that its training set holds, each cut at its
random moves like the training set (``dataset.training_pieces``), read from the raw trajectories so the scores are
computed on raw renders as during DAgger. Job --chunk of --chunks takes every --chunks-th of those demos. Torch is
reseeded per frame from (demo, frame), so the result does not depend on how the demos are split over jobs.
``thresholds.scorers`` picks the scores (flow loss is skipped without "flow"). ``steps.py thresholds`` merges the
chunks into the percentiles and ``demo_scores.npz``.

Output (``policies/moe/<moe>/demo_scores_parts/part_<chunk>.npz``): demo index, frame, vig, vig_gap (frames, levels - 1),
flow (when scored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from vig_dagger import config as config_module
from vig_dagger.config import Paths, collections, moe_key, target_of
from vig_dagger.dataset import training_pieces
from vig_dagger.jobs import exit_on_sigterm
from vig_dagger.policy import PolicySet, check_cameras
from vig_dagger.scoring import build_scorer


def demo_list(cfg: dict) -> list[tuple[Path, int]]:
    """(raw part file, trajectory index) of the first thresholds.num_demos planner demos, in order."""
    out, need = [], int(cfg["thresholds"]["num_demos"])
    for part in sorted(Paths.of(cfg).raw_demos().glob("part_*.h5")):
        with h5py.File(part, "r") as f:
            keys = sorted(int(k[5:]) for k in f if k.startswith("traj_"))
        out += [(part, i) for i in keys[: need - len(out)]]
        if len(out) >= need:
            break
    if not out:
        raise SystemExit(f"no raw demos in {Paths.of(cfg).raw_demos()}")
    return out


def round_demo_list(cfg: dict, r: int) -> list[tuple[Path, int]]:
    """(raw trajectory file, trajectory index) of every demo the round-r MoE trains on, in dataset order: the base
    planner demos (``demos.num_demos``), then each earlier round's DAgger episodes (``steps.dagger_sources``)."""
    from vig_dagger.steps import dagger_sources

    out, need = [], int(cfg["demos"]["num_demos"])
    for part in sorted(Paths.of(cfg).raw_demos().glob("part_*.h5")):
        with h5py.File(part, "r") as f:
            keys = sorted(int(k[5:]) for k in f if k.startswith("traj_"))
        out += [(part, i) for i in keys[: need - len(out)]]
        if len(out) >= need:
            break
    for c in collections(cfg)[: r - 1]:
        out += [(path, i) for path, indices in dagger_sources(Paths.of(cfg).dagger(c), target_of(cfg, c)) for i in indices]
    return out


def part_path(cfg: dict, seed: int | str, chunk: int) -> Path:
    return Paths.of(cfg).moe_dir(seed) / "demo_scores_parts" / f"part_{chunk:02d}.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, help="the MoE seed")
    p.add_argument("--moe", help="the MoE: a seed, or round<r> in iteration mode (default --seed)")
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--chunks", type=int, required=True)
    args = p.parse_args()
    exit_on_sigterm()

    cfg = config_module.load(args.config)
    moe = moe_key(args.moe) if args.moe is not None else args.seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policies = PolicySet.load(str(Paths.of(cfg).moe_checkpoint(moe)), device)
    check_cameras(policies)
    with_flow = "flow" in cfg["thresholds"]["scorers"]
    vig = build_scorer("vig", cfg["scorers"]["vig"])
    flow = build_scorer("flow", cfg["scorers"]["flow"]) if with_flow else None
    stride = int(cfg["thresholds"]["frame_stride"])
    if isinstance(moe, str):  # iteration mode: all the round MoE's training demos, cut like its training set
        r = cfg["retrain"]
        listed = round_demo_list(cfg, int(moe[5:]))
        cut, min_length = bool(r["cut_random_moves"]), int(r["min_piece_length"])
    else:
        listed, cut, min_length = demo_list(cfg), False, 16
    demos = list(enumerate(listed))[args.chunk::args.chunks]
    rows = {k: [] for k in ("demo", "frame", "vig", "vig_gap", *(["flow"] if with_flow else []))}
    for d, (path, i) in demos:
        with h5py.File(path, "r") as f:
            g = f[f"traj_{i}"]
            length = len(g["actions"])
            qpos = torch.from_numpy(g["obs/agent/qpos"][:length]).float()
            cams = [g[f"obs/sensor_data/{c}/rgb"] for c in policies.cameras]
            frames = [t for a, b in training_pieces(g, cut, min_length) for t in range(a, b, stride)]
            for t in frames:
                state = torch.cat([qpos[t, :7], qpos[t, -2:]])[None].to(device)
                images = torch.stack([torch.from_numpy(c[t]).permute(2, 0, 1) for c in cams])[None].to(device)
                torch.manual_seed(d * 10_000 + t)
                v = vig(policies, state, images)
                rows["demo"].append(d); rows["frame"].append(t)
                rows["vig"].append(v.value)
                rows["vig_gap"].append([v.details["w1_by_factor"][f] for f in policies.factors[1:]])
                if with_flow:
                    torch.manual_seed(d * 10_000 + t + 5_000)
                    rows["flow"].append(flow(policies, state, images).value)
        print(f"demo {d} ({path.name} traj_{i}): {len(frames)} frames", flush=True)
    out = part_path(cfg, moe, args.chunk)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in rows.items()}, factors=np.asarray(policies.factors))
    tmp.rename(out)


if __name__ == "__main__":
    main()
