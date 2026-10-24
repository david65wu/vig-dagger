"""Score demo frames with VIG and flow loss, as DAgger does, for a round's thresholds (part 2).

The frames are every ``thresholds.frame_stride``-th frame of the first ``thresholds.num_demos`` raw planner demos
(Isaac Lab demo files: the policy state and the raw RGB renders before each step), scored with the round's driving
policies. Job --chunk of
--chunks takes every --chunks-th of those demos. Torch is reseeded per frame from (demo, frame), so the result does
not depend on how the demos are split over jobs. ``steps.py thresholds`` merges the chunks into the percentiles.

Output (``<driver dir>/demo_scores_parts/part_<chunk>.npz``): demo index, frame, vig, vig_gap (frames, levels - 1),
flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from vig_dagger import config as config_module
from vig_dagger.config import Paths, driver_spec
from vig_dagger.jobs import exit_on_sigterm
from vig_dagger.policy import PolicySet, check_cameras
from vig_dagger.scoring import build_scorer


def demo_list(cfg: dict) -> list[tuple[Path, int]]:
    """(raw part file, demo index) of the first thresholds.num_demos planner demos, in order."""
    out, need = [], int(cfg["thresholds"]["num_demos"])
    for part in sorted(Paths.of(cfg).raw_demos().glob("part_*.hdf5")):
        with h5py.File(part, "r") as f:
            keys = sorted(int(k[5:]) for k in f["data"] if k.startswith("demo_"))
        out += [(part, i) for i in keys[: need - len(out)]]
        if len(out) >= need:
            break
    if not out:
        raise SystemExit(f"no raw demos in {Paths.of(cfg).raw_demos()}")
    return out


def part_path(cfg: dict, round_name: str, chunk: int) -> Path:
    return Paths.of(cfg).driver_dir(round_name) / "demo_scores_parts" / f"part_{chunk:02d}.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--round", required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--chunks", type=int, required=True)
    args = p.parse_args()
    exit_on_sigterm()

    cfg = config_module.load(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policies = PolicySet.load(driver_spec(cfg, args.round), device)
    check_cameras(policies)
    vig, flow = build_scorer("vig", cfg["scorers"]["vig"]), build_scorer("flow", cfg["scorers"]["flow"])
    stride = int(cfg["thresholds"]["frame_stride"])
    demos = list(enumerate(demo_list(cfg)))[args.chunk::args.chunks]
    rows = {k: [] for k in ("demo", "frame", "vig", "vig_gap", "flow")}
    for d, (path, i) in demos:
        with h5py.File(path, "r") as f:
            g = f[f"data/demo_{i}"]
            length = len(g["actions"])
            ee = torch.from_numpy(g["obs/pose/ee_pose"][:length]).float()
            joints = torch.from_numpy(g["obs/joint_pos"][:length]).float()
            cams = [g[f"obs/rgb_camera/{c}_cam"] for c in policies.cameras]
            for t in range(0, length, stride):
                state = torch.cat([ee[t], joints[t, -2:]])[None].to(device)  # isaac.policy_state
                images = torch.stack([torch.from_numpy(c[t]).permute(2, 0, 1) for c in cams])[None].to(device)
                torch.manual_seed(d * 10_000 + t)
                v = vig(policies, state, images)
                torch.manual_seed(d * 10_000 + t + 5_000)
                fl = flow(policies, state, images)
                rows["demo"].append(d); rows["frame"].append(t)
                rows["vig"].append(v.value); rows["flow"].append(fl.value)
                rows["vig_gap"].append([v.details["w1_by_factor"][f] for f in policies.factors[1:]])
        print(f"demo {d} ({path.name} demo_{i}): {len(range(0, length, stride))} frames", flush=True)
    out = part_path(cfg, args.round, args.chunk)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in rows.items()}, factors=np.asarray(policies.factors))
    tmp.rename(out)


if __name__ == "__main__":
    main()
