"""The experiment configuration (a YAML file) and where every output of an experiment goes.

The experiment (branch ``plugcharger-moe-seeds``): DAgger collections driven by 7-level resolution
mixture-of-experts (MoE) policies that differ only in their training seed. Each MoE drives at full resolution and
gives both query scores (VIG over its own degraded levels, flow loss at full resolution). There is one **collection**
for every combination of

- MoE seed (``moe.seeds``; a seed listed in ``moe.download`` is downloaded, the others are trained here),
- query scorer (``dagger.scorers``: ``vig``, ``flow``),
- threshold (``dagger.thresholds``: ``p<q>``, the q-th percentile of that MoE's score on demo frames). The same
  threshold decides both directions: the expert takes over after ``dagger.take_count`` scores above it and hands
  back after ``dagger.return_count`` scores below it,
- variant (``dagger.variants``): ``plain``, or ``random_move``: at every takeover the gripper first moves to a random
  pose nearby (``dagger.random_move``), then the expert replans.

Each collection runs until it has ``dagger.num_demos`` successful episodes; its metrics (collection success rate,
expert share of the DAgger data, expert share of the failed attempts) are the results.

Several machines can share the work through a task board (``sharing.board``, see ``board.py``): each machine claims
tasks itself, however many machines there are.

Values missing from the YAML file take the defaults below; ``--set key.sub=value`` overrides one value on the
command line (the value is parsed as YAML, so ``--set dagger.num_demos=200`` gives an int).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCORERS = ("vig", "flow")
VARIANTS = ("plain", "random_move")

DEFAULTS: dict[str, Any] = {
    "name": "plugcharger_moe_seeds",
    "work_dir": None,
    "resources": {
        "gpus": "auto",  # CUDA indices, or "auto": every GPU nvidia-smi lists
        "sim_slots": "auto",  # CPU budget in simulator-job units (a simulator job uses about one core); auto: cores // 2
        "max_load": "auto",  # start nothing new while the 1-minute load average is above this; auto: CPU cores
        "sim_gpu_memory_gb": 4,  # free GPU memory a simulator job needs to start
        "train_gpu_memory_gb": 8,  # free GPU memory a training needs to start (measured: 2.6-2.9 GB; one of ours per GPU)
        "sim_ram_gb": 4,
        "train_ram_gb": 24,  # measured: ~19 GB with 16 data-loading workers
        "train_slots": 2,  # CPU budget a training's data loading takes, in simulator-job units (measured: ~1.3 cores)
        "dataloader_workers": 16,
    },
    # Optional: every number and figure is always written to <work_dir>/results (REPORT.md, JSON, CSV, PNG).
    "wandb": {"enabled": False, "entity": "vig-dagger-exp", "project": "plugcharger"},
    "hub": {"dataset_repo": None, "revision": "main"},
    # Several machines: the task board, "host:/path" (reached over ssh) or a directory every machine mounts; every
    # machine runs the same config and claims tasks there (board.py). None: this machine alone (or tasks by hand).
    "sharing": {"board": None, "poll_s": 60, "heartbeat_s": 600, "publish_trajectories": False},
    "demos": {
        "source": "download",  # download (hub.dataset_repo) | collect
        "num_demos": 1000,  # planner demos the MoEs train on ("base")
        "demos_per_part": 50,
        "seed_start": 2_000_000,  # collect: part k collects from scene seed seed_start + 1000 * k
        "strategy_seed": 1,  # collect: planner strategy stream; part k uses strategy_seed + k
        "max_episode_steps": 300,
        "convert_workers": 8,
        "demo_names": None,  # what the converted demos are called: a pattern with {k}, the part number; None: the file name
    },
    "training": {
        "epochs": 60,  # passes over the training demos (60 on the 1000 demos = 188,580 updates, as the seed-0 MoE)
        "batch_size": 64,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_updates": 1000,
        "min_lr_ratio": 1.0e-4,
        "val_fraction": 0.02,
        "checkpoint_every": 5000,
        "keep_checkpoints": 1,  # rolling step checkpoints kept for resuming (the final latest.pt is always kept)
        "max_restarts": 5,
    },
    "moe": {
        "seeds": [0, 1, 2],
        "factors": [1, 2, 4, 8, 16, 32, 64],  # its resolution levels (1 = full resolution)
        "download": {},  # sd<seed>: Hugging Face model repo with base/moe/latest.pt; the other seeds are trained here
    },
    "scorers": {"vig": {"num_samples": 32}, "flow": {"num_loss_samples": 512, "num_action_samples": 1}},
    "thresholds": {"num_demos": 50, "frame_stride": 4, "chunks": 4,
                   "percentiles": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]},
    "dagger": {
        "scorers": ["vig", "flow"],
        "thresholds": ["p90", "p95", "p99"],  # one collection each; take over above, hand back below the same value
        "variants": ["plain", "random_move"],
        "num_demos": 500,  # successful DAgger episodes per collection
        "take_count": 2,  # consecutive scores above the threshold before the expert takes over
        "return_count": 16,  # consecutive scores below the threshold before the policy takes back control
        "expert": "demo",  # demo: move like the planner demos | fast: the earlier, faster DAgger expert
        "random_move": {  # the random_move variant's move at every takeover (expert/random_move.py)
            "radius": 0.05,  # m: target position uniform in a ball of this radius around the gripper tip
            "rot_deg": 5.0,  # deg: orientation turned uniformly by 0 ... this about a random axis
            "max_tries": 10000,  # targets drawn until one is reachable and collision-free (else no move)
            "margin": 0.004,  # m: obstacles grown by this for the collision check
            "tol": 0.0005,  # m: how much deeper than at the start a point may get into a grown obstacle
        },
        "max_episode_steps": 400,
        "policy_chunk_steps": 8,
        "seed_start": 1_000_000,
        "workers_per_collection": "auto",  # auto: half the simulator slots
        "max_attempts": None,  # per collection; default 20 x num_demos
        "strategy_seed": 0,
        "atomic_grasp": False,
        "save_failed_trajectories": False,  # failed attempts keep their per-step record (attempts/), not the images
    },
}

POLICY_TEMPLATE_DIR = Path(__file__).parent / "configs"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "download":
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _set(config: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """The experiment config: defaults, then the YAML (or a saved config.json), then ``key=value`` overrides."""
    path = Path(path)
    user = json.loads(path.read_text()) if path.suffix == ".json" else (yaml.safe_load(path.read_text()) or {})
    config = _merge(DEFAULTS, user)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set(config, key.strip(), yaml.safe_load(raw))
    if not config.get("work_dir"):
        raise ValueError("the config needs work_dir (where every output of the experiment goes)")
    m, d = config["moe"], config["dagger"]
    if [int(f) for f in m["factors"]][0] != 1 or len(m["factors"]) < 2:
        raise ValueError("moe.factors must start with 1 (full resolution) and have degraded levels for VIG")
    seeds = [int(s) for s in m["seeds"]]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError(f"moe.seeds must be distinct seeds, got {m['seeds']}")
    for key in m["download"] or {}:
        if not re.fullmatch(r"sd\d+", str(key)) or int(str(key)[2:]) not in seeds:
            raise ValueError(f"moe.download keys are sd<seed> for seeds in moe.seeds, got {key!r}")
    for scorer in d["scorers"]:
        if scorer not in SCORERS:
            raise ValueError(f"unknown scorer {scorer!r}; scorers are {SCORERS}")
    for variant in d["variants"]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; variants are {VARIANTS}")
    for spec in d["thresholds"]:
        q = re.fullmatch(r"p(\d+)", str(spec))
        if not q or int(q.group(1)) not in [int(p) for p in config["thresholds"]["percentiles"]]:
            raise ValueError(f"dagger.thresholds entries are p<q> with q in thresholds.percentiles, got {spec!r}")
    config["work_dir"] = str(Path(config["work_dir"]).expanduser().resolve())
    return config


def board_mode(cfg: dict) -> bool:
    """Several machines share the work through a task board (``sharing.board``)."""
    return bool(cfg["sharing"]["board"])


def downloaded(cfg: dict, seed: int) -> str | None:
    """The Hugging Face model repo a MoE seed is downloaded from, or None when it is trained here."""
    return (cfg["moe"]["download"] or {}).get(f"sd{int(seed)}")


@dataclass(frozen=True)
class Collection:
    """One DAgger collection: the MoE seed that drives, the query scorer, the threshold and the variant."""

    seed: int
    scorer: str
    threshold: str  # "p<q>"
    variant: str

    @property
    def id(self) -> str:
        return f"sd{self.seed}_{self.scorer}_{self.threshold}_{self.variant}"

    @property
    def label(self) -> str:
        variant = "" if self.variant == "plain" else " + random move"
        return f"MoE seed {self.seed}, {'VIG' if self.scorer == 'vig' else 'flow loss'} at {self.threshold}{variant}"

    @classmethod
    def parse(cls, text: str) -> Collection:
        m = re.fullmatch(r"sd(\d+)_(vig|flow)_(p\d+)_(plain|random_move)", text)
        if not m:
            raise ValueError(f"not a collection id: {text!r} (e.g. sd0_vig_p90_plain)")
        return cls(int(m.group(1)), m.group(2), m.group(3), m.group(4))


def collections(cfg: dict) -> list[Collection]:
    """Every collection of the experiment, ordered by MoE seed, variant, scorer and threshold (config order)."""
    d = cfg["dagger"]
    return [Collection(int(seed), scorer, str(spec), variant) for seed in cfg["moe"]["seeds"] for variant in d["variants"]
            for scorer in d["scorers"] for spec in d["thresholds"]]


@dataclass(frozen=True)
class Paths:
    """Every output location of one experiment, under ``work_dir``."""

    root: Path

    @classmethod
    def of(cls, config: dict) -> Paths:
        return cls(Path(config["work_dir"]))

    def config(self) -> Path:
        return self.root / "config.json"

    # part 1: planner demos (ManiSkill trajectories) and the training dataset (robopolicy small-files format)
    def raw_demos(self) -> Path:
        return self.root / "data" / "raw" / "planner"

    def dataset(self, name: str) -> Path:
        return self.root / "data" / "datasets" / name

    # part 2: the MoEs and their thresholds
    def moe_dir(self, seed: int) -> Path:
        return self.root / "policies" / "moe" / f"sd{int(seed)}"

    def moe_checkpoint(self, seed: int) -> Path:
        return self.moe_dir(seed) / "latest.pt"

    def thresholds(self, seed: int) -> Path:
        return self.moe_dir(seed) / "thresholds.json"

    def demo_scores(self, seed: int) -> Path:
        return self.moe_dir(seed) / "demo_scores.npz"

    # part 3: DAgger collections and their metrics
    def dagger(self, collection: Collection | str) -> Path:
        return self.root / "dagger" / (collection if isinstance(collection, str) else collection.id)

    def results(self) -> Path:
        return self.root / "results"

    def collection_results(self, collection: Collection | str) -> Path:
        """The stem of a collection's metric files: ``<stem>.json``, ``<stem>_episodes.csv``, ..."""
        return self.results() / "collections" / (collection if isinstance(collection, str) else collection.id)

    def logs(self) -> Path:
        return self.root / "logs"
