"""The experiment configuration (a YAML file) and where every output of an experiment goes.

Values missing from the YAML file take the defaults below; ``--set key.sub=value`` overrides one value on the
command line (the value is parsed as YAML, so ``--set dagger.num_demos=200`` gives an int).

An experiment runs one or more **rounds** of DAgger. A round fixes the policy that drives the DAgger rollouts and
where VIG's degraded-level samples come from:

- ``moe``: a 7-level resolution mixture-of-experts policy drives at full resolution and gives both scores (VIG over
  its own levels, flow loss at full resolution).
- ``per_resolution``: a dense full-resolution policy drives and gives the flow loss; VIG compares it with dense
  policies trained on images degraded by 2 ... 64. Its full-resolution policy is the ``base`` retraining of seed
  ``retrain.seeds[0]`` (the same data, recipe and seed), so only the degraded-level policies are trained for it.

Every retrained (evaluated) policy is a dense full-resolution policy. The training sets ``base`` and
``base_plus_planner`` are shared by all rounds; each round adds ``base_plus_<round>_<scorer>`` per scorer.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROUNDS = ("moe", "per_resolution")
SCORERS = ("vig", "flow")

DEFAULTS: dict[str, Any] = {
    "name": "plugcharger",
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
    "hub": {"dataset_repo": None, "model_repo": None, "revision": "main"},
    "demos": {
        "source": "download",  # download (hub.dataset_repo) | collect
        "num_demos": 1000,  # planner demos in the original training set ("base")
        "demos_per_part": 50,
        "seed_start": 2_000_000,  # collect: part k collects from scene seed seed_start + 1000 * k
        "extra_seed_start": 3_000_000,  # collect: the extra planner demos of the base_plus_planner baseline
        "strategy_seed": 1,  # collect: planner strategy stream; part k uses strategy_seed + k
        "max_episode_steps": 300,
        "convert_workers": 8,
    },
    "training": {
        "epochs": 60,  # every training of the task runs this many passes over its training demos
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
    "rounds": ["moe", "per_resolution"],  # run in this order; a round starts after the previous round's retraining
    "drivers": {
        "moe": {"source": "download", "factors": [1, 2, 4, 8, 16, 32, 64], "seed": 0},  # download | train
        "per_resolution": {"factors": [1, 2, 4, 8, 16, 32, 64], "seed": 0},
    },
    "scorers": {"vig": {"num_samples": 32}, "flow": {"num_loss_samples": 512, "num_action_samples": 1}},
    "thresholds": {"num_demos": 50, "frame_stride": 4, "chunks": 4,
                   "percentiles": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]},
    "dagger": {
        "scorers": ["vig", "flow"],
        "num_demos": 500,  # n: successful DAgger episodes collected per scorer and round
        "thresholds": {"vig": {"take": "p95", "return": "p50"}, "flow": {"take": "p95", "return": "p50"}},
        "take_count": 2,  # consecutive scores above the take threshold before the expert takes over
        "return_count": 16,  # consecutive scores below the return threshold before the policy takes back control
        "expert": "demo",  # demo: move like the planner demos | fast: the earlier, faster DAgger expert
        "max_episode_steps": 400,
        "policy_chunk_steps": 8,
        "seed_start": 1_000_000,
        "workers_per_scorer": "auto",  # auto: half the simulator slots per scorer
        "max_attempts": None,  # per scorer; default 20 x num_demos
        "strategy_seed": 0,
        "atomic_grasp": False,
        "save_failed_trajectories": False,
    },
    "retrain": {"seeds": [0, 1, 2], "planner_baseline": True},
    "evaluation": {
        "seed_start": 60_000,
        "num_episodes": 200,
        "max_episode_steps": 400,
        "actions_per_step": 8,
        "chunk": 25,  # episodes per evaluation job
        "evaluate_drivers": True,  # also evaluate the driving policies: the MoE, and round 2's res2 ... res64 (each at
        # its own level; its full-resolution policy is base/seed0, evaluated as a retrained policy)
        "driver_all_levels": True,  # evaluate the MoE at every level, not only full resolution
    },
}

POLICY_TEMPLATE_DIR = Path(__file__).parent / "configs"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
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
    for name in config["rounds"]:
        if name not in ROUNDS:
            raise ValueError(f"unknown round {name!r}; rounds are {ROUNDS}")
        if config["drivers"][name]["factors"][0] != 1:
            raise ValueError(f"drivers.{name}.factors must start with 1 (full resolution)")
    for scorer in config["dagger"]["scorers"]:
        if scorer not in SCORERS:
            raise ValueError(f"unknown scorer {scorer!r}; scorers are {SCORERS}")
    config["work_dir"] = str(Path(config["work_dir"]).expanduser().resolve())
    return config


def training_sets(cfg: dict) -> list[str]:
    """Every retraining set: base, base_plus_planner, then base_plus_<round>_<scorer> per round and scorer."""
    names = ["base"] + (["base_plus_planner"] if cfg["retrain"]["planner_baseline"] else [])
    return names + [f"base_plus_{r}_{s}" for r in cfg["rounds"] for s in cfg["dagger"]["scorers"]]


@dataclass(frozen=True)
class Paths:
    """Every output location of one experiment, under ``work_dir``."""

    root: Path

    @classmethod
    def of(cls, config: dict) -> Paths:
        return cls(Path(config["work_dir"]))

    def config(self) -> Path:
        return self.root / "config.json"

    # part 1: planner demos (ManiSkill trajectories) and training datasets (robopolicy small-files format)
    def raw_demos(self, extra: bool = False) -> Path:
        return self.root / "data" / "raw" / ("planner_extra" if extra else "planner")

    def dataset(self, name: str) -> Path:
        return self.root / "data" / "datasets" / name

    # part 2: the policies that drive DAgger, and their thresholds
    def driver_dir(self, round_name: str) -> Path:
        return self.root / "policies" / "drivers" / round_name

    def thresholds(self, round_name: str) -> Path:
        return self.driver_dir(round_name) / "thresholds.json"

    def demo_scores(self, round_name: str) -> Path:
        return self.driver_dir(round_name) / "demo_scores.npz"

    # part 3: DAgger collections
    def dagger(self, round_name: str, scorer: str) -> Path:
        return self.root / "dagger" / round_name / scorer

    # part 4: retrained policies and evaluations
    def retrain_run(self, dataset: str, seed: int) -> Path:
        return self.root / "policies" / "retrain" / dataset / f"seed{seed}"

    def evaluation(self, label: str) -> Path:
        return self.root / "eval" / label

    def logs(self) -> Path:
        return self.root / "logs"

    def results(self) -> Path:
        return self.root / "results"


def driver_runs(cfg: dict, round_name: str) -> dict[int, Path]:
    """{factor: run directory} of the policies that drive and score a round's DAgger."""
    paths, d = Paths.of(cfg), cfg["drivers"][round_name]
    if round_name == "moe":
        return {1: paths.driver_dir("moe")}
    runs = {int(f): paths.driver_dir(round_name) / f"res{f}" for f in d["factors"] if int(f) != 1}
    runs[1] = paths.retrain_run("base", int(cfg["retrain"]["seeds"][0]))
    return dict(sorted(runs.items()))


def driver_spec(cfg: dict, round_name: str):
    """What ``PolicySet.load`` takes for a round: the MoE checkpoint, or {factor: checkpoint}."""
    runs = driver_runs(cfg, round_name)
    if round_name == "moe":
        return str(runs[1] / "latest.pt")
    return {int(f): str(run / "latest.pt") for f, run in runs.items()}
