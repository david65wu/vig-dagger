"""The experiment configuration (a YAML file) and where every output of an experiment goes.

Values missing from the YAML file take the defaults below; ``--set key.sub=value`` overrides one value on the
command line (the value is parsed as YAML, so ``--set dagger.num_demos=200`` gives an int).

An experiment runs one or more **rounds** of DAgger. A round fixes the policy that drives the DAgger rollouts and
where VIG's degraded-level samples come from:

- ``moe``: a 7-level resolution mixture-of-experts policy (trained on the base demos) drives at full resolution and
  gives both scores (VIG over its own levels, flow loss at full resolution).
- ``per_resolution``: a dense full-resolution policy drives and gives the flow loss; VIG compares it with dense
  policies trained on images degraded by 2 ... 64. Its full-resolution policy is the ``base`` retraining of seed
  ``retrain.seeds[0]`` (the same data, recipe and seed), so only the degraded-level policies are trained for it.

Every retrained (evaluated) policy is a dense full-resolution policy. The training sets ``base`` and
``base_plus_planner`` are shared by all rounds; each round adds ``base_plus_<round>_<scorer>`` per scorer.

With ``sharing.mode: hub`` the experiment runs on any number of machines (``board.py``): each machine claims tasks
(a training, a round's thresholds, a DAgger collection, an evaluation) through a public Hugging Face repo, downloads
their inputs from it and uploads their outputs to it; both rounds run at the same time.

The task is OpenDrawer (Isaac Lab): simulator jobs (demo collection, DAgger, evaluation) run in the Isaac environment
(``isaac.python``, installed by ``scripts/install_isaac.sh``); everything else runs in the main environment.
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
    "name": "opendrawer",
    "work_dir": None,
    "resources": {
        "gpus": "auto",  # CUDA indices, or "auto": every GPU nvidia-smi lists
        "sim_slots": "auto",  # CPU budget in simulator-job units (an Isaac job uses about 1.5 cores); auto: cores // 3
        "max_load": "auto",  # start nothing new while the 1-minute load average is above this; auto: CPU cores
        "sim_gpu_memory_gb": 6,  # free GPU memory a simulator job needs to start (measured: 4.7 GB, renderer + MoE)
        "train_gpu_memory_gb": 8,  # free GPU memory a training needs to start (measured: 2.6-2.9 GB; one of ours per GPU)
        "sim_ram_gb": 8,  # measured: 6.8 GB for an Isaac job
        "sim_start_gap_s": 20,  # seconds between two simulator-job starts (Isaac Sim can crash when several start at once)
        "train_ram_gb": 24,  # measured: ~19 GB with 16 data-loading workers
        "train_slots": 2,  # CPU budget a training's data loading takes, in simulator-job units (measured: ~1.3 cores)
        "dataloader_workers": 16,
    },
    # Optional: every number and figure is always written to <work_dir>/results (REPORT.md, JSON, CSV, PNG).
    "wandb": {"enabled": False, "entity": "vig-dagger-exp", "project": "opendrawer"},
    "hub": {"dataset_repo": None, "model_repo": None, "revision": "main"},
    # Several machines: every machine runs the same config; tasks, their claims and their outputs go through one
    # public Hugging Face repo (board.py). none: one machine runs everything, without Hugging Face.
    "sharing": {"mode": "none", "repo": None, "public": True, "poll_s": 60, "heartbeat_s": 1800,
                "upload_trajectories": True},
    "isaac": {"python": "auto"},  # the Isaac environment's interpreter; auto: <repo>/.venv-isaac/bin/python
    "demos": {
        "source": "collect",  # collect | download (hub.dataset_repo)
        "num_demos": 100,  # planner demos in the original training set ("base")
        "demos_per_part": 25,
        "seed_start": 2_000_000,  # collect: part k collects from scene seed seed_start + 1000 * k
        "extra_seed_start": 3_000_000,  # collect: the extra planner demos of the base_plus_planner baseline
        "strategy_seed": 1,  # collect: planner strategy stream; part k uses strategy_seed + k
        "max_episode_steps": 1200,
        "convert_workers": 8,
    },
    "training": {
        "epochs": 150,  # every training of the task runs this many passes over its training demos
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
    "driver_barrier": False,  # train all drivers before any collection, without serializing rounds
    "rounds": ["moe", "per_resolution"],  # run in this order; a round starts after the previous round's retraining
    "drivers": {
        "moe": {"source": "train", "factors": [1, 2, 4, 8, 16, 32, 64], "seed": 0},  # train | download
        "per_resolution": {"factors": [1, 2, 4, 8, 16, 32, 64], "seed": 0},
    },
    "scorers": {"vig": {"num_samples": 32}, "flow": {"num_loss_samples": 512, "num_action_samples": 1}},
    "thresholds": {"num_demos": 50, "frame_stride": 4, "chunks": 4,
                   "percentiles": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]},
    "dagger": {
        "scorers": ["vig", "flow"],
        "num_demos": 100,  # n: successful DAgger episodes collected per scorer and round
        "thresholds": {"vig": {"take": "p95", "return": "p50"}, "flow": {"take": "p95", "return": "p50"}},
        "take_count": 2,  # consecutive scores above the take threshold before the expert takes over
        "return_count": 16,  # consecutive scores below the return threshold before the policy takes back control
        "expert": "demo",  # demo: move like the planner demos | fast: the earlier, faster DAgger expert
        "max_episode_steps": 1200,
        "policy_chunk_steps": 8,
        "seed_start": 1_000_000,
        "workers_per_scorer": "auto",  # auto: half the simulator slots per scorer
        "max_attempts": None,  # per scorer; default 20 x num_demos
        "strategy_seed": 0,
    },
    "retrain": {"seeds": [0, 1, 2], "planner_baseline": True},
    "evaluation": {
        "seed_start": 60_000,
        "num_episodes": 200,
        "max_episode_steps": 1200,
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
    if config["sharing"]["mode"] not in ("none", "hub"):
        raise ValueError(f"sharing.mode must be none or hub, got {config['sharing']['mode']!r}")
    if config["sharing"]["mode"] == "hub" and not config["sharing"]["repo"]:
        raise ValueError("sharing.mode hub needs sharing.repo (the Hugging Face repo the machines share)")
    for scorer in config["dagger"]["scorers"]:
        if scorer_type(config, scorer) not in SCORERS:
            raise ValueError(f"unknown scorer {scorer!r}; scorers are {SCORERS}")
    config["work_dir"] = str(Path(config["work_dir"]).expanduser().resolve())
    return config


def scorer_type(cfg: dict, name: str) -> str:
    """A collection variant can reuse a scorer with its own thresholds and output paths."""
    return cfg["dagger"].get("scorer_types", {}).get(name, name)


def isaac_python(cfg: dict) -> str:
    """The interpreter simulator jobs run with (the Isaac environment)."""
    value = cfg["isaac"]["python"]
    if value == "auto":
        return str(Path(__file__).resolve().parents[2] / ".venv-isaac" / "bin" / "python")
    return str(Path(value).expanduser())


def hub_mode(cfg: dict) -> bool:
    """True when several machines share the experiment through Hugging Face (``sharing.mode: hub``)."""
    return cfg["sharing"]["mode"] == "hub"


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

    # part 1: planner demos (Isaac Lab demo files) and training datasets (robopolicy small-files format)
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
