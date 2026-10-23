"""Policy training: run configs for the robopolicy library, and the training worker.

Two kinds of policy, the same Transformer flow-matching recipe otherwise (frozen DINOv2-small per camera, attention
pooling, 8-layer Transformer action head of width 256, 16-step chunks, batch 64, lr 1e-4 cosine):

- **MoE** (``factors`` = [1, 2, ..., 64]): every training sample is degraded to a random level, and each head layer's
  feed-forward block is 8 experts (top-2) with one router per level.
- **Per-resolution** (``factors`` = [k]): one dense policy trained only on images degraded by k (k = 1 is the
  full-resolution policy).

Training length is ``training.epochs`` passes over the training demos: max_updates = epochs x ceil(training
samples / batch size), with a sample per step that has a full 16-step future.

Worker: ``python -m vig_dagger.train --config <work_dir>/config.json --run <name>`` with ``<name>`` one of

  retrain/<training set>/seed<s>       a dense full-resolution policy (every retrained, evaluated policy)
  drivers/moe                          the MoE that drives round ``moe`` (when not downloaded)
  drivers/per_resolution/res<k>        a dense policy on images degraded by k (round ``per_resolution``)

The first start writes ``<run>/run_config.json`` (kept, so a restarted run keeps its update count and W&B run);
later starts resume from the newest ``step-<N>.pt``. ``latest.pt`` marks a finished run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from vig_dagger import config as config_module
from vig_dagger.config import POLICY_TEMPLATE_DIR, Paths


def run_config(cfg: dict, *, dataset: Path, out_dir: Path, factors: list[int], seed: int, name: str,
               tags: list[str]) -> dict[str, Any]:
    """A robopolicy run config (JSON-ready) for one training of this experiment."""
    from robopolicy import RunConfig, build_dataset

    template = json.loads((POLICY_TEMPLATE_DIR / "policy.json").read_text())
    template.pop("_comment", None)
    t = cfg["training"]
    run = dict(template)
    run["seed"] = int(seed)
    run["dataset"] = dict(template["dataset"], root=str(dataset),
                          selection={"val_fraction": t["val_fraction"], "split_seed": 0})
    run["dataset"]["sampling"] = dict(template["dataset"]["sampling"], seed=42 + int(seed))
    run["loader"] = dict(template["loader"], batch_size=t["batch_size"], num_workers=cfg["resources"]["dataloader_workers"])
    policy = json.loads(json.dumps(template["policy"]))
    policy["conditioning"] = {"type": "indexed", "vocab_sizes": [len(factors)]}
    if len(factors) == 1:
        policy["head"].pop("moe")
    run["policy"] = policy
    run["processing"] = dict(template["processing"], resolution={"factors": [int(f) for f in factors]})
    run["training"] = dict(template["training"], device="cuda:0", max_updates=1, learning_rate=t["learning_rate"],
                           weight_decay=t["weight_decay"], warmup_updates=t["warmup_updates"],
                           min_lr_ratio=t["min_lr_ratio"], checkpoint_every=t["checkpoint_every"],
                           keep_checkpoints=t["keep_checkpoints"])
    w = cfg["wandb"]
    run["logging"] = {"out_dir": str(out_dir), "wandb": {
        "enabled": bool(w["enabled"]), "entity": w["entity"], "project": w["project"], "name": name,
        "group": cfg["name"], "tags": tags}}
    config = RunConfig.model_validate(run)
    train = build_dataset(config.dataset, config.features, "train")
    per_epoch = math.ceil(len(train) / config.loader.batch_size)
    run["training"]["max_updates"] = int(t["epochs"]) * per_epoch
    run["_info"] = dict(training_demos=len(train.selection), training_samples=len(train), updates_per_epoch=per_epoch,
                        epochs=int(t["epochs"]))
    train.close()
    return run


def finished(out_dir: Path) -> bool:
    return (out_dir / "latest.pt").is_file()


def resolve(cfg: dict, name: str) -> tuple[str, Path, list[int], int]:
    """(training set, run directory, resolution factors, seed) of a run name (see the module docstring)."""
    paths, parts = Paths.of(cfg), name.split("/")
    if parts[0] == "retrain" and len(parts) == 3 and parts[2].startswith("seed"):
        seed = int(parts[2][4:])
        return parts[1], paths.retrain_run(parts[1], seed), [1], seed
    if parts == ["drivers", "moe"]:
        d = cfg["drivers"]["moe"]
        return "base", paths.driver_dir("moe"), [int(f) for f in d["factors"]], int(d["seed"])
    if parts[:2] == ["drivers", "per_resolution"] and len(parts) == 3 and parts[2].startswith("res"):
        return "base", paths.driver_dir("per_resolution") / parts[2], [int(parts[2][3:])], int(cfg["drivers"]["per_resolution"]["seed"])
    raise ValueError(f"unknown run {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="the experiment config (<work_dir>/config.json)")
    p.add_argument("--run", required=True)
    args = p.parse_args()
    from robopolicy import RunConfig, Trainer

    cfg = config_module.load(args.config)
    dataset_name, out_dir, factors, seed = resolve(cfg, args.run)
    config_path = out_dir / "run_config.json"
    if not config_path.exists():
        kind = "moe" if len(factors) > 1 else f"res{factors[0]}"
        run = run_config(cfg, dataset=Paths.of(cfg).dataset(dataset_name), out_dir=out_dir, factors=factors, seed=seed,
                         name=f"{cfg['name']}/train/{args.run}", tags=["train", *args.run.split("/")[:-1], f"seed{seed}", kind, dataset_name])
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(run, indent=1))
    values = json.loads(config_path.read_text())
    info = values.pop("_info", {})
    config = RunConfig.model_validate(values)
    out_dir = Path(config.logging.out_dir)
    if finished(out_dir):
        print(f"already finished: {out_dir / 'latest.pt'}")
        return
    steps = sorted(out_dir.glob("step-*.pt"), key=lambda p: int(p.stem[5:]))
    if steps:
        print(f"resuming from {steps[-1]}", flush=True)
        trainer = Trainer.resume(steps[-1])
    else:
        print(f"training {config.logging.wandb.name}: {info}", flush=True)
        trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
