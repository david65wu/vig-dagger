"""Prepare, detach, monitor, and resume the six toy-reaching policy runs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
FACTORS = [1, 2, 4, 8, 16, 32, 64]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def dataset_view(source: Path, destination: Path, count: int):
    """Keep training arrays and hard-link camera videos; preserve the raw collection."""
    import h5py

    destination.mkdir(parents=True, exist_ok=False)
    task = destination / "toy_reach"
    task.mkdir()
    manifest = json.loads((source / "toy_reach/demo_manifest.json").read_text())
    manifest["shards"] = manifest["shards"][:count]
    if len(manifest["shards"]) != count:
        raise ValueError(f"Expected {count} demos in {source}")
    for shard in manifest["shards"]:
        target = task / shard["hdf5"]
        target.parent.mkdir(exist_ok=True)
        with h5py.File(source / "toy_reach" / shard["hdf5"]) as original, h5py.File(target, "x") as out:
            for demo in shard["demos"]:
                key = demo["demo_key"]
                old = original[f"data/{key}"]
                new = out.create_group(f"data/{key}")
                new.attrs.update(dict(old.attrs))
                old.copy("actions", new)
                obs = new.create_group("obs")
                for feature in ("state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                    old["obs"].copy(feature, obs)
                if 'goal_xyz' in old['obs']:
                    old['obs'].copy('goal_xyz', obs)
        for files in shard["videos"].values():
            for relative in files.values():
                target = task / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source / "toy_reach" / relative, target)
    manifest["source_dataset"] = str(source)
    write_json(task / "demo_manifest.json", manifest)
    dataset = json.loads((source / "dataset_manifest.json").read_text())
    dataset["num_demos"] = count
    write_json(destination / "dataset_manifest.json", dataset)


def prepare(args):
    from robopolicy import RunConfig, build_dataset

    root = args.root.resolve()
    if (root / "experiment.json").exists():
        raise FileExistsError(root / "experiment.json")
    task_variant = getattr(args, 'task', 'line')
    for source in (args.source, args.validation):
        manifest = json.loads((source/'toy_reach/demo_manifest.json').read_text())
        if manifest.get('task_variant', 'line') != task_variant:
            raise ValueError(f'Dataset variant does not match requested task: {source}')
    dataset_view(args.source.resolve(), root / "datasets/train_5", 5)
    dataset_view(args.source.resolve(), root / "datasets/train_20", 20)
    dataset_view(args.validation.resolve(), root / "datasets/validation", 5)
    template = json.loads((REPO / "src/vig_dagger/configs/policy.json").read_text())
    template.pop("_comment", None)
    runs = []
    for gpu, (count, variant) in enumerate((n, v) for n in (5, 20) for v in ("dense_full", "moe_7level", "dense_7level")):
        name = f"demos{count}_{variant}"
        out = root / "policies" / name
        out.mkdir(parents=True)
        config = copy.deepcopy(template)
        config["seed"] = 0
        config["features"]["features"][2]["cameras"] = ["third_person", "wrist"]
        if task_variant in ('sphere', 'fixed_goal_sphere'):
            config['features']['features'][1]['shape'] = [12]
            config['features']['features'][1]['paths'].append('obs/goal_xyz')
            config['policy']['state_dim'] = 12
        config["dataset"].update(root=str(root / f"datasets/train_{count}"), selection={"val_fraction": 0.0})
        config["dataset"]["sampling"].update(k_passes=1, seed=42, window_shards=20)
        config["loader"].update(batch_size=64, num_workers=2, persistent_workers=True)
        levels = [1] if variant == "dense_full" else FACTORS
        config["processing"]["resolution"] = {"factors": levels}
        config["policy"]["conditioning"] = {"type": "indexed", "vocab_sizes": [len(levels)]}
        if variant != "moe_7level":
            config["policy"]["head"].pop("moe")
        config["training"].update(
            device="cuda:0", max_updates=1, learning_rate=1e-4, weight_decay=1e-4,
            warmup_updates=100, min_lr_ratio=1e-4, log_every=50, validate_every=100,
            validation_batches=None, checkpoint_every=250, keep_checkpoints=3,
        )
        config["logging"] = {"out_dir": str(out), "wandb": {"enabled": False}}
        validated = RunConfig.model_validate(config)
        data = build_dataset(validated.dataset, validated.features, "train")
        # IterableDataset emits a partial batch per worker. Account for both tails.
        lengths = [max(0, e.length - validated.features.max_future) for e in data.episodes]
        per_epoch = sum(math.ceil(sum(lengths[w::2]) / 64) for w in range(2))
        if len(data.selection) != count:
            raise ValueError("Training selection changed")
        config["training"].update(max_updates=100 * per_epoch, validate_every=5 * per_epoch)
        spec = dict(name=name, gpu=gpu, num_demos=count, variant=variant, epochs=100,
                    task_variant=task_variant,
                    updates_per_epoch=per_epoch, total_updates=100 * per_epoch,
                    training_samples=len(data), selection=[list(x) for x in data.selection],
                    dataset_fingerprint=data.fingerprint,
                    validation_root=str(root / "datasets/validation"), run_dir=str(out),
                    checkpoints={str(e * per_epoch): f"epoch-{e:03d}.pt" for e in (25, 50, 100)})
        if count == 20:
            matched_updates = runs[0]['total_updates']
            spec["checkpoints"][str(matched_updates)] = f"matched-{matched_updates}.pt"
        data.close()
        write_json(out / "run_config.json", RunConfig.model_validate(config).model_dump(mode="json"))
        write_json(out / "experiment.json", spec)
        runs.append(spec)
    assert runs[0]["selection"] == runs[1]["selection"] == runs[2]["selection"]
    assert runs[3]["selection"] == runs[4]["selection"] == runs[5]["selection"]
    assert set(map(tuple, runs[0]["selection"])) <= set(map(tuple, runs[3]["selection"]))
    write_json(root / "experiment.json", dict(
        created_at=time.time(), source=str(args.source.resolve()),
        validation_source=str(args.validation.resolve()), runs=runs,
        task_variant=task_variant,
        resolution_conditioning="level index supplied to both dense and MoE seven-level models",
        normalization="per-training-set minmax, unchanged robopolicy recipe",
        validation="five independent planner demos, seeds 1000 through 1004",
    ))
    print(json.dumps([{k: r[k] for k in ("name", "gpu", "training_samples", "total_updates")} for r in runs], indent=2))


def worker(args):
    import torch
    from robopolicy import RunConfig, Trainer, build_dataset
    from robopolicy.datasets.config import DatasetConfig

    out = args.run_dir.resolve()
    spec = json.loads((out / "experiment.json").read_text())
    config = RunConfig.model_validate_json((out / "run_config.json").read_text())
    torch.set_num_threads(2)

    class ToyTrainer(Trainer):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            self.validation_dataset.close()
            val = self.config.dataset.model_dump(mode="json")
            val.update(root=spec["validation_root"], selection={"val_fraction": 0.0})
            self.validation_dataset = build_dataset(DatasetConfig.model_validate(val), self.config.features, "all")
            self.validation_dataset.split = "val"  # fixed sequential validation traversal
            if len(self.validation_dataset.selection) != 5:
                raise ValueError("Expected exactly five independent validation demonstrations")
            self.logger.log({"event": "validation_provenance", "root": spec["validation_root"],
                             "selection": self.validation_dataset.selection,
                             "fingerprint": self.validation_dataset.fingerprint})
            self.segment_started = self.last_report = time.time()
            self.segment_step = self.last_report_step = self.completed_updates
            self.report_status("running")

        def report_status(self, state, loss=None):
            now = time.time()
            elapsed = now - self.segment_started
            rate = (self.completed_updates - self.segment_step) / max(elapsed, 1e-6)
            recent = (self.completed_updates - self.last_report_step) / max(now - self.last_report, 1e-6)
            result = dict(name=spec["name"], state=state, pid=os.getpid(), gpu=spec["gpu"],
                          updated_at=now, step=self.completed_updates, total_updates=spec["total_updates"],
                          epochs=spec.get("initial_epochs", 0) + self.completed_updates / spec["updates_per_epoch"], loss=loss,
                          updates_per_second=rate, recent_updates_per_second=recent,
                          eta_seconds=(spec["total_updates"] - self.completed_updates) / rate if rate > 0 else None)
            write_json(out / "status.json", result)
            self.last_report, self.last_report_step = now, self.completed_updates
            print(json.dumps(result), flush=True)

        def update(self, batch):
            loss, updated = super().update(batch)
            if updated:
                checkpoint = spec["checkpoints"].get(str(self.completed_updates))
                if checkpoint is not None:
                    self.save(out / checkpoint)
                if self.completed_updates % 50 == 0:
                    self.report_status("running", loss)
            return loss, updated

    try:
        write_json(out / "status.json", dict(state="starting", pid=os.getpid(), gpu=spec["gpu"], updated_at=time.time()))
        latest = out / "latest.pt"
        if latest.exists():
            write_json(out / "status.json", dict(state="complete", step=spec["total_updates"], updated_at=time.time()))
            return
        candidates = [(int(p.stem.split("-")[-1]), p) for p in out.glob("step-*.pt")]
        candidates += [(int(step), out / name) for step, name in spec["checkpoints"].items() if (out / name).exists()]
        if candidates:
            trainer = ToyTrainer.resume(max(candidates)[1])
        elif spec.get("initialize_from"):
            from robopolicy.checkpoints.serialization import read_checkpoint
            trainer = ToyTrainer.finetune(config, spec["initialize_from"])
            previous = read_checkpoint(spec["initialize_from"])
            # A continuation must retain the exact input/output interpretation.
            assert trainer.dataset.fingerprint == previous["dataset_fingerprint"]
            assert [list(x) for x in trainer.dataset.selection] == previous["selection"]
            assert trainer.packer.normalizer.statistics.model_dump(mode="json") == previous["statistics"]
            assert all(torch.equal(v.cpu(), previous["model"][k].cpu())
                       for k, v in trainer.policy.state_dict().items())
            trainer.data_epoch = spec["initial_epochs"]
            write_json(out / "initialization_check.json", dict(
                checkpoint=spec["initialize_from"], weights_equal=True,
                statistics_equal=True, selection_equal=True, fresh_optimizer=True,
                initial_lr=trainer.optimizer.param_groups[0]["lr"], data_epoch=trainer.data_epoch))
            del previous
        else:
            trainer = ToyTrainer(config)
        trainer.fit()
        trainer.report_status("complete")
    except BaseException as exc:
        write_json(out / "status.json", dict(state="failed", pid=os.getpid(), gpu=spec["gpu"],
                                            updated_at=time.time(), error=repr(exc)))
        raise


def active(pid, run_dir):
    import psutil
    if pid < 1:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and str(run_dir) in proc.cmdline()
    except psutil.Error:
        return False


def launch(args):
    import fcntl

    root = args.root.resolve()
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runs = json.loads((root / "experiment.json").read_text())["runs"]
        registry = root / "processes.json"
        processes = json.loads(registry.read_text()) if registry.exists() else {}
        (root / "logs").mkdir(exist_ok=True)
        for spec in runs:
            name, out = spec["name"], Path(spec["run_dir"])
            previous = processes.get(name, {})
            if (out / "latest.pt").exists() or active(previous.get("pid", -1), out):
                continue
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(spec["gpu"]), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                       HF_HOME="/cpfs02/user/caojianuo/.cache/huggingface",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
            with (root / "logs" / f"{name}.log").open("ab") as log:
                child = subprocess.Popen(
                    [sys.executable, "-m", "vig_dagger.toy_training", "worker", "--run-dir", str(out)],
                    cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes[name] = dict(pid=child.pid, gpu=spec["gpu"], started_at=time.time())
            write_json(registry, processes)
            print(f"Launched {name}: PID {child.pid}, GPU {spec['gpu']}", flush=True)


def status(args):
    root = args.root.resolve()
    processes = json.loads((root / "processes.json").read_text())
    for run in json.loads((root / "experiment.json").read_text())["runs"]:
        out = Path(run["run_dir"])
        state = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
        running = active(processes.get(run["name"], {}).get("pid", -1), out)
        eta = state.get("eta_seconds")
        print(f"GPU {run['gpu']} {run['name']:22s} {state.get('state', 'unknown'):9s} "
              f"alive={running} step={state.get('step', 0)}/{run['total_updates']} "
              f"rate={state.get('updates_per_second', 0):.2f}/s "
              f"ETA={eta / 60:.1f} min" if eta is not None else
              f"GPU {run['gpu']} {run['name']:22s} {state.get('state', 'unknown')} alive={running} {state.get('error', '')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument('--task', choices=['line', 'sphere', 'fixed_goal_sphere'], default='line')
    p = sub.add_parser("worker")
    p.add_argument("--run-dir", type=Path, required=True)
    for command in ("launch", "status"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    dict(prepare=prepare, worker=worker, launch=launch, status=status)[args.command](args)


if __name__ == "__main__":
    main()
