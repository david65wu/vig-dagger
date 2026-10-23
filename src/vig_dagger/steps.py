"""The pipeline's short steps, each a command the job queue runs: ``python -m vig_dagger.steps <step> --config ...``.

  download-demos              raw planner demos (and the extra ones) from hub.dataset_repo
  convert --which W           raw planner demos -> datasets/planner and datasets/base (W = planner),
                              or -> datasets/planner_extra (W = extra)
  planner-baseline            datasets/base_plus_planner = base + the first dagger.num_demos extra planner demos
  download-driver             the MoE driver and its thresholds from hub.model_repo
  thresholds --round R        merge the round's demo-frame scores into thresholds.json (percentiles)
  dagger-prepare --round R --scorer S   resolve the thresholds and write the collection's settings.json
  dagger-dataset --round R --scorer S   the first dagger.num_demos successful episodes -> datasets/dagger_R_S and
                              datasets/base_plus_R_S
  analyze --round R           DAgger collection metrics (results/, REPORT.md, W&B if enabled)
  summarize                   evaluation summary (results/, REPORT.md, W&B if enabled)
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vig_dagger import config as config_module
from vig_dagger import dataset
from vig_dagger.config import Paths, driver_spec


def raw_parts(raw: Path) -> list[Path]:
    return sorted(raw.glob("part_*.h5"))


def _convert_one(args: tuple[str, list[int] | None, str]) -> int:
    path, indices, out = args
    out_path = Path(out)
    if dataset.ready(out_path):
        return dataset.num_demos(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    n = dataset.convert([(Path(path), indices)], tmp)
    dataset.finalize(tmp, out_path)
    return n


def convert_sources(sources: list[tuple[Path, list[int] | None]], target: Path, workers: int,
                    first: list[tuple[Path, int | None]] | None = None) -> int:
    """Convert trajectory files in parallel (one piece per source) and combine them, after ``first`` datasets."""
    pieces = target.with_name(target.name + ".pieces")
    pieces.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), i, str(pieces / f"piece_{k:04d}")) for k, (p, i) in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_convert_one, jobs))
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    n = dataset.combine(list(first or []) + [(Path(out), None) for _, _, out in jobs], tmp)
    dataset.finalize(tmp, target)
    shutil.rmtree(pieces)
    return n


def convert(cfg: dict, which: str) -> None:
    paths = Paths.of(cfg)
    extra = which == "extra"
    name = "planner_extra" if extra else "planner"
    target = paths.dataset(name)
    if not dataset.ready(target):
        parts = raw_parts(paths.raw_demos(extra))
        if not parts:
            raise SystemExit(f"no raw demos in {paths.raw_demos(extra)}")
        n = convert_sources([(p, None) for p in parts], target, int(cfg["demos"]["convert_workers"]))
        print(f"[convert] {name}: {n} demos -> {target}")
    if not extra:
        base = paths.dataset("base")
        if not dataset.ready(base):
            need = int(cfg["demos"]["num_demos"])
            if dataset.num_demos(target) < need:
                raise SystemExit(f"{target} has {dataset.num_demos(target)} demos; demos.num_demos is {need}")
            tmp = base.with_name("base.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            dataset.combine([(target, need)], tmp)
            dataset.finalize(tmp, base)
            print(f"[convert] base: the first {need} demos of {target}")


def planner_baseline(cfg: dict) -> None:
    paths, n = Paths.of(cfg), int(cfg["dagger"]["num_demos"])
    target = paths.dataset("base_plus_planner")
    if dataset.ready(target):
        return
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    count = dataset.combine([(paths.dataset("base"), None), (paths.dataset("planner_extra"), n)], tmp)
    dataset.finalize(tmp, target)
    print(f"[planner-baseline] base_plus_planner: {count} demos")


def thresholds(cfg: dict, round_name: str) -> None:
    from vig_dagger.workers.score_demos import part_path

    paths, t = Paths.of(cfg), cfg["thresholds"]
    parts = [np.load(part_path(cfg, round_name, c)) for c in range(int(t["chunks"]))]
    parts = [p for p in parts if len(p["vig"])]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in ("vig", "vig_gap", "flow", "demo", "frame")}
    np.savez(paths.demo_scores(round_name), **merged, factors=parts[0]["factors"])
    table = {s: {f"p{q}": float(np.percentile(merged[s], q)) for q in t["percentiles"]} for s in ("vig", "flow")}
    table.update(frames=int(len(merged["vig"])), demos=int(t["num_demos"]), frame_stride=int(t["frame_stride"]))
    paths.thresholds(round_name).write_text(json.dumps(table, indent=1))
    shutil.rmtree(part_path(cfg, round_name, 0).parent)
    for scorer in ("vig", "flow"):
        print(f"[thresholds] {round_name} {scorer} on {table['frames']} demo frames: "
              + ", ".join(f"{q} {v:.6f}" for q, v in table[scorer].items()), flush=True)


def resolve_threshold(cfg: dict, round_name: str, scorer: str, value) -> float:
    """A number, or "p<q>": the q-th percentile of the scorer on the round's demo frames (thresholds.json)."""
    if isinstance(value, (int, float)):
        return float(value)
    table = json.loads(Paths.of(cfg).thresholds(round_name).read_text())
    if value not in table[scorer]:
        raise SystemExit(f"{value} of {scorer} is not in {Paths.of(cfg).thresholds(round_name)} (has "
                         f"{sorted(table[scorer])}); add it to thresholds.percentiles and recompute, or give a number")
    return float(table[scorer][value])


def dagger_settings(cfg: dict, round_name: str, scorer: str) -> dict:
    d = cfg["dagger"]
    target = int(d["num_demos"])
    return dict(
        round=round_name, scorer=scorer, policies=driver_spec(cfg, round_name),
        take_threshold=resolve_threshold(cfg, round_name, scorer, d["thresholds"][scorer]["take"]),
        return_threshold=resolve_threshold(cfg, round_name, scorer, d["thresholds"][scorer]["return"]),
        take_spec=d["thresholds"][scorer]["take"], return_spec=d["thresholds"][scorer]["return"],
        take_count=int(d["take_count"]), return_count=int(d["return_count"]), expert=d["expert"],
        atomic_grasp=bool(d["atomic_grasp"]), target=target, max_attempts=int(d["max_attempts"] or 20 * target),
        seed_start=int(d["seed_start"]), max_episode_steps=int(d["max_episode_steps"]),
        chunk_steps=int(d["policy_chunk_steps"]), strategy_seed=int(d["strategy_seed"]),
        scorer_settings=cfg["scorers"][scorer], save_failed=bool(d["save_failed_trajectories"]))


def dagger_prepare(cfg: dict, round_name: str, scorer: str) -> None:
    out = Paths.of(cfg).dagger(round_name, scorer)
    out.mkdir(parents=True, exist_ok=True)
    settings = dagger_settings(cfg, round_name, scorer)
    path = out / "settings.json"
    if path.exists():
        old = json.loads(path.read_text())
        changed = sorted(k for k in settings if k not in ("target", "max_attempts") and old.get(k) != settings[k])
        if changed:
            raise SystemExit(f"{out} was collected with different settings ({changed}); use a new work_dir, or delete "
                             "the collection to start it again")
    path.write_text(json.dumps(settings, indent=1))
    print(f"[dagger] {round_name}/{scorer}: take > {settings['take_threshold']:.6g} ({settings['take_spec']}) "
          f"x{settings['take_count']}, return < {settings['return_threshold']:.6g} ({settings['return_spec']}) "
          f"x{settings['return_count']}, expert {settings['expert']}, target {settings['target']}")


def dagger_sources(out: Path, target: int) -> list[tuple[Path, list[int]]]:
    """The first ``target`` successful episodes to finish, as (trajectory file, indices) in finishing order."""
    rows = []
    for path in (out / "episodes").glob("worker_*.jsonl"):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    used = sorted((r for r in rows if r["success"]), key=lambda r: r["finished_at"])[:target]
    if len(used) < target:
        raise SystemExit(f"{out} has {len(used)} successful episodes; dagger.num_demos is {target}")
    sources: list[tuple[Path, list[int]]] = []
    for r in used:
        path = out / r["file"]
        if sources and sources[-1][0] == path:
            sources[-1][1].append(int(r["traj"]))
        else:
            sources.append((path, [int(r["traj"])]))
    return sources


def dagger_dataset(cfg: dict, round_name: str, scorer: str) -> None:
    paths, n = Paths.of(cfg), int(cfg["dagger"]["num_demos"])
    root = paths.dataset(f"dagger_{round_name}_{scorer}")
    if not dataset.ready(root):
        count = convert_sources(dagger_sources(paths.dagger(round_name, scorer), n), root, int(cfg["demos"]["convert_workers"]))
        print(f"[dagger-dataset] dagger_{round_name}_{scorer}: {count} episodes")
    target = paths.dataset(f"base_plus_{round_name}_{scorer}")
    if not dataset.ready(target):
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        count = dataset.combine([(paths.dataset("base"), None), (root, n)], tmp)
        dataset.finalize(tmp, target)
        print(f"[dagger-dataset] base_plus_{round_name}_{scorer}: {count} demos")


def analyze(cfg: dict, round_name: str) -> None:
    from vig_dagger import analysis

    paths = Paths.of(cfg)
    results = {s: analysis.analyze(cfg, round_name, s, paths.dagger(round_name, s), paths.results())
               for s in cfg["dagger"]["scorers"]}
    if len(results) > 1:
        analysis.compare(cfg, round_name, results)
    from vig_dagger import report

    report.write(cfg)


def summarize(cfg: dict) -> None:
    from vig_dagger import evaluation

    paths = Paths.of(cfg)
    evaluation.summarize(cfg, paths.root / "eval", paths.results())
    from vig_dagger import report

    report.write(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--which", choices=["planner", "extra"], default="planner")
    p.add_argument("--round")
    p.add_argument("--scorer")
    args = p.parse_args()
    from vig_dagger.jobs import exit_on_sigterm

    exit_on_sigterm()
    cfg = config_module.load(args.config)
    from vig_dagger import hub

    step = args.step
    if step == "download-demos":
        hub.download(cfg, "demos")
    elif step == "convert":
        convert(cfg, args.which)
    elif step == "planner-baseline":
        planner_baseline(cfg)
    elif step == "download-driver":
        hub.download(cfg, "policy")
    elif step == "thresholds":
        thresholds(cfg, args.round)
    elif step == "dagger-prepare":
        dagger_prepare(cfg, args.round, args.scorer)
    elif step == "dagger-dataset":
        dagger_dataset(cfg, args.round, args.scorer)
    elif step == "analyze":
        analyze(cfg, args.round)
    elif step == "summarize":
        summarize(cfg)
    else:
        raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main()
