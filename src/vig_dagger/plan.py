"""The whole experiment as one graph of jobs (``vig-dagger run`` hands it to the queue in ``jobs.py``).

  part 1   download_demos (or collect_*)  -> convert_planner -> base dataset
           convert_extra -> planner_baseline (base + n extra planner demos)
  shared   train/retrain/base/seed<s>, train/retrain/base_plus_planner/seed<s>  (start as soon as their data exists,
           e.g. while round 1 collects DAgger demos)
  round r  driver (download_driver, or train/drivers/...) -> score_<r>_<chunk> -> thresholds_<r>
           -> dagger_prepare_<r>_<scorer> -> dagger_<r>_<scorer>_<worker> -> dagger_dataset_<r>_<scorer>
           -> train/retrain/base_plus_<r>_<scorer>/seed<s>;  analyze_<r> after the collections
           A round's driver starts after every retraining of the rounds before it.
  part 4   eval/<policy>/level<f>/<first seed> after each training -> summarize

Lower priority numbers start first when several jobs are ready: round 1 before round 2, DAgger before evaluation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from vig_dagger import dataset
from vig_dagger.config import Paths, driver_runs, training_sets
from vig_dagger.jobs import Job
from vig_dagger.steps import dagger_settings, raw_parts
from vig_dagger.train import resolve
from vig_dagger.workers.dagger import count_successes
from vig_dagger.workers.score_demos import part_path


def sim_slots(cfg: dict) -> int:
    value = cfg["resources"]["sim_slots"]
    return (os.cpu_count() or 8) // 2 if value == "auto" else int(value)


def settings_current(cfg: dict, round_name: str, scorer: str) -> bool:
    """The collection's settings.json exists and matches the current config (thresholds resolved)."""
    path = Paths.of(cfg).dagger(round_name, scorer) / "settings.json"
    try:
        return json.loads(path.read_text()) == json.loads(json.dumps(dagger_settings(cfg, round_name, scorer)))
    except (OSError, SystemExit, KeyError, ValueError):
        return False


def _eval_done(out: Path, count: int):
    return lambda: out.exists() and len(out.read_text().splitlines()) >= count


def build(cfg: dict, config_path: Path) -> list[Job]:
    paths = Paths.of(cfg)
    c = ["--config", str(config_path)]
    d, dg, ev = cfg["demos"], cfg["dagger"], cfg["evaluation"]
    n = int(dg["num_demos"])
    seeds = [int(s) for s in cfg["retrain"]["seeds"]]
    planner_baseline = bool(cfg["retrain"]["planner_baseline"])
    convert_slots = max(1, min(int(d["convert_workers"]), sim_slots(cfg)))
    jobs: dict[str, Job] = {}

    def add(job: Job) -> str:
        jobs.setdefault(job.name, job)
        return job.name

    def train(run: str, deps: list[str], priority: int) -> str:
        out = resolve(cfg, run)[1]
        return add(Job(f"train/{run}", ["vig_dagger.train", *c, "--run", run], kind="train", deps=deps, priority=priority,
                       retries=int(cfg["training"]["max_restarts"]), done=lambda out=out: (out / "latest.pt").is_file()))

    # ---- part 1: demos
    if d["source"] == "download":
        demo_deps = extra_deps = [add(Job("download_demos", ["vig_dagger.steps", "download-demos", *c], kind="cpu", priority=0,
                                          retries=2, done=lambda: bool(raw_parts(paths.raw_demos()))
                                          and (not planner_baseline or bool(raw_parts(paths.raw_demos(True))))))]
    else:
        per = int(d["demos_per_part"])
        demo_deps, extra_deps = [], []
        for extra, total, seed0, names in ((False, int(d["num_demos"]), d["seed_start"], demo_deps),
                                           (True, n if planner_baseline else 0, d["extra_seed_start"], extra_deps)):
            raw = paths.raw_demos(extra)
            for k in range(math.ceil(total / per)):
                count = min(per, total - k * per)
                names.append(add(Job(
                    f"collect_{'extra' if extra else 'planner'}_{k:03d}",
                    ["vig_dagger.workers.collect", "--out", str(raw), "--part", str(k), "--num-demos", str(count),
                     "--start-seed", str(seed0 + 1000 * k), "--strategy-seed", str(d["strategy_seed"] + k + (10_000 if extra else 0)),
                     "--max-episode-steps", str(d["max_episode_steps"])],
                    kind="sim", priority=0, retries=1, done=lambda p=raw / f"part_{k:03d}.done.json": p.is_file())))
    convert_planner = add(Job("convert_planner", ["vig_dagger.steps", "convert", "--which", "planner", *c], kind="cpu",
                              deps=demo_deps, priority=1, slots=convert_slots, retries=1,
                              done=lambda: dataset.ready(paths.dataset("planner")) and dataset.ready(paths.dataset("base"))))
    data_for = {"base": convert_planner}
    if planner_baseline:
        convert_extra = add(Job("convert_extra", ["vig_dagger.steps", "convert", "--which", "extra", *c], kind="cpu",
                                deps=extra_deps, priority=1, slots=convert_slots, retries=1,
                                done=lambda: dataset.ready(paths.dataset("planner_extra"))))
        data_for["base_plus_planner"] = add(Job("planner_baseline", ["vig_dagger.steps", "planner-baseline", *c], kind="cpu",
                                                deps=[convert_planner, convert_extra], priority=1,
                                                done=lambda: dataset.ready(paths.dataset("base_plus_planner"))))

    # ---- shared retraining (no DAgger needed)
    shared = [train(f"retrain/{name}/seed{s}", [data_for[name]], 30) for name in data_for for s in seeds]

    # ---- rounds
    before: list[str] = []  # every retraining of the earlier rounds
    for idx, r in enumerate(cfg["rounds"]):
        base_priority = 10 + 20 * idx
        if r == "moe":
            if cfg["drivers"]["moe"]["source"] == "download":
                driver = [add(Job("download_driver", ["vig_dagger.steps", "download-driver", *c], kind="cpu", deps=list(before),
                                  priority=base_priority, retries=2,
                                  done=lambda: (paths.driver_dir("moe") / "latest.pt").exists()))]
            else:
                driver = [train("drivers/moe", [convert_planner, *before], base_priority)]
        else:
            driver = [train(f"drivers/per_resolution/res{f}", [convert_planner, *before], base_priority)
                      for f in driver_runs(cfg, r) if f != 1]
            driver.append(train(f"retrain/base/seed{seeds[0]}", [convert_planner], 30))
        chunks = int(cfg["thresholds"]["chunks"])
        thr_done = lambda r=r: paths.thresholds(r).is_file()
        scores = [add(Job(f"score_{r}_{k}", ["vig_dagger.workers.score_demos", *c, "--round", r, "--chunk", str(k), "--chunks", str(chunks)],
                          kind="sim", deps=[*driver, *demo_deps], priority=base_priority, retries=1,
                          done=lambda r=r, k=k: thr_done(r) or part_path(cfg, r, k).is_file())) for k in range(chunks)]
        thresholds = add(Job(f"thresholds_{r}", ["vig_dagger.steps", "thresholds", *c, "--round", r], kind="cpu", deps=scores,
                             priority=base_priority, done=thr_done))
        workers = dg["workers_per_scorer"]
        workers = max(1, sim_slots(cfg) // len(dg["scorers"])) if workers == "auto" else int(workers)
        collections, retrains = [], []
        for s in dg["scorers"]:
            prepare = add(Job(f"dagger_prepare_{r}_{s}", ["vig_dagger.steps", "dagger-prepare", *c, "--round", r, "--scorer", s],
                              kind="cpu", deps=[thresholds], priority=base_priority + 1,
                              done=lambda r=r, s=s: settings_current(cfg, r, s)))
            out = paths.dagger(r, s)
            max_attempts = int(dg["max_attempts"] or 20 * n)
            finished = lambda out=out, max_attempts=max_attempts: (lambda k, a: k >= n or a >= max_attempts)(*count_successes(out))
            names = [add(Job(f"dagger_{r}_{s}_{w:02d}", ["vig_dagger.workers.dagger", *c, "--round", r, "--scorer", s,
                                                         "--worker", str(w), "--workers", str(workers)],
                             kind="sim", deps=[prepare], priority=base_priority + 2, retries=3, done=finished))
                     for w in range(workers)]
            collections += names
            data = add(Job(f"dagger_dataset_{r}_{s}", ["vig_dagger.steps", "dagger-dataset", *c, "--round", r, "--scorer", s],
                           kind="cpu", deps=names, priority=base_priority + 3, slots=convert_slots, retries=1,
                           done=lambda r=r, s=s: dataset.ready(paths.dataset(f"base_plus_{r}_{s}"))))
            retrains += [train(f"retrain/base_plus_{r}_{s}/seed{k}", [data], base_priority + 4) for k in seeds]
        add(Job(f"analyze_{r}", ["vig_dagger.steps", "analyze", *c, "--round", r], kind="cpu", deps=collections,
                priority=base_priority + 3,
                done=lambda r=r: all((paths.results() / f"dagger_{r}_{s}.json").exists() for s in dg["scorers"])))
        before = sorted(set(before) | set(shared) | set(retrains))

    # ---- evaluation and summary
    evals = []
    policies: dict[str, tuple[Path, str, list[int]]] = {}
    for name in training_sets(cfg):
        for s in seeds:
            policies[f"retrain/{name}/seed{s}"] = (paths.retrain_run(name, s) / "latest.pt", f"train/retrain/{name}/seed{s}", [1])
    if ev["evaluate_drivers"] and "moe" in cfg["rounds"]:
        source = "download_driver" if cfg["drivers"]["moe"]["source"] == "download" else "train/drivers/moe"
        levels = [int(f) for f in cfg["drivers"]["moe"]["factors"]] if ev["driver_all_levels"] else [1]
        policies["driver/moe"] = (paths.driver_dir("moe") / "latest.pt", source, levels)
    if ev["evaluate_drivers"] and "per_resolution" in cfg["rounds"]:
        # Each degraded-level policy on images degraded by its own factor (its full-resolution policy is base/seed0).
        for f, run in driver_runs(cfg, "per_resolution").items():
            if f != 1:
                policies[f"driver/per_resolution/res{f}"] = (run / "latest.pt", f"train/drivers/per_resolution/res{f}", [f])
    chunk = int(ev["chunk"])
    first, total = int(ev["seed_start"]), int(ev["num_episodes"])
    for label, (ckpt, dep, levels) in policies.items():
        for f in levels:
            for start in range(first, first + total, chunk):
                count = min(chunk, first + total - start)
                out = paths.evaluation(label) / f"level{f}" / f"seeds_{start}.jsonl"
                evals.append(add(Job(
                    f"eval/{label}/level{f}/{start}",
                    ["vig_dagger.workers.evaluate", "--checkpoint", str(ckpt), "--factor", str(f), "--seeds", str(start), str(count),
                     "--max-episode-steps", str(ev["max_episode_steps"]), "--actions-per-step", str(ev["actions_per_step"]),
                     "--out", str(out)],
                    kind="sim", deps=[dep], priority=60, retries=2, done=_eval_done(out, count))))
    add(Job("summarize", ["vig_dagger.steps", "summarize", *c], kind="cpu", deps=evals, priority=90))
    return list(jobs.values())


def write_config(cfg: dict) -> Path:
    """The resolved config every job reads (``<work_dir>/config.json``)."""
    path = Paths.of(cfg).config()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    return path
