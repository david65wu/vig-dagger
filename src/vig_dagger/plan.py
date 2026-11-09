"""The whole experiment as one graph of jobs (``vig-dagger run`` hands it to the queue in ``jobs.py``).

  part 1   download_demos (or collect_planner_*) -> convert_planner (the base dataset; only if a MoE is trained here)
  part 2   per MoE seed s:  download_moe_sd<s> or train/moe/sd<s>  -> score_sd<s>_<chunk> -> thresholds_sd<s>
  part 3   per collection c (seed x scorer x threshold x variant):
           dagger_prepare_<c> -> dagger_<c>_<worker> ... -> analyze_<c>
  end      summarize (after every collection's analysis)
  part 4   per collection c: dagger_dataset_<c> (base + its DAgger data) -> train/retrain/<c>/seed<s>
           -> eval/retrain/<c>/seed<s>/<first scene> (chunks of evaluation.chunk episodes)
           per MoE seed s: eval/moe/sd<s>/<first scene> (the MoE at full resolution, for comparison)
  end      final_report (after every collection's analysis and every evaluation)

Each job belongs to a numbered task (``tasks.py``): ``moe/sd<s>`` (the MoE and its thresholds), ``dagger/<c>`` (one
collection and its metrics), ``summary``, ``retrain/<c>`` (a collection's retraining and its evaluation; it needs the
collection's episodes, so it runs where they were collected), ``eval/moe/sd<s>`` or ``final``; the demo download and
conversion are preparation every machine does itself when a task needs them.

With a task board (``sharing.board``) every task T also gets a ``publish/T`` job (copy its outputs to the board, on the
machine that ran it) and a ``fetch/T`` job (copy them from the board, on a machine whose task needs them); a job that
needs another task's outputs depends on ``fetch/T`` instead of on T's jobs (``share``).

Lower priority numbers start first when several jobs are ready: the demos, the MoEs (trainings take hours, so they
start first), thresholds, then the collections in order (MoE seed, variant, scorer, threshold), so the first ones
finish first.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from vig_dagger import dataset
from vig_dagger.config import Collection, Paths, board_mode, collections, downloaded
from vig_dagger.jobs import Job
from vig_dagger.steps import dagger_settings, names_current, raw_parts
from vig_dagger.workers.dagger import count_successes
from vig_dagger.workers.score_demos import part_path


def sim_slots(cfg: dict) -> int:
    value = cfg["resources"]["sim_slots"]
    return (os.cpu_count() or 8) // 2 if value == "auto" else int(value)


def workers_per_collection(cfg: dict) -> int:
    value = cfg["dagger"]["workers_per_collection"]
    return max(1, sim_slots(cfg) // 2) if value == "auto" else int(value)


def settings_current(cfg: dict, collection: Collection) -> bool:
    """The collection's settings.json exists and matches the current config (threshold resolved)."""
    path = Paths.of(cfg).dagger(collection) / "settings.json"
    try:
        return json.loads(path.read_text()) == json.loads(json.dumps(dagger_settings(cfg, collection)))
    except (OSError, SystemExit, KeyError, ValueError):
        return False


def collection_finished(cfg: dict, collection: Collection) -> bool:
    n = int(cfg["dagger"]["num_demos"])
    max_attempts = int(cfg["dagger"]["max_attempts"] or 20 * n)
    successes, attempts = count_successes(Paths.of(cfg).dagger(collection))
    return successes >= n or attempts >= max_attempts


def analysis_current(cfg: dict, collection: Collection | str) -> bool:
    """The collection's metrics exist and cover every attempt it has now (an analysis run mid-collection by hand does
    not count once more attempts finished)."""
    path = Paths.of(cfg).collection_results(collection).with_suffix(".json")
    if not path.is_file():
        return False
    try:
        analyzed = json.loads(path.read_text())["metrics"]["collection"]["attempts"]
    except (OSError, KeyError, ValueError):
        return False
    return analyzed == count_successes(Paths.of(cfg).dagger(collection))[1]


def eval_done(out: Path, count: int):
    return lambda: out.exists() and len([line for line in out.read_text().splitlines() if line.strip()]) >= count


def evaluation_complete(cfg: dict, label: str) -> bool:
    """Every episode of a policy's evaluation is in eval/<label>/level1/."""
    ev = cfg["evaluation"]
    first, total, chunk = int(ev["seed_start"]), int(ev["num_episodes"]), int(ev["chunk"])
    folder = Paths.of(cfg).evaluation(label) / "level1"
    return all(eval_done(folder / f"seeds_{start}.jsonl", min(chunk, first + total - start))()
               for start in range(first, first + total, chunk))


def select(jobs: list[Job], pattern: str) -> list[Job]:
    """The jobs whose name matches the regular expression, plus every job they depend on."""
    by_name = {j.name: j for j in jobs}
    keep, stack = set(), [j.name for j in jobs if re.search(pattern, j.name)]
    while stack:
        name = stack.pop()
        if name not in keep:
            keep.add(name)
            stack += by_name[name].deps
    return [j for j in jobs if j.name in keep]


def share(cfg: dict, jobs: dict[str, Job], c: list[str]) -> None:
    """Several machines: add each task's publish and fetch jobs, and route every dependency on another task's jobs
    through that task's fetch job."""
    from vig_dagger import board

    members: dict[str, list[str]] = {}
    for job in jobs.values():
        if job.task is not None:
            members.setdefault(job.task, []).append(job.name)
    for task, names in members.items():
        priority = min(jobs[n].priority for n in names)
        jobs[f"publish/{task}"] = Job(f"publish/{task}", ["vig_dagger.steps", "publish", *c, "--task", task], kind="cpu",
                                      deps=list(names), priority=priority, retries=3, task=task,
                                      done=lambda task=task: board.task_done(cfg, task))
        jobs[f"fetch/{task}"] = Job(f"fetch/{task}", ["vig_dagger.steps", "fetch", *c, "--task", task], kind="cpu",
                                    deps=[f"publish/{task}"], priority=priority, retries=3,
                                    done=lambda task=task: board.exports_present(cfg, task))
    for job in jobs.values():
        if job.name.startswith(("publish/", "fetch/")):
            continue
        deps = []
        for dep in job.deps:
            other = jobs[dep].task
            if other is not None and other != job.task:
                dep = f"fetch/{other}"
            if dep not in deps:
                deps.append(dep)
        job.deps = deps


def build(cfg: dict, config_path: Path) -> list[Job]:
    paths = Paths.of(cfg)
    c = ["--config", str(config_path)]
    d = cfg["demos"]
    convert_slots = max(1, min(int(d["convert_workers"]), sim_slots(cfg)))
    jobs: dict[str, Job] = {}

    def add(job: Job) -> str:
        jobs.setdefault(job.name, job)
        return job.name

    # ---- part 1: demos (preparation, no task)
    if d["source"] == "download":
        demo_deps = [add(Job("download_demos", ["vig_dagger.steps", "download-demos", *c], kind="cpu", priority=0,
                             retries=2, done=lambda: bool(raw_parts(paths.raw_demos()))))]
    else:
        per, total, raw = int(d["demos_per_part"]), int(d["num_demos"]), paths.raw_demos()
        demo_deps = []
        for k in range(math.ceil(total / per)):
            count = min(per, total - k * per)
            demo_deps.append(add(Job(
                f"collect_planner_{k:03d}",
                ["vig_dagger.workers.collect", "--out", str(raw), "--part", str(k), "--num-demos", str(count),
                 "--start-seed", str(d["seed_start"] + 1000 * k), "--strategy-seed", str(d["strategy_seed"] + k),
                 "--max-episode-steps", str(d["max_episode_steps"])],
                kind="sim", priority=0, retries=1, done=lambda p=raw / f"part_{k:03d}.done.json": p.is_file())))
    convert = lambda: add(Job("convert_planner", ["vig_dagger.steps", "convert", *c], kind="cpu", deps=demo_deps,  # noqa: E731
                              priority=1, slots=convert_slots, retries=1,
                              done=lambda: dataset.ready(paths.dataset("base")) and names_current(cfg, paths.dataset("planner"))))

    # ---- part 2: the MoEs and their thresholds
    chunks = int(cfg["thresholds"]["chunks"])
    thresholds_job: dict[int, str] = {}
    moe_job: dict[int, str] = {}
    for seed in (int(s) for s in cfg["moe"]["seeds"]):
        task = f"moe/sd{seed}"
        has_moe = lambda seed=seed: paths.moe_checkpoint(seed).is_file()
        if downloaded(cfg, seed):
            moe = add(Job(f"download_moe_sd{seed}", ["vig_dagger.steps", "download-moe", *c, "--seed", str(seed)],
                          kind="cpu", priority=2, retries=2, done=has_moe, task=task))
        else:
            moe = add(Job(f"train/moe/sd{seed}", ["vig_dagger.train", *c, "--run", f"moe/sd{seed}"], kind="train",
                          deps=[convert()], priority=2, retries=int(cfg["training"]["max_restarts"]), done=has_moe, task=task))
        moe_job[seed] = moe
        thr_done = lambda seed=seed: paths.thresholds(seed).is_file()
        scores = [add(Job(f"score_sd{seed}_{k}", ["vig_dagger.workers.score_demos", *c, "--seed", str(seed), "--chunk", str(k),
                                                  "--chunks", str(chunks)],
                          kind="sim", deps=[moe, *demo_deps], priority=5, retries=1, task=task,
                          done=lambda seed=seed, k=k: thr_done(seed) or part_path(cfg, seed, k).is_file()))
                  for k in range(chunks)]
        thresholds_job[seed] = add(Job(f"thresholds_sd{seed}", ["vig_dagger.steps", "thresholds", *c, "--seed", str(seed)],
                                       kind="cpu", deps=scores, priority=5, done=thr_done, task=task))

    # ---- part 3: the DAgger collections
    workers = workers_per_collection(cfg)
    analyses, analysis_of = [], {}
    for index, col in enumerate(collections(cfg)):
        task, priority = f"dagger/{col.id}", 10 + index
        prepare = add(Job(f"dagger_prepare_{col.id}", ["vig_dagger.steps", "dagger-prepare", *c, "--collection", col.id],
                          kind="cpu", deps=[thresholds_job[col.seed]], priority=priority, task=task,
                          done=lambda col=col: settings_current(cfg, col)))
        finished = lambda col=col: collection_finished(cfg, col)
        names = [add(Job(f"dagger_{col.id}_{w:02d}", ["vig_dagger.workers.dagger", *c, "--collection", col.id,
                                                      "--worker", str(w), "--workers", str(workers)],
                         kind="sim", deps=[prepare], priority=priority, retries=3, done=finished, task=task))
                 for w in range(workers)]
        analysis_of[col.id] = add(Job(f"analyze_{col.id}", ["vig_dagger.steps", "analyze", *c, "--collection", col.id],
                                      kind="cpu", deps=names, priority=priority, task=task,
                                      done=lambda col=col: collection_finished(cfg, col) and analysis_current(cfg, col)))
        analyses.append(analysis_of[col.id])
    add(Job("summarize", ["vig_dagger.steps", "summarize", *c], kind="cpu", deps=analyses, priority=99, task="summary"))

    # ---- part 4: retraining on each collection's DAgger data, and evaluation
    ev = cfg["evaluation"]
    first, total, chunk = int(ev["seed_start"]), int(ev["num_episodes"]), int(ev["chunk"])

    def evaluate(label: str, checkpoint: Path, dep: str, task: str, priority: int) -> list[str]:
        out = []
        for start in range(first, first + total, chunk):
            count = min(chunk, first + total - start)
            path = paths.evaluation(label) / "level1" / f"seeds_{start}.jsonl"
            out.append(add(Job(
                f"eval/{label}/{start}",
                ["vig_dagger.workers.evaluate", "--checkpoint", str(checkpoint), "--factor", "1", "--seeds", str(start),
                 str(count), "--max-episode-steps", str(ev["max_episode_steps"]), "--actions-per-step",
                 str(ev["actions_per_step"]), "--out", str(path)],
                kind="sim", deps=[dep], priority=priority, retries=2, task=task, done=eval_done(path, count))))
        return out

    evals = []
    for index, col in enumerate(collections(cfg)):
        task, priority = f"retrain/{col.id}", 60 + index
        data = add(Job(f"dagger_dataset_{col.id}", ["vig_dagger.steps", "dagger-dataset", *c, "--collection", col.id],
                       kind="cpu", deps=[analysis_of[col.id], convert()], priority=priority, slots=convert_slots, retries=1,
                       task=task, done=lambda col=col: dataset.ready(paths.dataset(f"base_plus_{col.id}"))))
        for seed in (int(x) for x in cfg["retrain"]["seeds"]):
            run = f"retrain/{col.id}/seed{seed}"
            ckpt = paths.retrain_run(col, seed) / "latest.pt"
            trained = add(Job(f"train/{run}", ["vig_dagger.train", *c, "--run", run], kind="train", deps=[data],
                              priority=priority, retries=int(cfg["training"]["max_restarts"]), task=task,
                              done=lambda ckpt=ckpt: ckpt.is_file()))
            evals += evaluate(run, ckpt, trained, task, priority)
    if ev["evaluate_moes"]:
        for seed in (int(s) for s in cfg["moe"]["seeds"]):
            evals += evaluate(f"moe/sd{seed}", paths.moe_checkpoint(seed), moe_job[seed], f"eval/moe/sd{seed}", 55)
    add(Job("final_report", ["vig_dagger.steps", "summarize", *c], kind="cpu", deps=analyses + evals, priority=99,
            task="final"))
    if board_mode(cfg):
        share(cfg, jobs, c)
    return list(jobs.values())


def write_config(cfg: dict) -> Path:
    """The resolved config every job reads (``<work_dir>/config.json``)."""
    path = Paths.of(cfg).config()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    return path
