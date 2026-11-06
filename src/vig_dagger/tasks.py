"""Numbered tasks, for running the experiment on several machines with the tasks assigned by hand.

The job graph (``plan.py``) groups its work into tasks (``Job.task``), numbered in stage order so that a task's inputs
always come from lower numbers:

  stage 1  the MoEs            moe/sd<seed>: download or train the MoE, then score demo frames for its thresholds
  stage 2  DAgger collections  dagger/<collection>: one collection (500 successes) and its metrics; needs its MoE
  stage 3  summary             every collection side by side, the main figure and REPORT.md; needs every
                               collection's metrics

``vig-dagger run --tasks 4 17`` runs tasks on this machine, with the preparation they need (downloading and
converting the demos); every input from another task must already be in this work directory, at the same path as on
the machine that made it. ``vig-dagger pack`` puts a task's outputs (or only what a given task needs of them) into one
archive, ``vig-dagger unpack`` puts an archive's files in place. TASKS.md lists the tasks of configs/plugcharger.yaml.
"""

from __future__ import annotations

import re
import shutil
import tarfile
import time
from dataclasses import replace
from pathlib import Path

from vig_dagger.config import Collection, Paths, collections, downloaded

STAGES = {"moe": (1, "the MoEs and their thresholds"), "dagger": (2, "DAgger collections"), "summary": (3, "summary")}


def stage(task: str) -> str:
    return task.split("/")[0]


def _order_key(cfg: dict, task: str) -> tuple:
    """Stage, then MoE seeds and collections in the config's order. The same config gives the same numbers on every
    machine."""
    kind, _, rest = task.partition("/")
    if kind == "moe":
        inner = ([int(s) for s in cfg["moe"]["seeds"]].index(int(rest[2:])),)
    elif kind == "dagger":
        inner = ([c.id for c in collections(cfg)].index(rest),)
    else:
        inner = ()
    return (STAGES[kind][0], inner, task)


def numbered(cfg: dict, jobs: list) -> list[str]:
    """The experiment's tasks in number order (task number = index + 1)."""
    return sorted({j.task for j in jobs if j.task is not None}, key=lambda t: _order_key(cfg, t))


def parse(spec: list[str], tasks: list[str]) -> list[str]:
    """Task names from numbers and ranges ("3", "17", "32-40") or names."""
    out = []
    for item in spec:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(r"\d+(-\d+)?", part):
                lo, _, hi = part.partition("-")
                for n in range(int(lo), int(hi or lo) + 1):
                    if not 1 <= n <= len(tasks):
                        raise SystemExit(f"there is no task {n} (tasks are 1 ... {len(tasks)}; see `vig-dagger tasks`)")
                    out.append(tasks[n - 1])
            elif part in tasks:
                out.append(part)
            else:
                raise SystemExit(f"unknown task {part!r} (use the numbers of `vig-dagger tasks`)")
    return list(dict.fromkeys(out))


def needs(jobs: dict, task: str) -> set[str]:
    """The tasks whose outputs this task reads."""
    members = [n for n, j in jobs.items() if j.task == task]
    seen, stack, out = set(), list(members), set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        job = jobs[name]
        if job.task is not None and job.task != task:
            out.add(job.task)
            continue
        stack += job.deps
    return out


# ---------------------------------------------------------------- outputs


def outputs(cfg: dict, task: str) -> list[str]:
    """Every output of a task, as work_dir-relative glob patterns (``dir/**``: everything below dir)."""
    kind, _, rest = task.partition("/")
    if kind == "moe":
        run = f"policies/moe/{rest}"
        return [f"{run}/latest.pt", f"{run}/run_config.json", f"{run}/train_config.json", f"{run}/metrics.jsonl",
                f"{run}/thresholds.json", f"{run}/demo_scores.npz"]
    if kind == "dagger":
        return [f"dagger/{rest}/**", f"results/collections/{rest}.json", f"results/collections/{rest}_*"]
    return ["results/**"]


def provides(cfg: dict, provider: str, reader: str | None = None) -> list[str]:
    """The provider's outputs that ``reader`` uses (all of them when reader is None)."""
    kind, _, rest = provider.partition("/")
    if reader is None:
        return outputs(cfg, provider)
    if kind == "moe":
        run = f"policies/moe/{rest}"
        if reader == "summary":  # the training log and thresholds, for the report
            return [f"{run}/run_config.json", f"{run}/train_config.json", f"{run}/metrics.jsonl", f"{run}/thresholds.json"]
        return [f"{run}/latest.pt", f"{run}/run_config.json", f"{run}/thresholds.json", f"{run}/demo_scores.npz"]
    if kind == "dagger" and reader == "summary":
        return [f"results/collections/{rest}.json", f"results/collections/{rest}_episodes.csv",
                f"results/collections/{rest}_takeovers.csv", f"results/collections/{rest}_moves.json",
                f"results/collections/{rest}_background.png"]
    return outputs(cfg, provider)


def local_files(root: Path, patterns: list[str]) -> list[Path]:
    """The files under ``root`` matching the patterns (``dir/**``: every file below dir)."""
    out = []
    for pattern in patterns:
        if pattern.endswith("/**"):
            base = root / pattern[:-3]
            out += [p for p in base.rglob("*") if p.is_file()] if base.is_dir() else []
        else:
            out += [p for p in root.glob(pattern) if p.is_file()]
    return sorted(set(out))


def present(cfg: dict, task: str) -> bool:
    """The task's outputs that other tasks read are in this work directory."""
    paths = Paths.of(cfg)
    kind, _, rest = task.partition("/")
    if kind == "moe":
        seed = int(rest[2:])
        return paths.moe_checkpoint(seed).is_file() and paths.thresholds(seed).is_file()
    if kind == "dagger":
        from vig_dagger.plan import analysis_current, collection_finished

        if (paths.dagger(rest) / "episodes").is_dir():  # the machine that collected it: finished and analyzed
            return collection_finished(cfg, Collection.parse(rest)) and analysis_current(cfg, rest)
        return paths.collection_results(rest).with_suffix(".json").is_file()  # metrics unpacked from that machine
    return (paths.results() / "summary.json").is_file()


# ---------------------------------------------------------------- the list


def describe(cfg: dict, task: str) -> tuple[str, str, str]:
    """(what it is, what it needs to run, a rough time for configs/plugcharger.yaml)."""
    kind, _, rest = task.partition("/")
    t = cfg["thresholds"]
    if kind == "moe":
        seed = int(rest[2:])
        score = f"score {t['num_demos']} demos for its VIG and flow-loss percentiles"
        if downloaded(cfg, seed):
            return f"MoE seed {seed}: download it (Hugging Face), {score}", "1 GPU (inference)", "0.5–1 h"
        return (f"MoE seed {seed}: train it ({cfg['training']['epochs']} epochs on {cfg['demos']['num_demos']} demos), "
                f"then {score}"), "1 GPU (training, then inference)", "8–14 h"
    if kind == "dagger":
        c = Collection.parse(rest)
        return (f"{c.label}: {cfg['dagger']['num_demos']} successful DAgger episodes and their metrics",
                "GPU + as many CPU cores as possible (simulator workers)", "20–60 worker-hours")
    return "every collection side by side: results/summary.*, the main figure, REPORT.md", "CPU", "minutes"


def table(cfg: dict, jobs: list) -> list[dict]:
    by_name = {j.name: j for j in jobs}
    tasks = numbered(cfg, jobs)
    number = {t: i + 1 for i, t in enumerate(tasks)}
    rows = []
    for t in tasks:
        what, resources, hours = describe(cfg, t)
        rows.append(dict(number=number[t], task=t, stage=stage(t), what=what, resources=resources, time=hours,
                         needs=sorted(number[n] for n in needs(by_name, t))))
    users: dict[int, list[int]] = {}
    for row in rows:
        for n in row["needs"]:
            users.setdefault(n, []).append(row["number"])
    for row in rows:
        row["used_by"] = users.get(row["number"], [])
    return rows


def _ranges(numbers: list[int]) -> str:
    if not numbers:
        return "–"
    out, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(out)


def markdown(cfg: dict, jobs: list) -> str:
    rows = table(cfg, jobs)
    lines = []
    for key, (s, title) in sorted(STAGES.items(), key=lambda kv: kv[1][0]):
        group = [r for r in rows if r["stage"] == key]
        if not group:
            continue
        lines += [f"### Stage {s}: {title}", "",
                  "| # | task | runs on | time | needs tasks | used by tasks |", "|---|---|---|---|---|---|"]
        for r in group:
            lines.append(f"| {r['number']} | {r['what']} <br>`{r['task']}` | {r['resources']} | {r['time']} | "
                         f"{_ranges(r['needs'])} | {_ranges(r['used_by'])} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- running, packing


def check_inputs(cfg: dict, jobs: list, selected: list[str]) -> list[str]:
    """Problems that keep the selected tasks from running here: inputs from other (unselected) tasks not in this work
    directory."""
    by_name = {j.name: j for j in jobs}
    tasks = numbered(cfg, jobs)
    number = {t: i + 1 for i, t in enumerate(tasks)}
    problems = []
    for t in selected:
        for n in sorted(needs(by_name, t), key=lambda x: number[x]):
            if n not in selected and not present(cfg, n):
                problems.append(f"task {number[t]} ({t}) needs the outputs of task {number[n]} ({n}): "
                                + ", ".join(provides(cfg, n, t)))
    return problems


def select_jobs(cfg: dict, jobs: list, selected: list[str]) -> list:
    """The selected tasks' jobs and the preparation they need. The other tasks' jobs they depend on are kept as done,
    without dependencies of their own: ``check_inputs`` found their outputs here."""
    by_name = {j.name: j for j in jobs}
    keep, boundary, stack = set(), set(), [n for n, j in by_name.items() if j.task in selected]
    while stack:
        name = stack.pop()
        if name in keep:
            continue
        keep.add(name)
        job = by_name[name]
        if job.task is not None and job.task not in selected:
            boundary.add(name)
            continue
        stack += job.deps
    return [replace(j, deps=[], done=lambda: True) if j.name in boundary else j for j in jobs if j.name in keep]


def pack(cfg: dict, providers: list[str], reader: str | None, out: Path) -> Path:
    """One archive (paths relative to the work directory; checkpoints weights-only) of the providers' outputs, or only
    of what ``reader`` uses of them."""
    from vig_dagger.hub import export_checkpoint

    root = Paths.of(cfg).root
    files = []
    for provider in providers:
        if not present(cfg, provider):
            raise SystemExit(f"task {provider} is not done in {root}; nothing to pack")
        files += local_files(root, provides(cfg, provider, reader))
    if not files:
        raise SystemExit("nothing to pack")
    staging = root / "tmp" / "pack"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w") as tar:
        for path in sorted(set(files)):
            rel = path.relative_to(root)
            source = path
            if path.suffix == ".pt":  # weights only: enough for every later task, a third of the size
                source = staging / rel
                export_checkpoint(path, source)
            tar.add(source, arcname=str(rel))
    shutil.rmtree(staging, ignore_errors=True)
    return out


def unpack(cfg: dict, archive: Path) -> list[str]:
    """Put an archive's files at their paths in this work directory."""
    root = Paths.of(cfg).root
    if not Path(archive).is_file():
        raise SystemExit(f"{archive}: no such archive")
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for m in members:
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                raise SystemExit(f"{archive}: unsafe path {m.name!r}")
        tar.extractall(root, members=members, filter="data")
    return [m.name for m in members]


def default_archive(cfg: dict, numbers: list[int], reader_number: int | None) -> Path:
    tag = "_".join(str(n) for n in numbers[:6]) + ("_etc" if len(numbers) > 6 else "")
    name = f"tasks_{tag}" + (f"_for_{reader_number}" if reader_number else "") + f"_{time.strftime('%m%d_%H%M')}.tar"
    return Paths.of(cfg).root / "packs" / name
