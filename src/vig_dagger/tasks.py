"""Numbered tasks, for running the experiment on several machines with the tasks assigned by hand.

The job graph (``plan.py``) groups its work into 63 tasks (``Job.task``): 25 trainings, each round's thresholds, the
four DAgger collections, 31 evaluations (one policy at one level, 200 episodes) and the summary. They are numbered in
stage order, so a task's inputs always come from lower numbers:

  stage 1  trainings that need only the demos   the MoE, res2 ... res64, base x3, base_plus_planner x3
  stage 2  thresholds                           a round's driver checkpoints (and the demos)
  stage 3  DAgger collections                   a round's driver checkpoints and thresholds
  stage 4  retrainings                          their DAgger training set (and the demos)
  stage 5  evaluations                          one checkpoint each
  stage 6  summary                              every evaluation's episodes and every collection's metrics

``vig-dagger run --tasks 3 17`` runs tasks on this machine (with the preparation they need: downloading and
converting the demos); every input from another task must already be in this work directory, at the same path as on
the machine that made it. ``vig-dagger pack`` puts a task's outputs (or only what a given task needs of them) into one
archive, ``vig-dagger unpack`` puts an archive's files in place. TASKS.md lists the tasks of configs/opendrawer.yaml.
"""

from __future__ import annotations

import re
import tarfile
import time
from pathlib import Path

from vig_dagger.config import Paths

STAGES = {"train1": (1, "trainings that need only the demos"), "thresholds": (2, "thresholds"),
          "dagger": (3, "DAgger collections"), "train4": (4, "retrainings on DAgger data"),
          "eval": (5, "evaluations"), "summary": (6, "summary")}


def stage(task: str) -> str:
    if task.startswith("train/retrain/base_plus_") and not task.startswith("train/retrain/base_plus_planner"):
        return "train4"
    if task.startswith("train/"):
        return "train1"
    return task.split("/")[0]


def _order_key(cfg: dict, task: str) -> tuple:
    """A total order: stage, then rounds and scorers in the config's order, factors and seeds by number. The same
    config gives the same numbers on every machine."""
    from vig_dagger.config import training_sets

    rounds, scorers = list(cfg["rounds"]), list(cfg["dagger"]["scorers"])
    sets = training_sets(cfg)

    def run_key(run: str) -> tuple:  # drivers/moe, drivers/per_resolution/res<k>, retrain/<set>/seed<s>
        parts = run.split("/")
        if parts[:2] == ["drivers", "moe"]:
            return (0, 0, 0)
        if parts[:2] == ["drivers", "per_resolution"]:
            return (1, int(parts[2][3:]), 0)
        return (2, sets.index(parts[1]), int(parts[2][4:]))

    kind, _, rest = task.partition("/")
    s = STAGES[stage(task)][0]
    if kind == "train":
        inner = run_key(rest)
    elif kind == "thresholds":
        inner = (rounds.index(rest),)
    elif kind == "dagger":
        r, sc = rest.split("/")
        inner = (rounds.index(r), scorers.index(sc))
    elif kind == "eval":
        label, level = rest.rsplit("/level", 1)
        if label.startswith("driver/moe"):
            inner = (0, 0, int(level))
        elif label.startswith("driver/per_resolution/res"):
            inner = (1, int(label.rsplit("res", 1)[1]), int(level))
        else:  # retrain/<set>/seed<s>
            inner = (2,) + run_key(label)[1:] + (int(level),)
    else:
        inner = ()
    return (s, inner, task)


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


# ---------------------------------------------------------------- what a task reads from another


def provides(cfg: dict, provider: str, reader: str | None = None) -> list[str]:
    """Glob patterns (work_dir-relative; ``dir/**`` = everything below dir) of the provider's outputs that ``reader``
    uses, or all of them (reader None)."""
    from vig_dagger.board import outputs
    from vig_dagger.train import resolve

    paths = Paths.of(cfg)
    exports, archive = outputs(cfg, provider)
    if reader is None:
        return exports + archive
    kind = provider.split("/")[0]
    if reader == "summary":  # results only: evaluation episodes, DAgger metrics, training logs
        if kind == "eval":
            return exports
        if kind == "dagger":
            r, s = provider.split("/")[1:]
            return [f"results/dagger_{r}_{s}.json", f"results/dagger_{r}_{s}_*", f"results/figures/dagger_{r}_{s}_*"]
        if kind == "train":
            run = resolve(cfg, provider[len("train/"):])[1].relative_to(paths.root)
            return [f"{run}/run_config.json", f"{run}/metrics.jsonl"]
        return []
    if kind == "dagger":  # a retraining reads the DAgger training set
        r, s = provider.split("/")[1:]
        return [f"data/datasets/dagger_{r}_{s}/**"]
    return exports  # a checkpoint (with its config and log), or thresholds


def local_files(root: Path, patterns: list[str]) -> list[Path]:
    from vig_dagger.board import _local_files

    return _local_files(root, patterns)


def present(cfg: dict, provider: str) -> bool:
    from vig_dagger.board import exports_present

    return exports_present(cfg, provider)


# ---------------------------------------------------------------- the list


def describe(cfg: dict, task: str) -> tuple[str, str, str]:
    """(what it is, what it needs to run, a rough time for configs/opendrawer.yaml)."""
    kind, _, rest = task.partition("/")
    n_base, n_dagger = int(cfg["demos"]["num_demos"]), int(cfg["dagger"]["num_demos"])
    episodes = int(cfg["evaluation"]["num_episodes"])
    if kind == "train":
        demos = n_base + (n_dagger if "base_plus" in rest else 0)
        hours = "3–10 h" if demos == n_base else "7–20 h"
        if rest == "drivers/moe":
            what = f"the {len(cfg['drivers']['moe']['factors'])}-level MoE (round `moe`'s driver)"
        elif rest.startswith("drivers/per_resolution/res"):
            what = f"dense policy on images degraded by {rest.rsplit('res', 1)[1]} (round `per_resolution`)"
        else:
            what = f"retraining `{rest.split('/')[1]}` {rest.split('/')[2]} ({demos} demos)"
        return what, "1 GPU (training)", hours
    if kind == "thresholds":
        return (f"round `{rest}`: score {cfg['thresholds']['num_demos']} demos, VIG and flow-loss percentiles",
                "1 GPU (inference)", "0.5–1 h")
    if kind == "dagger":
        r, s = rest.split("/")
        return (f"round `{r}`, {'VIG' if s == 'vig' else 'flow-loss'} DAgger: {n_dagger} successful episodes, their "
                "training set and metrics", "RTX GPU + many CPU cores (Isaac Sim)", "3–10 h")
    if kind == "eval":
        label, level = rest.rsplit("/level", 1)
        return (f"evaluate `{label}` at {'full resolution' if level == '1' else '1/' + level}, {episodes} episodes",
                "RTX GPU + ~8 CPU cores (Isaac Sim)", "1–2 h")
    return "evaluation summary and REPORT.md", "CPU", "minutes"


def table(cfg: dict, jobs: list) -> list[dict]:
    by_name = {j.name: j for j in jobs}
    tasks = numbered(cfg, jobs)
    number = {t: i + 1 for i, t in enumerate(tasks)}
    rows = []
    for t in tasks:
        what, resources, hours = describe(cfg, t)
        rows.append(dict(number=number[t], task=t, stage=stage(t), what=what, resources=resources, time=hours,
                         needs=sorted(number[n] for n in needs(by_name, t)),
                         outputs=provides(cfg, t)[:3] if not t.startswith("summary") else ["results/"]))
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
            if n in selected:
                continue
            if not _reader_inputs_present(cfg, n, t):
                problems.append(f"task {number[t]} ({t}) needs the outputs of task {number[n]} ({n}): "
                                + ", ".join(provides(cfg, n, t)))
    return problems


def _reader_inputs_present(cfg: dict, provider: str, reader: str) -> bool:
    if reader != "summary":
        return present(cfg, provider)
    root = Paths.of(cfg).root
    if provider.startswith("eval/"):
        return present(cfg, provider)
    if provider.startswith("dagger/"):
        r, s = provider.split("/")[1:]
        return (root / "results" / f"dagger_{r}_{s}.json").is_file()
    return True


def select_jobs(cfg: dict, jobs: list, selected: list[str]) -> list:
    """The selected tasks' jobs and the preparation they need. The other tasks' jobs they depend on are kept as done,
    without dependencies of their own: ``check_inputs`` found their outputs here."""
    from dataclasses import replace

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


def pack(cfg: dict, jobs: list, providers: list[str], reader: str | None, out: Path) -> Path:
    """One archive (paths relative to the work directory; checkpoints weights-only) of the providers' outputs, or only
    of what ``reader`` uses of them."""
    from vig_dagger.hub import export_checkpoint

    root = Paths.of(cfg).root
    files = []
    for provider in providers:
        if not present(cfg, provider) and reader != "summary":
            raise SystemExit(f"{provider} is not done in {root}; nothing to pack")
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
    import shutil

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
