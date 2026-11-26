"""Numbered tasks, for running the experiment on several machines with the tasks assigned by hand.

The job graph (``plan.py``) groups its work into tasks (``Job.task``), numbered in stage order so that a task's inputs
always come from lower numbers:

  stage 1  the MoEs            moe/sd<seed>: download or train the MoE, then score demo frames for its thresholds
  stage 2  DAgger collections  dagger/<collection>: one collection (500 successes) and its metrics; needs its MoE
  stage 3  summary             every collection side by side, the main figure and REPORT.md; needs every
                               collection's metrics
  stage 4  retrainings         retrain/<collection>: a dense full-resolution policy on the base demos + the
                               collection's DAgger data, and its evaluation; needs the collection's episodes (so it
                               runs on the machine that collected them)
  stage 5  planner comparisons iteration mode with iterate.planner_baselines: planner/round<r>, a dense policy on
                               as many planner demos as round r's dense policy trains on, and its evaluation
  stage 6  MoE evaluations     eval/moe/sd<seed>: the MoE at full resolution on the same scenes; needs the MoE
  stage 7  final report        the evaluation tables and figure next to everything above; needs every collection's
                               metrics and every evaluation

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

from vig_dagger.config import Collection, Paths, collections, downloaded, iterating, moe_key, planner_sizes, target_of

STAGES = {"moe": (1, "the MoEs and their thresholds"), "dagger": (2, "DAgger collections"), "summary": (3, "summary"),
          "retrain": (4, "retrainings on each collection's DAgger data, and their evaluations"),
          "planner": (5, "dense policies on planner demos only, as many as each round's dense policy trains on"),
          "eval": (6, "the MoEs evaluated at full resolution, for comparison"), "final": (7, "final report")}


def stage(task: str) -> str:
    return task.split("/")[0]


def _round_of(task: str) -> int | None:
    m = re.search(r"round(\d+)", task)
    return int(m.group(1)) if m else None


def _order_key(cfg: dict, task: str) -> tuple:
    """Stage, then MoE seeds and collections in the config's order (iteration mode: round, then stage, so a task's
    inputs come from lower numbers there too). The same config gives the same numbers on every machine."""
    kind, _, rest = task.partition("/")
    if iterating(cfg):
        r = _round_of(task)
        return (r if r is not None else len(cfg["iterate"]["rounds"]) + 1, STAGES[kind][0], task)
    if kind == "moe":
        inner = ([int(s) for s in cfg["moe"]["seeds"]].index(int(rest[2:])),)
    elif kind in ("dagger", "retrain"):
        inner = ([c.id for c in collections(cfg)].index(rest),)
    elif kind == "eval" and rest.startswith("extra/"):  # eval/extra/<name>: after the MoEs, in the config's order
        inner = (len(cfg["moe"]["seeds"]) + list(cfg["evaluation"].get("extra_policies") or {}).index(rest[6:]),)
    elif kind == "eval":  # eval/moe/sd<s>
        inner = ([int(s) for s in cfg["moe"]["seeds"]].index(int(rest.rsplit("sd", 1)[1])),)
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
        return [f"dagger/{rest}/**", f"results/collections/{rest}.json", f"results/collections/{rest}_*", f"videos/{rest}/**"]
    if kind == "retrain":
        return [f"policies/retrain/{rest}/**", f"eval/retrain/{rest}/**"]
    if kind == "planner":
        return [f"policies/planner/{rest}/**", f"eval/planner/{rest}/**"]
    if kind == "eval":
        return [f"eval/{rest}/**"]
    return ["results/**"]


def provides(cfg: dict, provider: str, reader: str | None = None) -> list[str]:
    """The provider's outputs that ``reader`` uses (all of them when reader is None)."""
    kind, _, rest = provider.partition("/")
    if reader is None:
        return outputs(cfg, provider)
    if kind == "moe":
        run = f"policies/moe/{rest}"
        if reader in ("summary", "final"):  # the training log and thresholds, for the report
            return [f"{run}/run_config.json", f"{run}/train_config.json", f"{run}/metrics.jsonl", f"{run}/thresholds.json"]
        return [f"{run}/latest.pt", f"{run}/run_config.json", f"{run}/thresholds.json", f"{run}/demo_scores.npz"]
    if kind == "dagger" and reader.startswith("retrain/"):  # the collection's episodes, to build the training set
        return [f"dagger/{rest}/settings.json", f"dagger/{rest}/episodes/*", f"dagger/{rest}/trajectories/*"]
    if kind in ("retrain", "planner"):  # for the final report: the evaluation episodes and the training log
        return [f"eval/{kind}/{rest}/**", f"policies/{kind}/{rest}/*/run_config.json", f"policies/{kind}/{rest}/*/metrics.jsonl"]
    if kind == "eval":
        return [f"eval/{rest}/**"]
    if kind == "dagger" and reader in ("summary", "final"):
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


def present(cfg: dict, task: str, reader: str | None = None) -> bool:
    """The task's outputs that other tasks (or ``reader``) read are in this work directory."""
    from vig_dagger.plan import evaluation_complete

    paths = Paths.of(cfg)
    kind, _, rest = task.partition("/")
    if kind == "dagger" and reader is not None and reader.startswith("retrain/"):  # its episodes, not only its metrics
        from vig_dagger.plan import collection_finished

        return collection_finished(cfg, Collection.parse(rest)) and (paths.dagger(rest) / "trajectories").is_dir()
    if kind in ("retrain", "planner"):
        return all(evaluation_complete(cfg, f"{kind}/{rest}/seed{int(s)}") for s in cfg["retrain"]["seeds"])
    if kind == "eval":
        return evaluation_complete(cfg, rest)
    if kind == "final":
        return (paths.results() / "eval_summary.json").is_file() and (paths.results() / "REPORT.md").is_file()
    if kind == "moe":
        seed = moe_key(rest)
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
    if kind == "moe" and rest.startswith("round"):
        r = int(rest[5:])
        data = "the base demos" + ("" if r == 1 else " + the DAgger data of round 1" if r == 2 else
                                   f" + the DAgger data of rounds 1–{r - 1}")
        return (f"round {r} MoE: train it ({cfg['training']['epochs']} epochs on {data}), then score every "
                f"{t['frame_stride']}th frame of its training demos (the adaptive threshold's reference)",
                "1 GPU (training, then inference)", "2–14 h")
    if kind == "eval" and "round" in rest:
        return (f"evaluate the {rest.split('/')[-1]} MoE at full resolution on {cfg['evaluation']['num_episodes']} scenes",
                "CPU cores + a GPU", "under 1 h")
    if kind == "moe":
        seed = int(rest[2:])
        score = f"score {t['num_demos']} demos for its VIG and flow-loss percentiles"
        if downloaded(cfg, seed):
            return f"MoE seed {seed}: download it (Hugging Face), {score}", "1 GPU (inference)", "0.5–1 h"
        return (f"MoE seed {seed}: train it ({cfg['training']['epochs']} epochs on {cfg['demos']['num_demos']} demos), "
                f"then {score}"), "1 GPU (training, then inference)", "8–14 h"
    if kind == "dagger":
        c = Collection.parse(rest)
        return (f"{c.label}: {target_of(cfg, c)} successful DAgger episodes and their metrics",
                "GPU + as many CPU cores as possible (simulator workers)", "20–60 worker-hours")
    if kind == "retrain":
        c = Collection.parse(rest)
        n = int(cfg["evaluation"]["num_episodes"])
        cut = " with the random-move steps cut out" if cfg["retrain"]["cut_random_moves"] and c.variant == "random_move" else ""
        if c.round is not None:
            total = sum(int(x) for x in cfg["iterate"]["rounds"][: c.round])
            return (f"after round {c.round}: a dense full-resolution policy on the {cfg['demos']['num_demos']} base demos + "
                    f"the {total} DAgger episodes of {'round 1' if c.round == 1 else f'rounds 1–{c.round}'}{cut} "
                    f"({cfg['training']['epochs']} epochs), then "
                    f"evaluate it on {n} scenes", "1 GPU (training), then CPU cores + a GPU (evaluation)", "2–24 h")
        return (f"retrain on {c.label}: a dense full-resolution policy on the {cfg['demos']['num_demos']} base demos + its "
                f"{cfg['dagger']['num_demos']} DAgger episodes{cut} ({cfg['training']['epochs']} epochs, seed "
                f"{', '.join(str(s) for s in cfg['retrain']['seeds'])}), then evaluate it on {n} scenes",
                "1 GPU (training), then CPU cores + a GPU (evaluation)", "8–24 h")
    if kind == "planner" and int(rest[5:]) == 0:
        return (f"the starting point: a dense full-resolution policy on the {cfg['demos']['num_demos']} base demos alone "
                f"({cfg['training']['epochs']} epochs), then evaluate it on {cfg['evaluation']['num_episodes']} scenes",
                "1 GPU (training), then CPU cores + a GPU (evaluation)", "1–3 h")
    if kind == "planner":
        r = int(rest[5:])
        size, base = planner_sizes(cfg)[r - 1], int(cfg["demos"]["num_demos"])
        after = "" if cfg["iterate"]["planner_baselines"] == "only" else "; after that policy"
        return (f"planner comparison for round {r}: a dense full-resolution policy on {size} planner demos (the {base} "
                f"base demos + {size - base} more collected like them), as many as the dense policy after round {r} "
                f"trains on ({cfg['training']['epochs']} epochs{after}), then evaluate it on "
                f"{cfg['evaluation']['num_episodes']} scenes", "CPU cores (collecting the demos), 1 GPU (training), "
                "then CPU cores + a GPU (evaluation)", "2–24 h")
    if kind == "eval":
        return (f"evaluate MoE seed {rest.rsplit('sd', 1)[1]} at full resolution on {cfg['evaluation']['num_episodes']} scenes "
                "(the comparison for the retrained policies)", "CPU cores + a GPU", "under 1 h")
    if kind == "final":
        ref = " and the reference" if (cfg.get("reference") or {}).get("dirs") else ""
        return (f"final report: the evaluations next to the collections{ref} (REPORT.md, retraining.png)", "CPU", "minutes")
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
            if n not in selected and not present(cfg, n, t):
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
        if not present(cfg, provider, reader):
            raise SystemExit(f"task {provider} is not done in {root}" + (f" (or lacks what {reader} needs)" if reader else "")
                             + "; nothing to pack")
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
