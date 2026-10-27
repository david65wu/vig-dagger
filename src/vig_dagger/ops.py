"""Operating an experiment: ``vig-dagger check`` (preflight), ``status``, ``stop``, and ``merge`` (two machines).
With ``sharing.mode: hub``, ``check`` also checks the Hugging Face login and ``status`` shows the shared repo's tasks."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import psutil

from vig_dagger.config import SPLITS, Paths, active_rounds, hub_mode, training_sets

DISK_NEEDED_GB = 150  # downloads 39 GB, DAgger trajectories ~60 GB, checkpoints ~20 GB, datasets and logs


# ---------------------------------------------------------------- check


def check(cfg: dict) -> bool:
    """Everything a multi-day run needs, tested in a minute or two. Returns True when nothing failed."""
    failures, warnings = [], []

    def report(ok: bool, what: str, detail: str = "", warn_only: bool = False) -> None:
        tag = "ok  " if ok else ("WARN" if warn_only else "FAIL")
        print(f"[{tag}] {what}{': ' + detail if detail else ''}", flush=True)
        if not ok:
            (warnings if warn_only else failures).append(what)

    paths = Paths.of(cfg)
    print(f"experiment {cfg['name']}, work_dir {paths.root}, rounds {cfg['rounds']}, scorers {cfg['dagger']['scorers']}"
          + (f"; this machine runs only round {SPLITS[cfg['split']]} (split: {cfg['split']})" if cfg["split"] != "none" else ""))

    # disk
    paths.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.root).free / 2**30
    report(os.access(paths.root, os.W_OK), "work_dir is writable", str(paths.root))
    report(free >= DISK_NEEDED_GB, "free disk in work_dir", f"{free:.0f} GB (the experiment needs about {DISK_NEEDED_GB} GB)",
           warn_only=free >= 60)

    # tools and GPUs
    report(shutil.which("ffmpeg") is not None, "ffmpeg on PATH", "needed to write training videos")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        report(bool(out), "nvidia-smi lists GPUs", "; ".join(out.splitlines()))
    except Exception as exc:
        report(False, "nvidia-smi lists GPUs", str(exc))
    import torch

    report(torch.cuda.is_available(), "torch sees CUDA", f"{torch.cuda.device_count()} device(s), torch {torch.__version__}")

    # rendering (SAPIEN needs Vulkan)
    try:
        from vig_dagger.envs import POLICY_CAMERAS, make_env

        env = make_env(max_episode_steps=10)
        obs, _ = env.reset(seed=0)
        shapes = {c: tuple(obs["sensor_data"][c]["rgb"].shape) for c in POLICY_CAMERAS}
        env.close()
        report(all(s == (1, 256, 256, 3) for s in shapes.values()), "simulator renders the policy cameras", str(shapes))
    except Exception as exc:
        report(False, "simulator renders the policy cameras", f"{type(exc).__name__}: {exc} (SAPIEN needs a working Vulkan driver)")

    # Hugging Face
    from huggingface_hub import HfApi

    api = HfApi()
    for needed, repo, repo_type, path in (
        (cfg["demos"]["source"] == "download", cfg["hub"]["dataset_repo"], "dataset", "raw/planner/part_000.h5"),
        (cfg["retrain"]["planner_baseline"] and cfg["demos"]["source"] == "download", cfg["hub"]["dataset_repo"], "dataset",
         "raw/planner_extra/part_000.h5"),
        ("moe" in active_rounds(cfg) and cfg["drivers"]["moe"]["source"] == "download", cfg["hub"]["model_repo"], "model", "base/moe/latest.pt"),
    ):
        if not needed:
            continue
        try:
            ok = bool(repo) and api.file_exists(repo, path, repo_type=repo_type, revision=cfg["hub"]["revision"])
            report(ok, f"Hugging Face {repo_type} has {path}", str(repo))
        except Exception as exc:
            report(False, f"Hugging Face {repo_type} has {path}", f"{repo}: {exc}")
    try:
        ok = api.file_exists("facebook/dinov2-small", "config.json")
        report(ok, "DINOv2-small weights reachable", "facebook/dinov2-small (downloaded by the first training)")
    except Exception as exc:
        report(False, "DINOv2-small weights reachable", str(exc))

    # W&B
    if cfg["wandb"]["enabled"]:
        try:
            import wandb

            viewer = wandb.Api(timeout=30).viewer
            teams = list(getattr(viewer, "teams", None) or [])
            entity = cfg["wandb"]["entity"]
            report(True, "W&B login", viewer.username)
            member = entity == viewer.username or entity in teams
            report(member, f"W&B entity {entity!r} is yours or one of your teams",
                   "" if member else f"your teams: {teams}; ask the team owner (chengenwu) for an invitation, or set wandb.entity")
        except Exception as exc:
            report(False, "W&B login", f"{exc} (run `wandb login`, or set wandb.enabled: false)")
    else:
        report(True, "W&B disabled", "metrics only in <work_dir>/results")

    # several machines: Hugging Face login with write access, and the shared repo
    if hub_mode(cfg):
        from vig_dagger import board

        ok, who = board.token_ok()
        report(ok, "Hugging Face login with write access", who)
        if ok:
            try:
                board.get(cfg).ensure_repo()
                visibility = "public" if cfg["sharing"]["public"] else "private"
                report(True, f"shared repo ({visibility}) holds this experiment", cfg["sharing"]["repo"])
            except SystemExit as exc:
                report(False, "shared repo holds this experiment", str(exc))

    # resources the queue will use
    from vig_dagger.plan import sim_slots

    from vig_dagger.jobs import cpu_cores, ram_total_gb

    cores, ram = cpu_cores(), ram_total_gb()
    if cores < (os.cpu_count() or 0) or ram < psutil.virtual_memory().total / 2**30 - 1:
        print(f"container limits: {cores} of the host's {os.cpu_count()} cores, {ram:.0f} of its "
              f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM; the queue uses the limits")
    workers = cfg["dagger"]["workers_per_scorer"]
    workers = max(1, sim_slots(cfg) // len(cfg["dagger"]["scorers"])) if workers == "auto" else int(workers)
    print(f"resources: {cores} CPU cores, {ram:.0f} GB RAM; CPU budget {sim_slots(cfg)} simulator slots, "
          f"{workers} DAgger workers per scorer, at most one training per GPU")
    report(ram >= 64, "RAM", f"{ram:.0f} GB (a training uses ~19 GB, a simulator job ~2.5 GB)", warn_only=True)

    print(f"\n{len(failures)} failed, {len(warnings)} warnings" + (f": fix {failures} before starting" if failures else ""))
    return not failures


# ---------------------------------------------------------------- status


def _tail_step(metrics: Path) -> int | None:
    """The last train/step in a run's metrics.jsonl (reads only the end of the file)."""
    if not metrics.exists():
        return None
    with open(metrics, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 20000))
        lines = fh.read().decode(errors="ignore").splitlines()
    for line in reversed(lines):
        try:
            step = json.loads(line).get("train/step")
            if step is not None:
                return int(step)
        except ValueError:
            continue
    return None


def _progress(cfg: dict, name: str) -> str:
    """What a job has done so far, from its outputs."""
    paths = Paths.of(cfg)
    if name.startswith("train/"):
        from vig_dagger.train import resolve

        out = resolve(cfg, name[len("train/"):])[1]
        step = _tail_step(out / "metrics.jsonl")
        try:
            total = json.loads((out / "run_config.json").read_text())["training"]["max_updates"]
        except (OSError, KeyError, ValueError):
            total = None
        return f"{step or 0}/{total or '?'} updates"
    if name.startswith("dagger_") and not name.startswith(("dagger_prepare", "dagger_dataset")):
        from vig_dagger.workers.dagger import count_successes

        r, s = name[len("dagger_"):].rsplit("_", 2)[:2]
        k, a = count_successes(paths.dagger(r, s))
        return f"collection {r}/{s}: {k}/{cfg['dagger']['num_demos']} successes in {a} attempts"
    return ""


def status(cfg: dict) -> None:
    from vig_dagger import plan

    if hub_mode(cfg):
        board_status(cfg)
        return
    paths = Paths.of(cfg)
    state_path = paths.logs() / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is None:
        print("queue: never started in this work_dir")
        jobs = plan.build(cfg, paths.config())
        states = {j.name: ("done" if j.done is not None and j.done() else "not started") for j in jobs}
    else:
        alive = psutil.pid_exists(state["pid"]) and time.time() - state["heartbeat"] < 120
        age = time.time() - state["heartbeat"]
        states = {name: entry["state"] for name, entry in state["jobs"].items()}
        finished = all(s == "done" for s in states.values())
        label = "RUNNING" if alive else ("FINISHED, every job done" if finished else "NOT RUNNING")
        print(f"queue: {label} (pid {state['pid']}, last heartbeat {age:.0f} s ago)")
    counts: dict[str, int] = {}
    for s in states.values():
        counts[s] = counts.get(s, 0) + 1
    print("jobs: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1])))
    if state is not None:
        for name, entry in state["jobs"].items():
            if entry["state"] == "failed":
                print(f"  FAILED  {name} after {entry['attempts']} attempts -> {paths.logs() / (name + '.log')}")
        for name, entry in state["jobs"].items():
            if entry["state"] == "running":
                hours = (time.time() - entry["started"]) / 3600
                print(f"  running {name} (GPU {entry.get('gpu')}, {hours:.1f} h) {_progress(cfg, name)}")
        blocked = [n for n, e in state["jobs"].items() if e["state"] == "blocked"]
        if blocked:
            print(f"  blocked by failures: {len(blocked)} jobs (e.g. {blocked[0]})")
    # milestones
    from vig_dagger.workers.dagger import count_successes

    n = cfg["dagger"]["num_demos"]
    print("milestones:")
    print(f"  {'base dataset':40s} {'yes' if (paths.dataset('base') / 'dataset_manifest.json').exists() else 'no'}")
    if cfg["split"] != "none":
        print(f"  (this machine runs only round {SPLITS[cfg['split']]}: split {cfg['split']})")
    for r in active_rounds(cfg):
        print(f"  {'round ' + r + ' thresholds':40s} {'yes' if paths.thresholds(r).exists() else 'no'}")
        for s in cfg["dagger"]["scorers"]:
            out = paths.dagger(r, s)
            k, a = count_successes(out) if (out / "episodes").exists() else (0, 0)
            print(f"  {'round ' + r + ' ' + s + ' DAgger':40s} {k}/{n} successes in {a} attempts")
    if cfg["split"] != "round2":  # the round-2 machine has no summary; its results are merged into round 1's
        print(f"  {'evaluation summary':40s} {'yes' if (paths.results() / 'eval_summary.json').exists() else 'no'}")



def board_status(cfg: dict) -> None:
    """Several machines: every task's state in the shared repo, and every machine's last heartbeat."""
    from vig_dagger import board, plan

    b = board.get(cfg)
    b.refresh(force=True)
    tasks = sorted({j.task for j in plan.build(cfg, Paths.of(cfg).config()) if j.task is not None})
    machines = {m["machine"]: m for m in b.machines()}
    silent_after = 3 * float(cfg["sharing"]["heartbeat_s"])  # three heartbeats missed
    groups: dict[str, list[str]] = {"done": [], "running": [], "failed": [], "ready": [], "waiting": []}
    hub = board.HubScheduler(cfg, {j.name: j for j in plan.build(cfg, Paths.of(cfg).config())})
    for t in tasks:
        owner = b.owner(t)
        if b.is_done(t):
            groups["done"].append(t)
        elif b.failed_on(t):
            groups["failed"].append(f"{t} (on {', '.join(b.failed_on(t))})")
        elif owner is not None:
            seen = machines.get(owner, {}).get("time")
            silent = seen is None or time.time() - seen > silent_after
            groups["running"].append(f"{t} on {owner}" + (" -- that machine is SILENT" if silent else ""))
        elif all(b.is_done(n) for n in hub.needs[t]):
            groups["ready"].append(t)
        else:
            groups["waiting"].append(t)
    print(f"shared repo: {b.repo} (this machine: {b.machine})")
    print("tasks: " + ", ".join(f"{len(v)} {k}" for k, v in groups.items()) + f" (of {len(tasks)})")
    for label in ("failed", "running", "ready"):
        for item in groups[label]:
            print(f"  {label.upper() if label == 'failed' else label:8s} {item}")
    print("machines:")
    for name, m in sorted(machines.items()):
        age = (time.time() - m["time"]) / 60
        state = "SILENT" if age * 60 > silent_after else "alive"
        progress = "; ".join(f"{k} {v}" for k, v in (m.get("progress") or {}).items() if v)
        print(f"  {name:32s} {state}, last heartbeat {age:.0f} min ago, {len(m.get('running', []))} jobs running"
              + (f": {progress}" if progress else ""))
    if not b.is_done("summary") and len(groups["done"]) == len(tasks) - 1:
        print("  (only the summary is left)")


# ---------------------------------------------------------------- stop


def stop(cfg: dict, wait: float = 60.0, queue_only: bool = False) -> None:
    """Stop the queue and every job it started (SIGTERM to each job's process group, SIGKILL after ``wait`` s).
    Trainings resume from their last rolling checkpoint, DAgger workers and evaluations from their last finished
    episode, when ``vig-dagger run`` is started again. With ``queue_only`` the jobs keep running and the next
    ``vig-dagger run`` takes them over (e.g. to change ``resources`` settings without interrupting anything)."""
    logs = Paths.of(cfg).logs()
    targets: list[tuple[str, int]] = []
    queue_pid = logs / "queue.pid"
    if queue_pid.exists():
        try:
            pid = int(queue_pid.read_text())
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGTERM)
                print(f"stopped the queue (pid {pid})")
        except (ValueError, ProcessLookupError):
            pass
    if queue_only:
        return
    for path in sorted((logs / "pids").glob("**/*.pid")) if (logs / "pids").exists() else []:
        try:
            pid = int(path.read_text().split()[0])
            if psutil.pid_exists(pid):
                os.killpg(pid, signal.SIGTERM)
                targets.append((str(path.relative_to(logs / "pids"))[:-4], pid))
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    print(f"sent SIGTERM to {len(targets)} job(s): {', '.join(name for name, _ in targets) or 'none'}")
    deadline = time.time() + wait
    while time.time() < deadline and any(psutil.pid_exists(pid) for _, pid in targets):
        time.sleep(2)
    for name, pid in targets:
        if psutil.pid_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
                print(f"killed {name} (pid {pid}) after {wait:.0f} s")
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------- merge


EXPERIMENT_KEYS = ("demos", "training", "rounds", "drivers", "scorers", "thresholds", "dagger", "retrain", "evaluation")


def merge(cfg: dict, source: Path) -> None:
    """Bring round ``per_resolution``'s results from the round-2 machine into this (round-1) work directory, then
    refresh the evaluation summary and the report.

    ``source`` is that machine's work directory, or a copy of it: its ``config.json``, ``eval/``, ``results/`` and
    ``policies/`` are enough (checkpoints are copied too if they are there, but are not needed). Only round 2's own
    outputs are copied: the evaluations of ``res2`` ... ``res64`` and of the ``base_plus_per_resolution_*``
    retrainings, the DAgger metrics, and the training logs; round 2's own ``base`` seed 0 stays behind (this
    machine's is the evaluated one). Copying again replaces what an earlier merge copied.
    """
    import shutil

    from vig_dagger import steps

    paths, src = Paths.of(cfg), Path(source).expanduser().resolve()
    if not (src / "config.json").is_file():
        raise SystemExit(f"{src} has no config.json; give the round-2 machine's work directory (or a copy of it)")
    other = json.loads((src / "config.json").read_text())
    if other.get("split") != "round2":
        raise SystemExit(f"{src} was not run with split: round2 (its split is {other.get('split', 'none')!r})")
    if src == paths.root:
        raise SystemExit("the source is this work directory; give the round-2 machine's work directory")
    def experiment(c: dict) -> dict:  # the settings that define the experiment (DAgger worker counts don't)
        out = {k: c.get(k) for k in EXPERIMENT_KEYS}
        out["dagger"] = {k: v for k, v in (out["dagger"] or {}).items() if k != "workers_per_scorer"}
        return out

    mine, theirs = experiment(cfg), experiment(other)
    differ = [k for k in EXPERIMENT_KEYS if mine[k] != theirs[k]]
    if differ:
        raise SystemExit(f"the two machines ran different experiments (settings differ in {differ}); not merging")

    copied: list[str] = []

    def copy_tree(rel: str) -> None:
        if (src / rel).is_dir():
            shutil.copytree(src / rel, paths.root / rel, dirs_exist_ok=True)
            copied.append(rel + "/")

    def copy_file(rel: str) -> None:
        if (src / rel).is_file():
            (paths.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, paths.root / rel)
            copied.append(rel)

    round2_sets = [name for name in training_sets(cfg, ["per_resolution"]) if "per_resolution" in name]
    copy_tree("eval/driver/per_resolution")
    for name in round2_sets:
        copy_tree(f"eval/retrain/{name}")
    for path in sorted((src / "results").glob("dagger_per_resolution*")) + sorted((src / "results" / "figures").glob("dagger_per_resolution*")):
        copy_file(str(path.relative_to(src)))
    run_dirs = [d for d in sorted((src / "policies" / "drivers" / "per_resolution").glob("res*")) if d.is_dir()]
    run_dirs += [d for name in round2_sets for d in sorted((src / "policies" / "retrain" / name).glob("seed*"))]
    for d in run_dirs:
        for f in ("run_config.json", "metrics.jsonl", "latest.pt"):
            copy_file(str((d / f).relative_to(src)))
    for f in ("thresholds.json", "demo_scores.npz"):
        copy_file(f"policies/drivers/per_resolution/{f}")
    if not copied:
        raise SystemExit(f"nothing of round per_resolution found in {src}")
    print(f"[merge] copied from {src}:\n  " + "\n  ".join(c for c in copied if not c.endswith("latest.pt"))
          + f"\n  ({sum(c.endswith('latest.pt') for c in copied)} checkpoints)")
    steps.summarize(cfg)
    print(f"[merge] refreshed {paths.results() / 'eval_summary.json'} and {paths.results() / 'REPORT.md'}")
