"""Operating an experiment: ``vig-dagger check`` (preflight), ``status`` and ``stop``."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import psutil

from vig_dagger.config import Paths, board_mode, collections, downloaded

COLLECTION_GB = 15  # ~30 MB per saved successful episode x 500
BASE_GB = 35  # downloaded demos ~26 GB, the converted dataset, checkpoints, logs


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
    n = len(collections(cfg))
    print(f"experiment {cfg['name']}, work_dir {paths.root}: MoE seeds {cfg['moe']['seeds']}, {n} DAgger collections "
          f"(scorers {cfg['dagger']['scorers']}, thresholds {cfg['dagger']['thresholds']}, variants {cfg['dagger']['variants']})")

    # disk
    paths.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.root).free / 2**30
    need = BASE_GB + COLLECTION_GB * n
    report(os.access(paths.root, os.W_OK), "work_dir is writable", str(paths.root))
    report(free >= need, "free disk in work_dir",
           f"{free:.0f} GB (every task on this machine: about {need} GB; {COLLECTION_GB} GB per collection, {BASE_GB} GB for "
           "the demos and checkpoints)", warn_only=free >= BASE_GB + 2 * COLLECTION_GB)

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
    checks = [(cfg["demos"]["source"] == "download", cfg["hub"]["dataset_repo"], "dataset", "raw/planner/part_000.h5")]
    checks += [(True, downloaded(cfg, int(s)), "model", "base/moe/latest.pt") for s in cfg["moe"]["seeds"] if downloaded(cfg, int(s))]
    for needed, repo, repo_type, path in checks:
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

    # the task board (several machines)
    if board_mode(cfg):
        from vig_dagger import board

        ok, detail = board.check(cfg)
        report(ok, "task board reachable and writable (ssh, rsync)", detail)
        if ok:
            try:
                board.get(cfg).ensure()
                report(True, "task board holds this experiment", "same settings as the other machines (or the first one)")
            except SystemExit as exc:
                report(False, "task board holds this experiment", str(exc))
    else:
        report(True, "no task board", "this machine runs every task itself (or tasks given with --tasks)")

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

    # resources the queue will use
    from vig_dagger.plan import sim_slots, workers_per_collection

    cores, ram = os.cpu_count() or 0, psutil.virtual_memory().total / 2**30
    print(f"resources: {cores} CPU cores, {ram:.0f} GB RAM; CPU budget {sim_slots(cfg)} simulator slots, "
          f"{workers_per_collection(cfg)} DAgger workers per collection, at most one training per GPU")
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
    if name.startswith("dagger_") and not name.startswith("dagger_prepare"):
        from vig_dagger.workers.dagger import count_successes

        collection = name[len("dagger_"):].rsplit("_", 1)[0]
        k, a = count_successes(paths.dagger(collection))
        return f"{k}/{cfg['dagger']['num_demos']} successes in {a} attempts"
    return ""


def board_status(cfg: dict) -> None:
    """Several machines: every task's state on the board, and every machine's last heartbeat."""
    from vig_dagger import board, plan
    from vig_dagger import tasks as task_list

    b = board.get(cfg)
    b.refresh(force=True)
    jobs = plan.build(cfg, Paths.of(cfg).config())
    names = task_list.numbered(cfg, jobs)
    number = {t: i + 1 for i, t in enumerate(names)}
    machines = {m["machine"]: m for m in b.machines()}
    silent_after = 3 * float(cfg["sharing"]["heartbeat_s"])  # three heartbeats missed
    scheduler = board.Scheduler(cfg, {j.name: j for j in jobs})
    groups: dict[str, list[str]] = {"done": [], "running": [], "failed": [], "ready": [], "waiting": []}
    for t in names:
        label = f"{number[t]} {t}"
        owner = b.owner(t)
        if b.is_done(t):
            groups["done"].append(label)
        elif b.failed_on(t):
            groups["failed"].append(f"{label} (on {', '.join(b.failed_on(t))})")
        elif owner is not None:
            seen = machines.get(owner, {}).get("time")
            silent = seen is None or time.time() - seen > silent_after
            groups["running"].append(f"{label} on {owner}" + (" -- that machine is SILENT" if silent else ""))
        elif all(b.is_done(n) for n in scheduler.needs[t]):
            groups["ready"].append(label)
        else:
            groups["waiting"].append(label)
    print(f"task board: {b.spec} (this machine: {b.machine})")
    print("tasks: " + ", ".join(f"{len(v)} {k}" for k, v in groups.items()) + f" (of {len(names)})")
    for label in ("failed", "running", "ready"):
        for item in groups[label]:
            print(f"  {label.upper() if label == 'failed' else label:8s} {item}")
    print("machines:")
    for name, m in sorted(machines.items()):
        age = (time.time() - m["time"]) / 60
        state = "SILENT" if age * 60 > silent_after else "alive"
        progress = "; ".join(f"{k}: {v}" for k, v in (m.get("progress") or {}).items() if v and k.startswith(("train/", "dagger_")))
        print(f"  {name:32s} {state}, last heartbeat {age:.0f} min ago, tasks {', '.join(m.get('tasks', [])) or 'none'}"
              + (f"; {progress}" if progress else ""))
    if b.is_done("summary"):
        print(f"the summary is done: {b.location('files/results')}REPORT.md (or `vig-dagger report` here)")


def status(cfg: dict) -> None:
    from vig_dagger import plan

    paths = Paths.of(cfg)
    if board_mode(cfg):
        board_status(cfg)
        print("---- this machine")
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
    from vig_dagger import tasks as task_list
    from vig_dagger.workers.dagger import count_successes

    n = cfg["dagger"]["num_demos"]
    print("milestones:")
    print(f"  {'base dataset (for the MoEs trained here)':48s} {'yes' if (paths.dataset('base') / 'dataset_manifest.json').exists() else 'no'}")
    for seed in (int(s) for s in cfg["moe"]["seeds"]):
        moe = "yes" if paths.moe_checkpoint(seed).exists() else "no"
        print(f"  {'MoE seed ' + str(seed) + ' / its thresholds':48s} {moe} / {'yes' if paths.thresholds(seed).exists() else 'no'}")
    for c in collections(cfg):
        out = paths.dagger(c)
        if not (out / "episodes").exists():
            continue
        k, a = count_successes(out)
        done = " (analyzed)" if paths.collection_results(c).with_suffix(".json").exists() else ""
        print(f"  {c.id:48s} {k}/{n} successes in {a} attempts{done}")
    print(f"  {'summary':48s} {'yes' if (paths.results() / 'summary.json').exists() else 'no'}")
    if not board_mode(cfg):
        jobs = plan.build(cfg, paths.config())
        names = task_list.numbered(cfg, jobs)
        here = [i + 1 for i, t in enumerate(names) if task_list.present(cfg, t)]
        print(f"tasks done in this work_dir: {task_list._ranges(here) if here else 'none'} (of 1–{len(names)}; `vig-dagger tasks`)")


# ---------------------------------------------------------------- stop


def stop(cfg: dict, wait: float = 60.0, queue_only: bool = False) -> None:
    """Stop the queue and every job it started (SIGTERM to each job's process group, SIGKILL after ``wait`` s).
    Trainings resume from their last rolling checkpoint, DAgger workers from their last finished
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
