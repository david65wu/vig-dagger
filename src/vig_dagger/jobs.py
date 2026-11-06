"""A queue of worker processes with dependencies, started when the machine has room for them.

Every job is one Python module invocation (``python -m vig_dagger.<module> ...``, with the queue's interpreter or the
job's own, e.g. the Isaac environment's) with a log file. A job starts once the jobs it depends on are done and the
machine has room for it:

- CPU: simulator and conversion jobs take ``slots`` from the CPU budget (``resources.sim_slots``, default one slot
  per 3 cores; an Isaac simulator job uses about 1.5 cores), a training takes ``resources.train_slots`` for its data
  loading; nothing new starts while the 1-minute load average is above ``resources.max_load`` (the machine may be
  shared).
- RAM: the available memory must cover the job's estimate.
- Isaac Sim: simulator jobs start at least ``resources.sim_start_gap_s`` seconds apart (Isaac Sim processes that
  start at the same moment can crash while creating the renderer).
- GPU: a simulator job (rendering, policy inference) goes to the GPU with the most free memory, if that has at
  least ``resources.sim_gpu_memory_gb``; a training needs ``resources.train_gpu_memory_gb`` free on a GPU that runs
  none of our other trainings.

Jobs started in the last few minutes count with their estimates (they may not have allocated memory yet). Among
jobs ready at the same time, lower ``priority`` numbers start first. A finished job counts as done only if its
``done`` check passes; otherwise it is retried up to ``retries`` times (every worker continues where it stopped).

With several machines (``sharing.mode: hub``) every machine holds the whole graph, but runs only the jobs of the
tasks it claimed (``board.HubScheduler``) and the preparation they need; another machine's tasks count as done when
their outputs are in the shared repo. The queue then runs until every task of the experiment is done.

Each running job's PID is kept in ``<work_dir>/logs/pids/<job>.pid``. If the queue itself is restarted while jobs
are still running (e.g. after the SSH session dropped), it adopts them instead of starting them twice. Only one
queue may run per work directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import psutil

RECENT_SECONDS = 180  # a job started this recently counts with its resource estimate


def exit_on_sigterm() -> None:
    """Workers call this first: SIGTERM (``vig-dagger stop``) raises SystemExit, so ``finally`` blocks close their
    files (an HDF5 file left open could be unreadable) before the process exits."""
    import signal

    def handler(signum, frame):
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, handler)


@dataclass
class Job:
    name: str
    args: list[str]  # arguments after ``python -m``
    kind: str = "sim"  # sim (simulator / GPU inference) | train (a training) | cpu (no GPU)
    deps: list[str] = field(default_factory=list)
    priority: int = 50
    slots: int = 1  # CPU budget units (sim and cpu jobs)
    retries: int = 0
    done: Callable[[], bool] | None = None
    env: dict[str, str] = field(default_factory=dict)
    python: str | None = None  # the interpreter (default: the queue's)
    task: str | None = None  # several machines: the task the job belongs to (board.py); None for local preparation


@dataclass
class Running:
    job: Job
    pid: int
    gpu: int | None
    started: float
    proc: subprocess.Popen | None = None  # None for a job adopted from an earlier queue
    log: object = None


class Machine:
    """What the machine has free: GPUs (nvidia-smi, cached; a hung nvidia-smi keeps the last answer), RAM, load."""

    def __init__(self, gpus: list[int] | str) -> None:
        self._gpus = gpus
        self._cache: tuple[float, dict[int, tuple[float, float]]] = (0.0, {})

    def gpu_memory(self) -> dict[int, tuple[float, float]]:
        """{gpu: (total GB, used GB)}."""
        now = time.time()
        if now - self._cache[0] < 15 and self._cache[1]:
            return self._cache[1]
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
            info = {}
            for line in out.strip().splitlines():
                index, total, used = (x.strip() for x in line.split(","))
                info[int(index)] = (float(total) / 1024, float(used) / 1024)
            if self._gpus != "auto":
                info = {g: v for g, v in info.items() if g in set(self._gpus)}
            self._cache = (now, info)
        except Exception as exc:  # keep the last answer
            print(f"[queue] nvidia-smi failed ({exc}); using the previous reading", flush=True)
            self._cache = (now, self._cache[1])
        return self._cache[1]


class Queue:
    def __init__(self, jobs: list[Job], resources: dict, log_dir: Path, poll: float = 10.0, hub=None) -> None:
        self.jobs = {job.name: job for job in jobs}
        missing = {d for job in jobs for d in job.deps if d not in self.jobs}
        if missing:
            raise ValueError(f"jobs depend on unknown jobs: {sorted(missing)}")
        self.r = resources
        cores = os.cpu_count() or 8
        self.sim_slots = cores // 3 if resources["sim_slots"] == "auto" else int(resources["sim_slots"])
        self.max_load = float(cores if resources["max_load"] == "auto" else resources["max_load"])
        self.machine = Machine(resources["gpus"])
        self.log_dir, self.pid_dir = log_dir, log_dir / "pids"
        self.poll = poll
        self.sim_start_gap = float(resources.get("sim_start_gap_s", 0))
        self.last_sim_start = 0.0
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.running: dict[str, Running] = {}
        self.attempts: dict[str, int] = {}
        # --only may select a subgraph; pauses outside it remain in the config
        # for the next full run but do not affect this queue.
        self.paused = set(resources.get("paused_jobs", [])) & self.jobs.keys()
        self.hub = hub(self.jobs) if hub is not None else None  # a board.HubScheduler, built on the job table

    # ---- bookkeeping
    def _is_done(self, job: Job) -> bool:
        return job.done is not None and job.done()

    def _pid_file(self, job: Job) -> Path:
        return self.pid_dir / f"{job.name}.pid"

    def _adopt(self) -> None:
        """Running jobs of an earlier queue (their PID file names a live process running our module)."""
        for job in self.jobs.values():
            path = self._pid_file(job)
            if not path.exists():
                continue
            try:
                pid, gpu = (int(x) if x != "None" else None for x in path.read_text().split())
                proc = psutil.Process(pid)
                if proc.is_running() and job.args[0] in " ".join(proc.cmdline()):
                    self.running[job.name] = Running(job, pid, gpu, time.time() - RECENT_SECONDS)
                    print(f"[queue] adopted running {job.name} (pid {pid})", flush=True)
                    continue
            except (ValueError, psutil.Error):
                pass
            path.unlink(missing_ok=True)

    # ---- resources
    def _cpu_used(self) -> int:
        return sum(self.r["train_slots"] if run.job.kind == "train" else run.job.slots for run in self.running.values())

    def _reserved(self, key: str, gpu: int | None = None) -> float:
        """Estimates (``ram_gb``, or ``gpu_memory_gb`` on one GPU) of recently started jobs."""
        now, total = time.time(), 0.0
        for run in self.running.values():
            if now - run.started > RECENT_SECONDS:
                continue
            if key == "gpu_memory_gb" and (run.job.kind == "cpu" or run.gpu != gpu):
                continue
            total += float(self.r[f"{'train' if run.job.kind == 'train' else 'sim'}_{key}"])
        return total

    def place(self, job: Job) -> int | None | bool:
        """The GPU the job would get now (None: a CPU job), or False when the machine has no room for it."""
        return self._place(job)

    def _place(self, job: Job) -> int | None | bool:
        """The GPU for the job (None for CPU jobs), or False when the machine has no room for it now."""
        slots = self.r["train_slots"] if job.kind == "train" else job.slots
        # With nothing of ours running, start anyway: other users' load must not stall the queue.
        if self.running and (self._cpu_used() + slots > self.sim_slots or os.getloadavg()[0] > self.max_load):
            return False
        ram = self.r["train_ram_gb" if job.kind == "train" else "sim_ram_gb"]
        if psutil.virtual_memory().available / 2**30 - self._reserved("ram_gb") < ram:
            return False
        if job.kind == "cpu":
            return None
        gpus = self.machine.gpu_memory()
        free = {g: total - used - self._reserved("gpu_memory_gb", g) for g, (total, used) in gpus.items()}
        if job.kind == "train":
            ours = {run.gpu for run in self.running.values() if run.job.kind == "train"}
            options = [g for g in free if g not in ours and free[g] >= self.r["train_gpu_memory_gb"]]
        else:
            options = [g for g in free if free[g] >= self.r["sim_gpu_memory_gb"]]
        return max(options, key=lambda g: free[g]) if options else False

    # ---- running
    def _start(self, job: Job, gpu: int | None) -> None:
        log_path = self.log_dir / f"{job.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a")
        python = job.python or sys.executable
        log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} start {job.name} on GPU {gpu}: {python} -m {' '.join(job.args)}\n")
        log.flush()
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "4")  # dozens of spinning OpenMP threads per worker slow a shared machine
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        env.update(job.env)
        proc = subprocess.Popen([python, "-m", *job.args], stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        self._pid_file(job).parent.mkdir(parents=True, exist_ok=True)
        self._pid_file(job).write_text(f"{proc.pid} {gpu}")
        self.running[job.name] = Running(job, proc.pid, gpu, time.time(), proc, log)
        print(f"[queue] {time.strftime('%H:%M:%S')} started {job.name} on {'GPU ' + str(gpu) if gpu is not None else 'CPU'} "
              f"({len(self.running)} running) -> {log_path}", flush=True)

    def _finished(self, run: Running) -> int | None:
        """The exit code of a finished job, 0 for a finished adopted job (judged by its done check), None if running."""
        if run.proc is not None:
            return run.proc.poll()
        try:
            return None if psutil.Process(run.pid).status() != psutil.STATUS_ZOMBIE else 0
        except psutil.NoSuchProcess:
            return 0

    def _reap(self) -> None:
        for name, run in list(self.running.items()):
            code = self._finished(run)
            if code is None:
                continue
            del self.running[name]
            if run.log is not None:
                run.log.close()
            self._pid_file(run.job).unlink(missing_ok=True)
            if code == 0 and (run.job.done is None or run.job.done()):
                self.done.add(name)
                print(f"[queue] {time.strftime('%H:%M:%S')} finished {name}", flush=True)
                continue
            self.attempts[name] = self.attempts.get(name, 0) + 1
            if self.attempts[name] <= run.job.retries:
                print(f"[queue] {name} exited with {code}; retry {self.attempts[name]}/{run.job.retries} "
                      f"(log {self.log_dir / (name + '.log')})", flush=True)
            else:
                self.failed.add(name)
                print(f"[queue] FAILED {name} (exit {code}); see {self.log_dir / (name + '.log')}", flush=True)

    def write_state(self) -> None:
        """``logs/queue_state.json``: the queue's heartbeat and every job's state (read by ``vig-dagger status``)."""
        import json

        jobs = {}
        for name, job in self.jobs.items():
            if name in self.done:
                state = "done"
            elif name in self.running:
                state = "running"
            elif name in self.failed:
                state = "failed"
            elif name in self.paused:
                state = "paused"
            elif self.blocked(job):
                state = "blocked"
            else:
                state = "ready" if all(d in self.done for d in job.deps) else "waiting"
            entry = {"state": state, "attempts": self.attempts.get(name, 0)}
            if name in self.running:
                run = self.running[name]
                entry.update(pid=run.pid, gpu=run.gpu, started=run.started)
            jobs[name] = entry
        path = self.log_dir / "queue_state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"heartbeat": time.time(), "pid": os.getpid(), "jobs": jobs}))
        tmp.replace(path)

    def blocked(self, job: Job) -> bool:
        """A job that can never run because something it depends on failed."""
        # Shared phase dependencies form a DAG; visit each ancestor once rather
        # than recursively expanding the same subgraph for every path.
        pending, seen = list(job.deps), set()
        while pending:
            name = pending.pop()
            if name in self.failed:
                return True
            if name not in seen and name in self.jobs:
                seen.add(name)
                pending.extend(self.jobs[name].deps)
        return False

    def run(self) -> tuple[set[str], set[str]]:
        """Run every job; returns (done, failed or blocked)."""
        for job in self.jobs.values():
            if self._is_done(job):
                self.done.add(job.name)
        self._adopt()
        print(f"[queue] {len(self.jobs)} jobs: {len(self.done)} done, {len(self.running)} running; "
              f"CPU budget {self.sim_slots} slots, max load {self.max_load:.0f}", flush=True)
        waiting_for_room = False
        while True:
            self._reap()
            waiting = [j for j in self.jobs.values()
                       if j.name not in self.done and j.name not in self.failed and j.name not in self.running and not self.blocked(j)]
            if self.hub is not None:  # several machines: only this machine's tasks and what they need
                self.hub.tick(self, waiting_for_room)
                runnable, needed = self.hub.eligible()
                for job in waiting:  # another machine's task finished: its outputs are in the shared repo
                    if job.name in needed and job.name not in runnable and self._is_done(job):
                        self.done.add(job.name)
                waiting = [j for j in waiting if j.name in runnable and j.name not in self.done]
                if self.hub.all_done() and not self.running:
                    break
            elif not waiting and not self.running:
                break
            ready = sorted((j for j in waiting if j.name not in self.paused
                            and all(d in self.done for d in j.deps)), key=lambda j: j.priority)
            waiting_for_room = False
            for job in ready:
                if self._is_done(job):  # e.g. finished by an adopted run of a sibling (a DAgger target reached)
                    self.done.add(job.name)
                    continue
                isaac = job.python is not None  # a simulator job in the Isaac environment
                if isaac and time.time() - self.last_sim_start < self.sim_start_gap:
                    waiting_for_room = True
                    continue
                gpu = self._place(job)
                if gpu is False:
                    waiting_for_room = True
                    continue
                self._start(job, gpu)
                if isaac:
                    self.last_sim_start = time.time()
            self.write_state()
            time.sleep(self.poll)
        self.write_state()
        stuck = self.failed | {j.name for j in self.jobs.values() if self.blocked(j)}
        return self.done, stuck


def lock(work_dir: Path) -> None:
    """Refuse to start a second queue on the same work directory."""
    path = work_dir / "logs" / "queue.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            pid = int(path.read_text())
            if psutil.pid_exists(pid) and pid != os.getpid() and "vig" in " ".join(psutil.Process(pid).cmdline()):
                raise SystemExit(f"another queue (pid {pid}) is running on {work_dir}; stop it first")
        except (ValueError, psutil.Error):
            pass
    path.write_text(str(os.getpid()))
