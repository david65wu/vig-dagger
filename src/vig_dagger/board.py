"""Several machines sharing one task board (``sharing.board``): the machines claim the experiment's tasks themselves.

Every machine runs the same config and the same command (``vig-dagger run``); however many machines there are, they
divide the work among them. The work comes in the numbered tasks of ``tasks.py``:

  moe/sd<s>        a MoE (downloaded or trained) and its thresholds
  dagger/<c>       one DAgger collection and its metrics
  summary          every collection side by side and REPORT.md, once every collection is done

The rest (downloading and converting the demos) is local preparation that a machine does when one of its tasks needs
it. A machine with room claims the most urgent task whose inputs are on the board, fetches those inputs
(``fetch/<task>`` jobs), runs the task's jobs in its own queue, and publishes the outputs (``publish/<task>``). A
task nobody can start yet waits until its inputs are published; a machine with nothing to claim waits too, and every
machine's queue ends when the summary is done.

**The board** is a directory, ``sharing.board``: ``host:/path`` on one machine (reached with ``ssh`` and ``rsync``
from the others: they need passwordless ssh to that host), or ``/path`` on a filesystem all machines mount. The
machine that hosts it can use either form.

    experiment.json                  the experiment's settings: a machine with other settings refuses to join
    tasks/<key>/claim/machine        who runs the task. The claim is an ``mkdir``, which succeeds for one machine only
    tasks/<key>/done.json            written after the task's outputs are in files/
    tasks/<key>/failed/<machine>     the task failed there (``vig-dagger release`` frees it for another machine)
    machines/<machine>.json          heartbeat: what the machine runs (every sharing.heartbeat_s)
    files/<work_dir-relative path>   the outputs other machines need: MoE checkpoints (weights-only) and thresholds,
                                     every collection's metrics, the final results/

``<key>`` is the task name with ``/`` replaced by ``__``; ``<machine>`` is the host name plus a short hash of the work
directory (``machine_id``), so two work directories on one host are two machines. A collection's DAgger episodes
(``dagger/<c>/``, ~15 GB) stay on the machine that collected them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from vig_dagger.config import Paths

EXPERIMENT_KEYS = ("demos", "training", "moe", "scorers", "thresholds", "dagger")
MIN_FREE_GB = 25  # a machine claims no new collection with less free disk in its work directory (one needs ~15 GB)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


def key(task: str) -> str:
    return task.replace("/", "__")


def machine_id(cfg: dict) -> str:
    """This machine's name on the board (kept in <work_dir>/machine_id)."""
    path = Paths.of(cfg).root / "machine_id"
    if path.is_file():
        return path.read_text().strip()
    root = str(Paths.of(cfg).root.resolve())
    name = f"{socket.gethostname()}-{hashlib.sha1(root.encode()).hexdigest()[:6]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return name


def experiment(cfg: dict) -> dict:
    """The settings that define the experiment (worker counts don't)."""
    out = {k: cfg.get(k) for k in EXPERIMENT_KEYS}
    out["dagger"] = {k: v for k, v in (out["dagger"] or {}).items() if k != "workers_per_collection"}
    return json.loads(json.dumps(out))


# ---------------------------------------------------------------- what a task publishes and what others fetch


def outputs(cfg: dict, task: str) -> tuple[list[str], list[str]]:
    """(exports, archive): glob patterns relative to the work directory (``dir/**``: everything below dir). Exports are
    what other tasks need and fetch; the archive is published for the record only."""
    kind, _, rest = task.partition("/")
    if kind == "moe":
        run = f"policies/moe/{rest}"
        return [f"{run}/latest.pt", f"{run}/run_config.json", f"{run}/train_config.json", f"{run}/metrics.jsonl",
                f"{run}/thresholds.json", f"{run}/demo_scores.npz"], []
    if kind == "dagger":
        archive = [f"dagger/{rest}/settings.json", f"dagger/{rest}/episodes/*", f"dagger/{rest}/attempts/*"]
        if cfg["sharing"]["publish_trajectories"]:
            archive.append(f"dagger/{rest}/trajectories/*")
        return [f"results/collections/{rest}.json", f"results/collections/{rest}_*"], archive
    if task == "summary":
        return ["config.json", "results/*", "results/figures/**"], []
    raise ValueError(f"unknown task {task!r}")


def exports_present(cfg: dict, task: str) -> bool:
    """The task's outputs that other tasks need are in this work directory."""
    paths = Paths.of(cfg)
    kind, _, rest = task.partition("/")
    if kind == "moe":
        seed = int(rest[2:])
        return paths.moe_checkpoint(seed).is_file() and paths.thresholds(seed).is_file()
    if kind == "dagger":
        return paths.collection_results(rest).with_suffix(".json").is_file()
    if task == "summary":
        return (paths.results() / "REPORT.md").is_file() and (paths.results() / "summary.json").is_file()
    raise ValueError(f"unknown task {task!r}")


def _local_files(root: Path, patterns: list[str]) -> list[Path]:
    files = set()
    for pattern in patterns:
        found = (root / pattern[:-3]).rglob("*") if pattern.endswith("/**") else root.glob(pattern)
        files.update(p for p in found if p.is_file())
    return sorted(files)


def _rsync_filters(patterns: list[str]) -> list[str]:
    """rsync include rules for work_dir-relative glob patterns (and every directory, pruned when empty)."""
    rules = ["--include=*/"]
    for pattern in patterns:
        rules.append(f"--include=/{pattern}")
    return rules + ["--exclude=*", "--prune-empty-dirs"]


# ---------------------------------------------------------------- the board


def _is_this_machine(host: str) -> bool:
    """Whether ``host`` (a name, an ``ssh`` alias or an address, optionally ``user@host``) is this machine."""
    import psutil

    name = host.rsplit("@", 1)[-1]
    if name in ("localhost", socket.gethostname(), socket.getfqdn()):
        return True
    try:  # an ssh alias may stand for another name: ask ssh what it resolves to
        out = subprocess.run(["ssh", "-G", name], capture_output=True, text=True, timeout=10).stdout
        name = next((line.split()[1] for line in out.splitlines() if line.startswith("hostname ")), name)
        address = socket.gethostbyname(name)
    except (OSError, subprocess.SubprocessError):
        return False
    mine = {a.address for addrs in psutil.net_if_addrs().values() for a in addrs}
    return address in mine or address.startswith("127.")


class Board:
    """The board directory: shell commands run there (locally, or with ssh on its host), files move with rsync."""

    def __init__(self, cfg: dict) -> None:
        spec = str(cfg["sharing"]["board"])
        host, path = ("", spec)
        if ":" in spec and not spec.startswith("/"):
            host, _, path = spec.partition(":")
        if host and _is_this_machine(host):  # the board's own host uses the directory directly
            host = ""
        self.cfg, self.spec, self.host = cfg, spec, host
        self.path = path.rstrip("/") if host else str(Path(path).expanduser())
        self.machine = machine_id(cfg)
        self.poll = float(cfg["sharing"]["poll_s"])
        self.state: dict[str, dict] = {}  # key -> {"owner": str | None, "done": bool, "failed": [machines]}
        self.read_at = 0.0

    # ---- transport
    def sh(self, script: str, input: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell script in the board's place (with ``B`` set to the board directory)."""
        full = f"B={self.path}; {script}"
        cmd = [*SSH, self.host, full] if self.host else ["sh", "-c", full]
        result = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=600)
        if check and result.returncode != 0:
            raise RuntimeError(f"board command failed ({result.returncode}): {result.stderr.strip()[-500:]}")
        return result

    def location(self, rel: str = "") -> str:
        where = f"{self.path}/{rel}".rstrip("/") + "/"
        return f"{self.host}:{where}" if self.host else where

    def _rsync(self, args: list[str]) -> None:
        cmd = ["rsync", "-a", *(["-e", " ".join(SSH)] if self.host else []), *args]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
        if result.returncode != 0:
            raise RuntimeError(f"rsync failed ({result.returncode}): {result.stderr.strip()[-500:]}")

    def push(self, root: Path, files: list[Path]) -> None:
        """Copy files (under root) to files/ on the board, at their root-relative paths."""
        if not files:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(str(f.relative_to(root)) for f in files) + "\n")
            listing = fh.name
        try:
            self.sh('mkdir -p "$B/files"')
            self._rsync([f"--files-from={listing}", f"{root}/", self.location("files")])
        finally:
            os.unlink(listing)

    def pull(self, patterns: list[str], root: Path) -> None:
        """Copy the files matching the patterns from files/ on the board into root."""
        root.mkdir(parents=True, exist_ok=True)
        self._rsync([*_rsync_filters(patterns), self.location("files"), f"{root}/"])

    # ---- reading
    def refresh(self, force: bool = False) -> None:
        """Read every task's state (at most every sharing.poll_s seconds unless forced). An unforced read that fails
        keeps the last state; a forced one (before a claim or a publish) raises."""
        if not force and time.time() - self.read_at < self.poll:
            return
        script = ('cd "$B/tasks" 2>/dev/null || exit 0; for t in *; do [ -d "$t" ] || continue; '
                  '[ -f "$t/done.json" ] && echo "done $t"; '
                  '[ -d "$t/claim" ] && echo "claim $t $(cat "$t/claim/machine" 2>/dev/null)"; '
                  'for f in "$t"/failed/*; do [ -e "$f" ] && echo "failed $t ${f##*/}"; done; done; true')
        try:
            out = self.sh(script).stdout
        except Exception as exc:
            if force or not self.read_at:
                raise
            print(f"[board] could not read {self.spec} ({exc}); using the state of "
                  f"{(time.time() - self.read_at) / 60:.0f} min ago", flush=True)
            self.read_at = time.time()
            return
        state: dict[str, dict] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            entry = state.setdefault(parts[1], {"owner": None, "done": False, "failed": []})
            if parts[0] == "done":
                entry["done"] = True
            elif parts[0] == "claim":
                entry["owner"] = parts[2] if len(parts) > 2 else "?"  # "?": claimed a moment ago, name not yet written
            elif parts[0] == "failed":
                entry["failed"].append(parts[2])
        self.state, self.read_at = state, time.time()

    def owner(self, task: str) -> str | None:
        return self.state.get(key(task), {}).get("owner")

    def is_done(self, task: str) -> bool:
        return bool(self.state.get(key(task), {}).get("done"))

    def failed_on(self, task: str) -> list[str]:
        return sorted(self.state.get(key(task), {}).get("failed", []))

    def mine(self) -> set[str]:
        return {k.replace("__", "/") for k, v in self.state.items() if v["owner"] == self.machine}

    def machines(self) -> list[dict]:
        """Every machine's last heartbeat."""
        out = self.sh('for f in "$B"/machines/*.json; do [ -f "$f" ] && { cat "$f"; echo; }; done; true').stdout
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    # ---- writing
    def ensure(self) -> None:
        """Create the board if needed and check this machine runs the same experiment as the others."""
        mine = json.dumps(experiment(self.cfg), indent=1, sort_keys=True)
        script = ('mkdir -p "$B/tasks" "$B/machines" "$B/files" && t="$B/.experiment.$$" && cat > "$t" && '
                  '{ ln "$t" "$B/experiment.json" 2>/dev/null; rm -f "$t"; cat "$B/experiment.json"; }')
        theirs = json.loads(self.sh(script, input=mine).stdout)
        differ = sorted(k for k in EXPERIMENT_KEYS if theirs.get(k) != json.loads(mine)[k])
        if differ:
            raise SystemExit(f"the board {self.spec} holds an experiment with other settings ({differ}); use the same "
                             "config as the other machines, or another sharing.board")

    def claim(self, task: str) -> bool:
        """Claim the task for this machine; False if another machine has it (or it is done)."""
        k = key(task)
        self.refresh(force=True)
        if self.owner(task) is not None or self.is_done(task):
            return self.owner(task) == self.machine
        result = self.sh(f'mkdir -p "$B/tasks/{k}" && mkdir "$B/tasks/{k}/claim" 2>/dev/null && '
                         f'echo {shlex.quote(self.machine)} > "$B/tasks/{k}/claim/machine"', check=False)
        self.refresh(force=True)
        return result.returncode == 0 and self.owner(task) == self.machine

    def mark_failed(self, task: str) -> None:
        k = key(task)
        self.sh(f'mkdir -p "$B/tasks/{k}/failed" && date +%s > "$B/tasks/{k}/failed/{self.machine}"')
        self.refresh(force=True)

    def mark_done(self, task: str, info: dict) -> None:
        k = key(task)
        self.sh(f'cat > "$B/tasks/{k}/done.json.tmp" && mv "$B/tasks/{k}/done.json.tmp" "$B/tasks/{k}/done.json"',
                input=json.dumps(info))
        self.refresh(force=True)

    def release(self, task: str) -> bool:
        """Remove the task's claim and failure marks, so any machine may take it again."""
        self.refresh(force=True)
        if self.is_done(task):
            raise SystemExit(f"{task} is done; nothing to release")
        had = self.owner(task) is not None or bool(self.failed_on(task))
        self.sh(f'rm -rf "$B/tasks/{key(task)}/claim" "$B/tasks/{key(task)}/failed"')
        self.refresh(force=True)
        return had

    def heartbeat(self, info: dict) -> None:
        body = json.dumps(dict(machine=self.machine, host=socket.gethostname(), work_dir=str(Paths.of(self.cfg).root),
                               time=time.time(), **info))
        self.sh(f'mkdir -p "$B/machines" && cat > "$B/machines/{self.machine}.json.tmp" && '
                f'mv "$B/machines/{self.machine}.json.tmp" "$B/machines/{self.machine}.json"', input=body)


_BOARDS: dict[str, Board] = {}


def get(cfg: dict) -> Board:
    """The process's board for this experiment (refreshed by whoever uses it)."""
    root = str(Paths.of(cfg).root)
    if root not in _BOARDS:
        _BOARDS[root] = Board(cfg)
    return _BOARDS[root]


def _published_mark(cfg: dict, task: str) -> Path:
    return Paths.of(cfg).root / "tmp" / "published" / key(task)


def task_done(cfg: dict, task: str) -> bool:
    """For ``publish/<task>`` jobs: the task's outputs are on the board (this machine published them, or the board,
    read at most every sharing.poll_s seconds, says so)."""
    if _published_mark(cfg, task).exists():
        return True
    board = get(cfg)
    board.refresh()
    return board.is_done(task)


# ---------------------------------------------------------------- the publish / fetch steps


def publish(cfg: dict, task: str) -> None:
    """Copy the task's outputs to the board (checkpoints weights-only), then mark it done."""
    import shutil

    from vig_dagger.hub import export_checkpoint

    board, root = get(cfg), Paths.of(cfg).root
    board.refresh(force=True)
    if board.is_done(task):
        print(f"[publish] {task} is already done")
    else:
        if board.owner(task) != board.machine:
            raise SystemExit(f"[publish] {task} is claimed by {board.owner(task)}, not by this machine ({board.machine})")
        if not exports_present(cfg, task):
            raise SystemExit(f"[publish] {task}: its outputs are not complete in {root}")
        exports, archive = outputs(cfg, task)
        files = _local_files(root, exports + archive)
        staging = root / "tmp" / "publish" / key(task)
        shutil.rmtree(staging, ignore_errors=True)
        plain = [f for f in files if f.suffix != ".pt"]
        weights = []
        for path in files:
            if path.suffix == ".pt":  # weights only: enough for every later task, a third of the size
                export_checkpoint(path, staging / path.relative_to(root))
                weights.append(staging / path.relative_to(root))
        size = sum(p.stat().st_size for p in plain + weights) / 2**20
        print(f"[publish] {task}: {len(files)} files, {size:.0f} MB -> {board.spec}", flush=True)
        for attempt in range(5):
            try:
                board.push(root, plain)
                board.push(staging, weights)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"[publish] copy failed ({exc}); retrying in a minute", flush=True)
                time.sleep(60)
        shutil.rmtree(staging, ignore_errors=True)
        board.mark_done(task, dict(task=task, machine=board.machine, time=time.time(), files=len(files)))
        print(f"[publish] {task} done", flush=True)
    mark = _published_mark(cfg, task)
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(str(time.time()))


def fetch(cfg: dict, task: str) -> None:
    """Copy the task's exports from the board into this work directory."""
    root = Paths.of(cfg).root
    if exports_present(cfg, task):
        return
    get(cfg).pull(outputs(cfg, task)[0], root)
    if not exports_present(cfg, task):
        raise SystemExit(f"[fetch] {task}: its outputs are not on the board {cfg['sharing']['board']} (or not complete)")
    print(f"[fetch] {task} -> {root}", flush=True)


def pull_results(cfg: dict) -> None:
    """Everything the report reads, from every machine: the collections' metrics, the MoEs' logs and thresholds."""
    get(cfg).pull(["results/collections/**", "policies/moe/*/metrics.jsonl", "policies/moe/*/run_config.json",
                   "policies/moe/*/train_config.json", "policies/moe/*/thresholds.json"], Paths.of(cfg).root)


def check(cfg: dict) -> tuple[bool, str]:
    """Whether this machine can use the board: run commands there and copy files both ways."""
    board = get(cfg)
    probe = f".probe-{board.machine}"
    try:
        board.sh(f'mkdir -p "$B" && echo ok > "$B/{probe}" && cat "$B/{probe}"')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / probe).write_text("ok")
            board._rsync([f"{tmp}/{probe}", board.location()])
            board._rsync([board.location() + probe, f"{tmp}/back"])
        board.sh(f'rm -f "$B/{probe}"')
        return True, board.spec + (f" (over ssh to {board.host})" if board.host else " (a local or shared directory)")
    except Exception as exc:
        return False, f"{board.spec}: {exc}"


# ---------------------------------------------------------------- the queue's side


class Scheduler:
    """What the local queue may run (the jobs of this machine's tasks and the preparation they need), and claiming
    new tasks when the machine has room. Used by ``jobs.Queue`` with a board."""

    def __init__(self, cfg: dict, jobs: dict) -> None:
        self.cfg, self.jobs = cfg, jobs
        self.board = get(cfg)
        self.tasks = sorted({j.task for j in jobs.values() if j.task is not None})
        self.members = {t: [n for n, j in jobs.items() if j.task == t] for t in self.tasks}
        self.priority = {t: min(jobs[n].priority for n in self.members[t]) for t in self.tasks}
        self.needs = {t: self._needs(t) for t in self.tasks}
        self.reported_failed: set[str] = set()
        self.beat_at = 0.0
        self.low_disk_reported = False

    def _closure(self, roots: list[str], mine: set[str]) -> set[str]:
        """Jobs reachable from roots without entering another machine's task (its publish job is included)."""
        seen, stack = set(), list(roots)
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            job = self.jobs[name]
            if job.task is not None and job.task not in mine:
                continue  # another task: only its publish job is reached, through a fetch
            stack += job.deps
        return seen

    def _needs(self, task: str) -> set[str]:
        """The tasks whose outputs this task waits for."""
        closure = self._closure(self.members[task], {task})
        return {self.jobs[n].task for n in closure if self.jobs[n].task not in (None, task)}

    def mine(self) -> set[str]:
        return {t for t in self.board.mine() if t in self.members}

    def eligible(self) -> tuple[set[str], set[str]]:
        """(jobs this machine may start, jobs whose completion this machine waits for)."""
        mine = self.mine()
        needed = self._closure([n for t in mine for n in self.members[t]], mine)
        runnable = {n for n in needed if self.jobs[n].task is None or self.jobs[n].task in mine}
        return runnable, needed

    def claimable(self) -> list[str]:
        """Tasks nobody runs whose inputs are all on the board, most urgent first."""
        b = self.board
        free = [t for t in self.tasks if not b.is_done(t) and b.owner(t) is None and all(b.is_done(n) for n in self.needs[t])]
        return sorted(free, key=lambda t: (self.priority[t], t))

    def all_done(self) -> bool:
        return all(self.board.is_done(t) for t in self.tasks)

    def tick(self, queue, waiting_for_room: bool) -> None:
        """Refresh the board; report failed tasks; claim one task if the machine has room; send a heartbeat. A board
        error (network) is logged and the queue goes on; the next tick tries again."""
        try:
            self._tick(queue, waiting_for_room)
        except Exception as exc:
            print(f"[board] {time.strftime('%H:%M:%S')} {type(exc).__name__}: {exc}; trying again later", flush=True)

    def _tick(self, queue, waiting_for_room: bool) -> None:
        b = self.board
        fresh = time.time() - b.read_at >= b.poll
        b.refresh()
        for name in queue.failed:
            task = self.jobs[name].task
            if task in self.mine() and task not in self.reported_failed and not b.is_done(task):
                b.mark_failed(task)
                self.reported_failed.add(task)
                print(f"[board] {task} failed here; marked on the board", flush=True)
        open_tasks = [t for t in self.mine() if not b.is_done(t)]
        unstarted = [t for t in open_tasks if not any(n in queue.running or n in queue.done for n in self.members[t])]
        if fresh and not waiting_for_room and not unstarted:  # one new task at a time, once the last one got going
            import shutil

            free_gb = shutil.disk_usage(Paths.of(self.cfg).root).free / 2**30
            for task in self.claimable():
                if task.startswith("dagger/") and free_gb < MIN_FREE_GB:  # a collection needs ~15 GB
                    if not self.low_disk_reported:
                        print(f"[board] only {free_gb:.0f} GB free in the work directory: no more collections here", flush=True)
                        self.low_disk_reported = True
                    continue
                probe = next((self.jobs[n] for n in self.members[task] if self.jobs[n].kind == "train"), None) or \
                    next((self.jobs[n] for n in self.members[task] if self.jobs[n].kind == "sim"), None) or \
                    self.jobs[self.members[task][0]]
                if queue.place(probe) is False:  # no room for it here now
                    continue
                if b.claim(task):
                    print(f"[board] {time.strftime('%H:%M:%S')} claimed {task}", flush=True)
                break
        if time.time() - self.beat_at >= float(self.cfg["sharing"]["heartbeat_s"]):
            from vig_dagger.ops import _progress

            self.beat_at = time.time()
            b.heartbeat(dict(running=sorted(queue.running), tasks=sorted(t for t in self.mine() if not b.is_done(t)),
                             progress={n: _progress(self.cfg, n) for n in sorted(queue.running)}))
