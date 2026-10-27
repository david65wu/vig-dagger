"""Several machines through one public Hugging Face repo (``sharing.mode: hub``).

Every machine runs the same config and the same job graph (``plan.py``). The graph's work comes in **tasks**:

  train/<run>                  one training (the MoE, a res<k> driver, a retrained policy)
  thresholds/<round>           a round's demo-frame scores and thresholds
  dagger/<round>/<scorer>      a DAgger collection, its training set (datasets/dagger_<round>_<scorer>) and its metrics
  eval/<policy>/level<f>       one policy's evaluation at one level (its 25-episode chunks)
  summary                      the evaluation summary and REPORT.md, once every evaluation and collection is in

The rest (downloading and converting the demos, combining a DAgger set with the base demos) is local preparation
that any machine does when one of its tasks needs it. A machine with room claims the most urgent task whose inputs
are all in the repo, downloads them (``fetch/<task>`` jobs), runs the task's jobs in its own queue, and uploads the
outputs (``publish/<task>``). A task nobody can start yet waits for the uploads it needs, so the steps that cannot run
in parallel (a round's thresholds, a DAgger collection) run on one machine while the others work on something else.

Repo layout (``sharing.repo``, a public dataset repo; the outputs sit at their work_dir paths):

    experiment.json                         the experiment's settings: a machine with other settings refuses to join
    tasks/<key>/claimed_by/<machine>        who runs the task. Claimed with a commit that fails if anything was
                                            committed since the machine last looked, so two machines never run it both
    tasks/<key>/done.json                   committed together with the task's outputs
    tasks/<key>/failed/<machine>            the task failed there (``vig-dagger release`` frees it for another machine)
    machines/<machine>.json                 heartbeat: what the machine runs (every sharing.heartbeat_s)
    policies/<run>/latest.pt, ...           outputs (checkpoints weights-only)

``<key>`` is the task name with ``/`` replaced by ``__``; ``<machine>`` is the host name plus a short hash of the work
directory (``machine_id``), so two work directories on one host are two machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

from vig_dagger.config import Paths

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # job logs, not terminals

EXPERIMENT_KEYS = ("demos", "training", "rounds", "drivers", "scorers", "thresholds", "dagger", "retrain", "evaluation")
REPO_TYPE = "dataset"
# Hugging Face allows 128 commits per hour per repository. The run makes about 16 an hour on average (a claim and an
# upload per task, two heartbeats per machine), so the limit only matters in bursts: a machine that hits it pauses its
# commits (claims, heartbeats) for RATE_LIMIT_PAUSE_S, and an upload waits and tries again.
RATE_LIMIT_PAUSE_S = 600


def rate_limited(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 429


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
    """The settings that define the experiment (DAgger worker counts don't)."""
    out = {k: cfg.get(k) for k in EXPERIMENT_KEYS}
    out["dagger"] = {k: v for k, v in (out["dagger"] or {}).items() if k != "workers_per_scorer"}
    return json.loads(json.dumps(out))


# ---------------------------------------------------------------- what a task uploads and what others download


def outputs(cfg: dict, task: str) -> tuple[list[str], list[str]]:
    """(exports, archive): glob patterns relative to the work directory. Exports are what other tasks need and
    download; the archive is uploaded for the record only (DAgger attempts and trajectories)."""
    from vig_dagger.train import resolve

    paths = Paths.of(cfg)
    kind, _, rest = task.partition("/")
    if kind == "train":
        run = resolve(cfg, rest)[1].relative_to(paths.root)
        return [f"{run}/latest.pt", f"{run}/run_config.json", f"{run}/metrics.jsonl"], []
    if kind == "thresholds":
        driver = paths.driver_dir(rest).relative_to(paths.root)
        return [f"{driver}/thresholds.json", f"{driver}/demo_scores.npz"], []
    if kind == "dagger":
        r, s = rest.split("/")
        out = paths.dagger(r, s).relative_to(paths.root)
        exports = [f"data/datasets/dagger_{r}_{s}/**", f"results/dagger_{r}_{s}.json", f"results/dagger_{r}_{s}_*",
                   f"results/figures/dagger_{r}_{s}_*", f"{out}/settings.json", f"{out}/episodes/*"]
        archive = [f"{out}/attempts/*"] + ([f"{out}/trajectories/*"] if cfg["sharing"]["upload_trajectories"] else [])
        return exports, archive
    if kind == "eval":
        return [f"eval/{rest}/*.jsonl"], []
    if task == "summary":  # with config.json, a download of the repo is a work directory `vig-dagger merge` accepts
        return ["config.json", "results/*", "results/figures/*"], []
    raise ValueError(f"unknown task {task!r}")


def exports_present(cfg: dict, task: str) -> bool:
    """The task's outputs that other tasks need are in this work directory."""
    from vig_dagger import dataset
    from vig_dagger.train import resolve

    paths = Paths.of(cfg)
    kind, _, rest = task.partition("/")
    if kind == "train":
        return (resolve(cfg, rest)[1] / "latest.pt").is_file()
    if kind == "thresholds":
        return paths.thresholds(rest).is_file()
    if kind == "dagger":
        r, s = rest.split("/")
        return dataset.ready(paths.dataset(f"dagger_{r}_{s}"))
    if kind == "eval":
        n, chunk = int(cfg["evaluation"]["num_episodes"]), int(cfg["evaluation"]["chunk"])
        files = list((paths.root / "eval" / rest).glob("seeds_*.jsonl"))
        lines = sum(len(f.read_text().splitlines()) for f in files)
        return len(files) >= -(-n // chunk) and lines >= n
    if task == "summary":
        return (paths.results() / "eval_summary.json").is_file()
    raise ValueError(f"unknown task {task!r}")


def _local_files(root: Path, patterns: list[str]) -> list[Path]:
    """Files matching the patterns; ``dir/**`` is everything below dir (as in the repo's patterns)."""
    files = set()
    for pattern in patterns:
        found = (root / pattern[:-3]).rglob("*") if pattern.endswith("/**") else root.glob(pattern)
        files.update(p for p in found if p.is_file())
    return sorted(files)


# ---------------------------------------------------------------- the board


class Board:
    """The shared state of the tasks, read from the repo at most every ``sharing.poll_s`` seconds."""

    def __init__(self, cfg: dict) -> None:
        from huggingface_hub import HfApi

        self.cfg, self.repo = cfg, cfg["sharing"]["repo"]
        self.api = HfApi()
        self.machine = machine_id(cfg)
        self.poll = float(cfg["sharing"]["poll_s"])
        self.head: str | None = None
        self.files: set[str] = set()
        self.read_at = 0.0

    # ---- reading
    def refresh(self, force: bool = False) -> None:
        """Read the board (at most every sharing.poll_s seconds unless forced). An unforced read that fails keeps the
        last state; a forced one (before a claim or an upload) raises."""
        if not force and time.time() - self.read_at < self.poll:
            return
        try:
            self._read()
        except Exception as exc:
            if force or not self.read_at:
                raise
            print(f"[board] could not read {self.repo} ({type(exc).__name__}); using the state of "
                  f"{(time.time() - self.read_at) / 60:.0f} min ago", flush=True)
            self.read_at = time.time()  # try again after the next poll interval

    def _read(self) -> None:
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            head = self.api.repo_info(self.repo, repo_type=REPO_TYPE).sha
        except RepositoryNotFoundError:  # not created yet (the first machine's `run` creates it): nothing claimed or done
            self.head, self.files, self.read_at = None, set(), time.time()
            return
        try:  # only tasks/: the outputs are thousands of files
            entries = self.api.list_repo_tree(self.repo, path_in_repo="tasks", recursive=True, repo_type=REPO_TYPE,
                                              revision=head)
            files = {e.path for e in entries if not hasattr(e, "tree_id")}  # RepoFile has no tree_id
        except EntryNotFoundError:
            files = set()
        self.head, self.files, self.read_at = head, files, time.time()

    def owner(self, task: str) -> str | None:
        prefix = f"tasks/{key(task)}/claimed_by/"
        owners = [f[len(prefix):] for f in self.files if f.startswith(prefix)]
        return owners[0] if owners else None

    def is_done(self, task: str) -> bool:
        return f"tasks/{key(task)}/done.json" in self.files

    def failed_on(self, task: str) -> list[str]:
        prefix = f"tasks/{key(task)}/failed/"
        return sorted(f[len(prefix):] for f in self.files if f.startswith(prefix))

    def mine(self) -> set[str]:
        """Tasks this machine has claimed (done or not), as task names."""
        out = set()
        for f in self.files:
            parts = f.split("/")
            if len(parts) == 4 and parts[2] == "claimed_by" and parts[3] == self.machine:
                out.add(parts[1].replace("__", "/"))
        return out

    # ---- writing
    def _commit(self, operations: list, message: str, parent: str | None = None):
        return self.api.create_commit(self.repo, repo_type=REPO_TYPE, operations=operations, commit_message=message,
                                      parent_commit=parent)

    def ensure_repo(self) -> None:
        """Create the public repo if needed and check this machine runs the same experiment as the others."""
        from huggingface_hub import CommitOperationAdd, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

        public = bool(self.cfg["sharing"]["public"])
        self.api.create_repo(self.repo, repo_type=REPO_TYPE, private=not public, exist_ok=True)
        if public and self.api.repo_info(self.repo, repo_type=REPO_TYPE).private:
            self.api.update_repo_settings(self.repo, repo_type=REPO_TYPE, private=False)
        mine = experiment(self.cfg)
        for _ in range(5):
            try:
                theirs = json.loads(Path(hf_hub_download(self.repo, "experiment.json", repo_type=REPO_TYPE,
                                                         force_download=True)).read_text())
            except EntryNotFoundError:
                head = self.api.repo_info(self.repo, repo_type=REPO_TYPE).sha
                try:
                    self._commit([CommitOperationAdd("experiment.json", json.dumps(mine, indent=1).encode())],
                                 "experiment settings", parent=head)
                    return
                except HfHubHTTPError as exc:  # another machine wrote it first: compare with that one
                    if exc.response is None or exc.response.status_code not in (409, 412):
                        raise
                    continue
            differ = sorted(k for k in EXPERIMENT_KEYS if theirs.get(k) != mine[k])
            if differ:
                raise SystemExit(f"{self.repo} holds an experiment with other settings ({differ}); use the same config "
                                 "as the other machines, or another sharing.repo")
            return
        raise SystemExit(f"could not read or write experiment.json in {self.repo}")

    def claim(self, task: str) -> bool:
        """Claim the task for this machine; False if another machine has it (or it is done)."""
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.utils import HfHubHTTPError

        for _ in range(6):
            self.refresh(force=True)
            owner = self.owner(task)
            if owner is not None or self.is_done(task):
                return owner == self.machine
            body = json.dumps(dict(machine=self.machine, host=socket.gethostname(), time=time.time())).encode()
            try:
                self._commit([CommitOperationAdd(f"tasks/{key(task)}/claimed_by/{self.machine}", body)],
                             f"{self.machine} claims {task}", parent=self.head)
                self.refresh(force=True)
                return self.owner(task) == self.machine
            except HfHubHTTPError as exc:  # someone committed since we looked: look again
                if exc.response is None or exc.response.status_code not in (409, 412):
                    raise
                time.sleep(1 + 2 * (hash(self.machine) % 5))
        return False

    def mark_failed(self, task: str) -> None:
        from huggingface_hub import CommitOperationAdd

        self._commit([CommitOperationAdd(f"tasks/{key(task)}/failed/{self.machine}", str(time.time()).encode())],
                     f"{task} failed on {self.machine}")
        self.refresh(force=True)

    def release(self, task: str) -> list[str]:
        """Remove the task's claim and failure marks, so any machine may take it again. Returns what was removed."""
        from huggingface_hub import CommitOperationDelete

        self.refresh(force=True)
        if self.is_done(task):
            raise SystemExit(f"{task} is done; nothing to release")
        gone = [f for f in self.files if f.startswith(f"tasks/{key(task)}/claimed_by/") or
                f.startswith(f"tasks/{key(task)}/failed/")]
        if gone:
            self._commit([CommitOperationDelete(f) for f in gone], f"release {task}")
            self.refresh(force=True)
        return gone

    def heartbeat(self, info: dict) -> None:
        from huggingface_hub import CommitOperationAdd

        body = json.dumps(dict(machine=self.machine, host=socket.gethostname(), work_dir=str(Paths.of(self.cfg).root),
                               time=time.time(), **info), indent=1).encode()
        self._commit([CommitOperationAdd(f"machines/{self.machine}.json", body)], f"heartbeat {self.machine}")

    def machines(self) -> list[dict]:
        """Every machine's last heartbeat."""
        from huggingface_hub import hf_hub_download

        from huggingface_hub.utils import EntryNotFoundError

        out = []
        try:
            entries = list(self.api.list_repo_tree(self.repo, path_in_repo="machines", repo_type=REPO_TYPE))
        except EntryNotFoundError:
            return out
        for entry in entries:
            if entry.path.endswith(".json"):
                path = hf_hub_download(self.repo, entry.path, repo_type=REPO_TYPE, force_download=True)
                out.append(json.loads(Path(path).read_text()))
        return out


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
    """For ``publish/<task>`` jobs: the task's outputs are in the repo (this machine published them, or the board,
    read at most every sharing.poll_s seconds, says so)."""
    if _published_mark(cfg, task).exists():
        return True
    board = get(cfg)
    board.refresh()
    return board.is_done(task)


def _mark_published(cfg: dict, task: str) -> None:
    mark = _published_mark(cfg, task)
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(str(time.time()))


# ---------------------------------------------------------------- the publish / fetch steps


def publish(cfg: dict, task: str) -> None:
    """Upload the task's outputs and mark it done, in one commit (checkpoints weights-only)."""
    from huggingface_hub import CommitOperationAdd

    from vig_dagger.hub import export_checkpoint

    board, root = get(cfg), Paths.of(cfg).root
    board.refresh(force=True)
    if board.is_done(task):
        print(f"[publish] {task} is already done")
        _mark_published(cfg, task)
        return
    if board.owner(task) != board.machine:
        raise SystemExit(f"[publish] {task} is claimed by {board.owner(task)}, not by this machine ({board.machine})")
    exports, archive = outputs(cfg, task)
    files = _local_files(root, exports + archive)
    if not exports_present(cfg, task):
        raise SystemExit(f"[publish] {task}: its outputs are not complete in {root}")
    staging = root / "tmp" / "publish" / key(task)
    operations = []
    for path in files:
        rel = str(path.relative_to(root))
        source = path
        if path.suffix == ".pt":  # weights only: loads for inference and evaluation, a third of the size
            source = staging / rel
            export_checkpoint(path, source)
        operations.append(CommitOperationAdd(rel, str(source)))
    done = dict(task=task, machine=board.machine, time=time.time(), files=len(files))
    operations.append(CommitOperationAdd(f"tasks/{key(task)}/done.json", json.dumps(done).encode()))
    size = sum(p.stat().st_size for p in files) / 2**20
    print(f"[publish] {task}: {len(files)} files, {size:.0f} MB -> {board.repo}", flush=True)
    failures = 0
    while True:
        try:
            board._commit(operations, f"{board.machine}: {task} done")
            break
        except Exception as exc:  # try again with fresh operation objects
            if rate_limited(exc):  # the repo's hourly commit limit: wait, up to about an hour and a half
                failures += 1
                if failures > 9:
                    raise
                print(f"[publish] Hugging Face's commit limit is reached; waiting {RATE_LIMIT_PAUSE_S // 60} min", flush=True)
                time.sleep(RATE_LIMIT_PAUSE_S)
            else:  # network: three tries
                failures += 3
                if failures > 9:
                    raise
                print(f"[publish] upload failed ({exc}); retrying", flush=True)
                time.sleep(30)
            operations = [CommitOperationAdd(op.path_in_repo, op.path_or_fileobj) for op in operations]
    import shutil

    shutil.rmtree(staging, ignore_errors=True)
    _mark_published(cfg, task)
    print(f"[publish] {task} done", flush=True)


def fetch(cfg: dict, task: str) -> None:
    """Download the task's exports into this work directory."""
    from huggingface_hub import snapshot_download

    root = Paths.of(cfg).root
    if exports_present(cfg, task):
        return
    exports, _ = outputs(cfg, task)
    snapshot_download(cfg["sharing"]["repo"], repo_type=REPO_TYPE, local_dir=root, allow_patterns=exports)
    if not exports_present(cfg, task):
        raise SystemExit(f"[fetch] {task}: its outputs are not in {cfg['sharing']['repo']} (or not complete)")
    print(f"[fetch] {task} -> {root}", flush=True)


def pull_results(cfg: dict) -> None:
    """Everything the summary and the report read, from every machine: evaluations, DAgger metrics, training logs,
    and the demo-frame scores the stage figures compare with."""
    from huggingface_hub import snapshot_download

    snapshot_download(cfg["sharing"]["repo"], repo_type=REPO_TYPE, local_dir=Paths.of(cfg).root,
                      allow_patterns=["eval/**", "results/dagger_*", "results/figures/dagger_*_timelines.png",
                                      "policies/**/metrics.jsonl", "policies/**/run_config.json",
                                      "policies/drivers/*/thresholds.json", "policies/drivers/*/demo_scores.npz"])


# ---------------------------------------------------------------- the queue's side


class HubScheduler:
    """What the local queue may run (the jobs of this machine's tasks and the preparation they need), and claiming
    new tasks when the machine has room. Used by ``jobs.Queue`` in hub mode."""

    def __init__(self, cfg: dict, jobs: dict) -> None:
        self.cfg, self.jobs = cfg, jobs
        self.board = get(cfg)
        self.tasks = sorted({j.task for j in jobs.values() if j.task is not None})
        self.members = {t: [n for n, j in jobs.items() if j.task == t] for t in self.tasks}
        self.priority = {t: min(jobs[n].priority for n in self.members[t]) for t in self.tasks}
        self.needs = {t: self._needs(t) for t in self.tasks}
        self.reported_failed: set[str] = set()
        self.beat_at = 0.0
        self.paused_until = 0.0  # no commits from this machine before then (Hugging Face's rate limit)

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
        roots = [n for t in mine for n in self.members[t]]
        needed = self._closure(roots, mine)
        runnable = {n for n in needed if self.jobs[n].task is None or self.jobs[n].task in mine}
        return runnable, needed

    def claimable(self) -> list[str]:
        """Tasks nobody runs whose inputs are all in the repo, most urgent first."""
        b = self.board
        free = [t for t in self.tasks if not b.is_done(t) and b.owner(t) is None
                and all(b.is_done(n) for n in self.needs[t])]
        return sorted(free, key=lambda t: (self.priority[t], t))

    def all_done(self) -> bool:
        return all(self.board.is_done(t) for t in self.tasks)

    def tick(self, queue, waiting_for_room: bool) -> None:
        """Refresh the board; report failed tasks; claim one task if the machine has room; send a heartbeat. A Hugging
        Face error (network, rate limit) is logged and the queue goes on; the next tick tries again."""
        try:
            self._tick(queue, waiting_for_room)
        except Exception as exc:
            if rate_limited(exc):
                self.paused_until = time.time() + RATE_LIMIT_PAUSE_S
                print(f"[board] Hugging Face's commit limit is reached; no claims or heartbeats from this machine for "
                      f"{RATE_LIMIT_PAUSE_S // 60} min", flush=True)
            else:
                print(f"[board] {time.strftime('%H:%M:%S')} {type(exc).__name__}: {exc}; trying again later", flush=True)

    def _tick(self, queue, waiting_for_room: bool) -> None:
        b = self.board
        commits_ok = time.time() >= self.paused_until
        fresh = time.time() - b.read_at >= b.poll
        b.refresh()
        for name in queue.failed:
            task = self.jobs[name].task
            if commits_ok and task in self.mine() and task not in self.reported_failed and not b.is_done(task):
                b.mark_failed(task)
                self.reported_failed.add(task)
                print(f"[board] {task} failed here; marked in {b.repo}", flush=True)
        open_tasks = [t for t in self.mine() if not b.is_done(t)]
        unstarted = [t for t in open_tasks if not any(n in queue.running or n in queue.done for n in self.members[t])]
        if commits_ok and fresh and not waiting_for_room and not unstarted:  # one new task at a time, once the last one got going
            for task in self.claimable():
                probe = next((self.jobs[n] for n in self.members[task] if self.jobs[n].kind == "train"), None) or \
                    next((self.jobs[n] for n in self.members[task] if self.jobs[n].kind == "sim"), None) or \
                    self.jobs[self.members[task][0]]
                if queue.place(probe) is False:  # no room for it here now
                    continue
                if b.claim(task):
                    print(f"[board] {time.strftime('%H:%M:%S')} claimed {task}", flush=True)
                break
        if commits_ok and time.time() - self.beat_at >= float(self.cfg["sharing"]["heartbeat_s"]):
            from vig_dagger.ops import _progress

            self.beat_at = time.time()
            b.heartbeat(dict(running=sorted(queue.running), tasks=sorted(t for t in self.mine() if not b.is_done(t)),
                             progress={n: _progress(self.cfg, n) for n in sorted(queue.running)}))


def token_ok() -> tuple[bool, str]:
    """Whether this machine is logged in to Hugging Face with write access."""
    from huggingface_hub import HfApi

    try:
        who = HfApi().whoami()
    except Exception as exc:
        return False, f"not logged in ({exc}); run `hf auth login` with the token the humans give you"
    role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role")
    return role in ("write", "fineGrained"), f"{who.get('name')} (token: {role})"

