# Running the experiment: instructions for an agent

These instructions are for an agent (or a person) who runs `configs/plugcharger.yaml` (branch
`plugcharger-moe-seeds`) from start to finish without supervision, on one machine or several. The README explains
what the pipeline does; this file says how to operate it: what to run, what a healthy run looks like, when to stop
and ask, and what never to change.

## Short commands

The humans start and check the run with short messages. The wording varies ("start the experiment", "how is it
going"); go by the intent, and ask only if you can't tell which of these is meant:

| the human says | do |
|---|---|
| "run plugcharger" | the whole experiment on this one machine: set up, check and launch (steps 0–4), then report |
| "run plugcharger with the task board at *HOST:PATH*" (or "… board *PATH*") | several machines (every machine gets the same message): set `sharing.board` to that value (step 2), set up, check and launch (steps 0–4), then report. This machine then claims tasks on the board by itself until the experiment is done; see [Several machines](#several-machines) |
| "release task 17" / "release dagger/sd1_vig_p95_plain" | `vig-dagger release <task name>` (`vig-dagger tasks` maps numbers to names), only when the humans say so: it lets another machine take a task whose machine is gone |
| "run task 2", "run tasks 1, 4–15" | machines without a board: only these tasks of the numbered list ([TASKS.md](TASKS.md)): set the machine up if it isn't yet (steps 0–3), then launch with `--tasks` (step 4) and report. If `run` refuses because inputs are missing, report exactly what it lists |
| "pack task 2 for task 16" | `vig-dagger pack -c configs/run.yaml --tasks 2 --for 16`; report the archive's path and size (the humans move it) |
| "unpack *FILE*" | `vig-dagger unpack -c configs/run.yaml FILE`, then report what it put in place |
| "tasks" | `vig-dagger tasks -c configs/run.yaml` (the numbered list, as in TASKS.md) |
| "status" | `vig-dagger status`, `tail run.log`, `df -h <work_dir>`; summarize: queue alive, failed jobs, progress per collection; with a board also every task's state and every machine's heartbeat |
| "report" | `vig-dagger report`, then send `results/REPORT.md` and `results/figures/collections.png` and summarize them (step 7) |
| "stop" | `vig-dagger stop` (every job pauses; the same "run" again resumes) |
| "update" (or "git pull and read UPDATES.md") | `git pull`, then read [UPDATES.md](UPDATES.md) and do what its entries newer than this machine's previous commit say for this machine (the humans approved those steps); then start the queue again (step 4) and report |

If an earlier session already set up and launched this machine (there is a `configs/run.yaml` and a work directory),
"run plugcharger" means: start the queue again if it is not running (step 4), then report the status. Never
reinstall or reconfigure a machine that has started.

## The goal and what "done" means

- **The run:** PlugCharger in ManiSkill. Three 7-level MoE policies that differ only in their seed: seed 0 is
  downloaded from Hugging Face, seeds 1 and 2 are trained here (60 epochs on the 1000 planner demos, 188,580 updates
  each). Each MoE gets VIG and flow-loss thresholds (p90, p95, p99 of its score on demo frames). Then 36 DAgger
  collections of 500 successful episodes each: 3 MoE seeds × {VIG, flow loss} × {p90, p95, p99} × {plain, random
  move}. One threshold decides both directions (take over after 2 scores above, hand back after 16 below). Then, per
  collection, a dense full-resolution policy retrained on the 1000 base demos + its 500 DAgger episodes (60 epochs,
  training seed 0) and evaluated on 200 scenes, and the three MoEs evaluated on the same scenes.
- **The results:** per collection, the collection success rate, the expert share of the DAgger data and the expert
  share of the failed attempts (with the random move's share separate), and per setting their mean ± std over the
  three seeds; per retrained policy its success rate next to its driving MoE's.
- **Done when** (on one machine; with a task board, when `vig-dagger status` shows all 80 tasks done and every
  machine's queue ended; without a board, each machine is done when its assigned tasks are, and the experiment when
  the final report, task 80, has run):
  - `vig-dagger status` shows every job `done`;
  - `<work_dir>/results/summary.json` says 36 collections finished;
  - `<work_dir>/results/REPORT.md` has the tables filled in (also "Retrained policies"), and
    `results/figures/collections.png` and `eval_success.png` exist.

Every result is written to files in `<work_dir>/results`; W&B is not used (`wandb.enabled: false`).
- **Then report** the numbers listed in [When it finishes](#7-when-it-finishes).

Expect about 3 days for the collections on one 64-core machine with 2 GPUs (~20–60 worker-hours each), then the 36
retrainings (6–20 h each on one GPU, one per GPU at a time; each starts as soon as its collection is finished); less
on several machines.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `moe`,
  `scorers`, `thresholds`, `dagger`. These define the experiment. A changed value either silently mixes two
  experiments, or the pipeline refuses to continue.
- **Delete or move anything in `work_dir`**, and above all `dagger/`, `policies/`, `data/`. Collections take a day
  or more; partial outputs are what lets a run resume.
- **Upload** anything to Hugging Face, **push** to git, or turn on W&B.
- **Copy files between machines yourself.** With a task board the pipeline moves what other machines need;
  without one, the humans move the archives. Never run tasks the humans did not assign to a machine without a board.
- **Release a task** (`vig-dagger release`) unless the humans tell you to.
- **Run two queues on the same `work_dir`**, or start workers by hand outside the queue.
- **Edit the code**, including the planner, the random move and the environment.

**You may:**

- set `work_dir` before the first start;
- change `resources.*` settings, e.g. fewer `sim_slots` when the machine is shared. To apply them without
  interrupting jobs: `vig-dagger stop --queue-only`, edit the config, then `vig-dagger run` again;
- stop and restart the queue (`vig-dagger stop`, `vig-dagger run`);
- rerun `vig-dagger analyze` and `vig-dagger report` at any time;
- read any file or log.

**Stop and ask the humans when:**

- `vig-dagger check` reports a failure you cannot fix by installing something: no Vulkan, not enough disk, no access
  to Hugging Face;
- a job is `FAILED` again after you fixed what its log showed and restarted once, or you can't tell why it failed;
- `vig-dagger run --tasks` refuses because inputs from other tasks are missing: tell the humans what it lists;
- a collection is heading for its attempt limit (10,000 attempts). Concretely: after 300 attempts it has fewer than
  15 successes (a success rate under 5 %);
- free disk in `work_dir` drops below 30 GB (each collection needs about 15 GB);
- a training's loss is NaN, or its validation loss rises steadily over the second half of training;
- a milestone value is far outside the expected range below.

## 0. Before you start

**With a task board** (the message names one), check first that this machine can use it, unless the board is a
local or shared path: `ssh -o BatchMode=yes <host> true` must succeed without asking for a password, and `rsync` must be
installed here and on the host. If either fails, ask the humans (they set up the SSH key); don't create keys or edit
SSH settings yourself.

**Everything else has a default.** Use it unless the human's message says otherwise, and say in your first report
which values you used:

| setting | default |
|---|---|
| work directory | `~/vig-dagger-runs/plugcharger_moe_seeds` if that disk has room (about 15 GB per collection this machine runs, plus 35 GB; ~580 GB for all of them); otherwise a directory you can write to on a disk that has; if there is none, ask |
| GPUs | every GPU (`resources.gpus: auto`) |
| CPU budget | the default (`resources.sim_slots: auto`, half the cores' worth of simulator jobs). If other users' jobs already load the machine (`uptime` load average above half of `nproc`), halve it and say so |
| reporting | a message after the launch, after each milestone you see, and the final report (step 7) |

Say in your first report that the run takes days and keeps going after your session ends; the humans check on it
with "status".

## 1. Setup

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout plugcharger-moe-seeds
uv sync
source .venv/bin/activate
```

Hugging Face downloads need no login (both repositories are public). `uv` and `ffmpeg` are the only programs to
install. If either is missing and installing it needs administrator rights, ask.

## 2. Configure

Copy `configs/plugcharger.yaml` to `configs/run.yaml`. Set `work_dir` (step 0). With a task board, set
`sharing.board` to exactly what the humans gave (`host:/path`, or a path). If the human named GPUs or a CPU budget, set
`resources.gpus` / `resources.sim_slots`. Leave everything else as it is: with several machines, every machine must use
the same settings (the board refuses a machine with other settings).

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable (a disk WARN means the disk holds only a few collections:
fine for a machine that runs a few tasks, not for one that runs everything). It checks:
- free disk, `ffmpeg`, the GPUs, and that torch sees CUDA;
- rendering (Vulkan);
- that the demos and the seed-0 MoE exist on Hugging Face, and the DINOv2 weights are reachable;
- with a board: that this machine can run commands there and copy files both ways, and that the board holds this
  experiment (the first machine creates it);
- the W&B setting (it should say `W&B disabled`).

It prints the CPU and GPU budget the queue will use and the DAgger workers per collection.

On a new machine, also run the smoke test (15–30 minutes, writes only to the given directory):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

It must end with `[queue] finished: 36 of 36 jobs done`. The smoke MoEs are barely trained, so their episodes are
mostly the expert's; that is expected. Delete `/tmp/vig-dagger-smoke` afterwards.

## 4. Launch

```bash
setsid nohup vig-dagger run -c configs/run.yaml > run.log 2>&1 < /dev/null &
```

With a task board the command is the same on every machine (no `--tasks`): the queue claims a task whenever the
machine has room, fetches its inputs from the board, runs it and publishes its outputs, and keeps going until every
task of the experiment is done; a machine with nothing to claim waits. Its log shows `[board] claimed <task>` lines.

Without a board, for assigned tasks, add `--tasks` (and use a log file of their own, e.g. `run_tasks_2_16-27.log`):
`vig-dagger run -c configs/run.yaml --tasks 2 16-27`. It first checks that the outputs of the other tasks these need
are in this work directory, then runs the tasks and the preparation they need (downloading and converting the
demos), and ends by naming the tasks that use their outputs. A machine can run another `--tasks` command after the
first one has finished (not two at a time: one queue per work directory).

`run.log` is the queue's log: every job it starts, finishes, retries or gives up on. Each job writes its own log
to `<work_dir>/logs/<job name>.log`.

## 5. Monitor

About every hour:

```bash
vig-dagger status -c configs/run.yaml    # queue alive? failed and running jobs, progress, milestones
tail -n 30 run.log
df -h <work_dir>
```

About once a day, or after a milestone, run `vig-dagger report -c configs/run.yaml`. It refreshes
`<work_dir>/results/REPORT.md`, `summary.json` and `figures/collections.png` with everything finished so far.

`status` shows:
- whether the queue is RUNNING (its heartbeat is less than 2 minutes old);
- every FAILED job with its log path;
- every running job with its GPU, run time and progress (training updates, a collection's successes and attempts);
- the milestones: each MoE and its thresholds, each collection's successes and attempts, and which task numbers are
  done in this work directory.

If the queue is NOT RUNNING and jobs remain, start it again with the command from step 4. It takes over jobs that
are still running.

### Milestones and expected values

| milestone | when | what to check | expected |
|---|---|---|---|
| demos downloaded | first ~30 min | `<work_dir>/data/raw/planner/` | 20 `part_*.h5` (24 GB) |
| dataset converted | right after (only where a MoE is trained) | `num_demos` in `data/datasets/base/dataset_manifest.json` | 1000 |
| MoE seed 0 | minutes | `policies/moe/sd0/latest.pt` (a link into `hub/`) | 386 MB |
| MoE seeds 1, 2 trainings | 4–12 h each, in parallel on two GPUs | `policies/moe/sd{1,2}/metrics.jsonl`, `run_config.json` → `training.max_updates` and `_info` | exactly 188,580 updates, 975 training demos, 201,099 training samples (seed 0's); validation loss falling to about seed 0's (its final value is in the Trainings table of `vig-dagger report`). Anything else: stop and ask |
| thresholds | ~30 min after each MoE | `policies/moe/sd<s>/thresholds.json` | all positive, p90 < p95 < p99. For scale, seed 0 measured earlier: VIG p90 ≈ 0.026, p95 ≈ 0.032, p99 ≈ 0.044; flow p90 ≈ 0.027, p95 ≈ 0.035, p99 ≈ 0.056 (seeds 1 and 2 should be similar) |
| collections | seed 0's from ~1 h on; seeds 1 and 2 after their trainings | `dagger/<collection>/settings.json`; `status` progress | take and return thresholds both equal the collection's percentile; successes rising steadily. For scale, an earlier run of seed 0 with a faster expert: VIG success rate 63–95 % per attempt, flow 35–41 %; lower thresholds give higher expert shares |
| collection metrics | as each collection finishes | `results/collections/<collection>.json`; the tables and `figures/collections.png` in `results/REPORT.md` | the experiment's new numbers |
| summary | after the collections | `results/summary.json` | `collections_finished` 36 |
| retraining datasets | when a collection is finished | `logs/dagger_dataset_<collection>.log` | "dagger_<collection>: 500 episodes", "base_plus_<collection>: 1500 demos" |
| retrainings | right after | `policies/retrain/<collection>/seed0/run_config.json` → `training.max_updates`, `_info`; no `moe` in `policy.head` | about 280–330k updates; `_info.training_demos` 1,455–1,475 (seed 0's 975 base demos + the 500 DAgger episodes less the ~2 % held out for validation: 10 on average, 0–20 by chance, different for every collection); validation loss falling |
| evaluations | after each training, and the MoEs' | `eval/<policy>/level1/*.jsonl` | 200 episodes per policy (8 files of 25); the MoEs at full resolution near 34 % (seed 0 measured earlier) |
| final report | at the end | `results/eval_summary.json`; "Retrained policies" in `results/REPORT.md` | 36 retrained policies and 3 MoEs |

The runtimes are for a shared 2-GPU, 64-core machine.

## 6. When something goes wrong

| symptom | likely cause | what to do |
|---|---|---|
| `status` lists a FAILED job | see its log (path in `status`) | fix the cause if it is in the "you may" list, then `vig-dagger run` again; otherwise ask |
| a job retries again and again | the same error each time | read the log; don't wait for the retries to run out |
| `CUDA out of memory` in a training or worker log | other users filled the GPU | nothing; the retry runs when memory frees up. If it keeps happening, ask whether to raise `resources.train_gpu_memory_gb` |
| W&B error when a training starts ("could not verify", 401/403) | `wandb.enabled` was set to true | set it back to false (W&B is not used); then rerun |
| `vk::`, "no device", rendering errors | Vulkan driver | ask (needs a system fix) |
| `No space left on device` | disk full | stop the queue (`vig-dagger stop`) and ask; never delete experiment outputs to make room |
| queue NOT RUNNING, jobs remain | the queue process died (SSH session, reboot) | start it again (step 4) |
| machine rebooted | | start it again; trainings resume from their last checkpoint (at most 5000 updates lost) |
| `convert_planner` log: "… names its demos like 'part_000__traj_0' …", or a training log: "run_config.json was written for another dataset" | the dataset (or the training) was made by an earlier version of this branch, whose demo names gave other validation demos | ask the humans for approval, then delete exactly the directories the message names (`data/datasets/planner`, `data/datasets/base`, and `policies/moe/sd<s>` of a training started on them) and run again; nothing else needs to be redone |
| "was collected with different settings" in a `dagger_prepare_*` log | the config or the thresholds changed after the collection started | restore the original config; ask if unsure |
| collection progress stalls (successes not rising for hours while its workers run) | something is wrong in the rollouts | read the workers' logs (`logs/dagger_<collection>_*.log`) and ask |
| `run --tasks` says "cannot run yet; copy these outputs here first" | another task's outputs are not on this machine | tell the humans exactly what it lists; they bring the archive (`vig-dagger pack … --for <this task>` on the other machine), then `vig-dagger unpack` it and run again |
| `unpack` refuses ("unsafe path") | the archive was not made by `vig-dagger pack` | ask |
| queue ends with "failed or blocked" | a failure stopped the jobs that depend on it | fix the failure, then `vig-dagger run` again; finished work is kept |

**Reading a DAgger worker's log** (`logs/dagger_<collection>_<w>.log`): one line per attempt, e.g.
`[seed 1000123] SUCCESS (success_expert) steps=231 expert=150 move=14 takeovers=1 returns=0 96.2s`. `move` counts the
random move's steps (always 0 in the `plain` variant).

## 7. When it finishes

Run `vig-dagger report -c configs/run.yaml` once more (with a board, on any machine: it copies every collection's
metrics from the board first; without one, on the machine that ran the summary task) and send the humans `<work_dir>/results/REPORT.md`, `results/figures/collections.png`, the random-move figures
`results/figures/moves/`, `results/figures/eval_success.png` and `results/summary.csv` (or the whole `results/`
folder, tens of MB). In your message, summarize:

1. **Per setting** (the report's first table, mean ± std over the 3 MoE seeds), for VIG and flow loss, plain and
   random move, at p90, p95 and p99:
   - collection success rate;
   - expert share of the DAgger data (and the random move's share);
   - expert share of the failed attempts (and the random move's share).
2. **The retrained policies** ("Retrained policies" in the report): per setting the success rate (mean ± std over the
   MoE seeds), the driving MoEs' success, and the paired comparison (only this / only MoE, p).
3. **What the seeds say:** settings where the three seeds disagree by more than ~10 points.
4. **The thresholds** of the three MoEs.
5. **Anything unusual:** failures you fixed, restarts, jobs that needed several attempts, collections that hit their
   attempt limit.

## Several machines

**With a task board** (the humans name it, e.g. "run plugcharger with the task board at gpu1:/data/vig-board"): every
machine gets the same message, runs the same config and the same `vig-dagger run`, and claims the numbered tasks
([TASKS.md](TASKS.md)) itself. How many machines there are doesn't matter; machines can join later.

- The board is a directory on one machine (`host:/path`, reached with passwordless `ssh` and `rsync`) or on a shared
  filesystem. It holds who runs which task, which tasks are done, every machine's heartbeat, and the outputs other
  machines need: the MoE checkpoints and thresholds, and every collection's metrics. DAgger episodes stay where they
  were collected.
- `vig-dagger status` (on any machine) lists every task: done, running (and on which machine), FAILED (and where),
  ready, waiting; and every machine with its last heartbeat. A machine whose heartbeat is older than 30 minutes is
  SILENT.
- **A failed task** stays marked failed on the board. Fix the cause on that machine if you can (the "you may" list)
  and restart its queue: it retries its own tasks. Otherwise report it; the humans decide whether to
  `release` it for another machine.
- **A SILENT machine:** report it. If the humans say it is gone, `vig-dagger release <task>` for each of its running
  tasks (only then); another machine takes them (a collection starts again from its first episode). If it comes back
  with the same work directory, it continues its own tasks.
- **The end:** the summary task runs on whichever machine claims it once every collection is done, and puts the final
  `results/` on the board (`<board>/files/results/REPORT.md`). Every machine's queue then ends with `[queue] every task
  of the experiment is done`.

**Without a board** (no network between the machines), the humans assign the tasks by number:

| tasks | what | needs | runs on |
|---|---|---|---|
| 1–3 | the MoEs of seeds 0, 1, 2 and their thresholds | – | any machine |
| 4–39 | the 36 collections (4–15 seed 0, 16–27 seed 1, 28–39 seed 2) | its MoE's task | any machine |
| 40 | the collections' summary | 4–39 | any machine |
| 41–76 | the retrainings: task N + 37 retrains collection task N's data and evaluates the result | collection N's episodes | the machine that ran task N |
| 77–79 | MoE seed 0, 1, 2 evaluated at full resolution | task 1, 2, 3 | where task 1, 2, 3 ran |
| 80 | the final report | 4–39 and 41–79 | any machine |

- "Run task 16" means `vig-dagger run -c configs/run.yaml --tasks 16` (step 4). It refuses if an input from another
  task is missing, and names the task and the files.
- **Moving outputs** is the humans' job: `vig-dagger pack -c configs/run.yaml --tasks 2 --for 16` writes
  `<work_dir>/packs/<name>.tar` with what task 16 needs of task 2's outputs (the MoE checkpoint, weights only, ~180 MB,
  and its thresholds); `vig-dagger unpack -c configs/run.yaml <that file>` on the machine that runs task 16. For the
  summary (task 40): `pack … --tasks 4-39 --for 40` (a few MB per collection).

## Reference

**Job names** (what `status`, `run.log` and `<work_dir>/logs/` use; `<c>` is a collection id such as
`sd1_vig_p95_random_move`):

| job | what it does |
|---|---|
| `download_demos` | download the raw planner demos |
| `convert_planner` | convert them to the training dataset (only where a MoE is trained) |
| `download_moe_sd0` | download the seed-0 MoE |
| `train/moe/sd<s>` | train the MoE of seed s |
| `score_sd<s>_<chunk>`, `thresholds_sd<s>` | demo-frame scores of MoE s and their percentiles |
| `dagger_prepare_<c>` | resolve the collection's threshold, write `settings.json` |
| `dagger_<c>_<worker>` | DAgger workers, sharing one collection |
| `analyze_<c>` | the collection's metrics |
| `summarize` | every collection side by side, the figure, REPORT.md |
| `dagger_dataset_<c>` | a retraining's training set: base demos + the collection's DAgger data |
| `train/retrain/<c>/seed<s>` | the dense policy retrained on it |
| `eval/<policy>/<first scene>` | 25 evaluation episodes of a retrained policy or a MoE |
| `final_report` | the evaluation summary and the full report |
| `publish/<task>`, `fetch/<task>` | with a board: copy a task's outputs to the board / from it |

**Useful commands:**

- `vig-dagger run -c … --dry-run`: every job with its priority and dependencies (`done` marks finished ones).
- `vig-dagger run -c … --only REGEX`: run a subset and what it needs.
- `vig-dagger stop -c …`: stop everything. With `--queue-only`, only the queue stops.
- `vig-dagger show-config -c …`: the resolved configuration.

**Where things are:** see "Output directory" in the README. A training's progress is in
`policies/moe/sd<s>/metrics.jsonl` (`train/step`, `train/loss`, `val/loss`), summarized by `vig-dagger report`.
