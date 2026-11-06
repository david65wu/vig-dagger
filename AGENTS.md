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
| "run task 2", "run tasks 1, 4–15" | only these tasks of the numbered list ([TASKS.md](TASKS.md), see [Several machines](#several-machines)): set the machine up if it isn't yet (steps 0–3), then launch with `--tasks` (step 4) and report. If `run` refuses because inputs are missing, report exactly what it lists |
| "pack task 2 for task 16" | `vig-dagger pack -c configs/run.yaml --tasks 2 --for 16`; report the archive's path and size (the humans move it) |
| "unpack *FILE*" | `vig-dagger unpack -c configs/run.yaml FILE`, then report what it put in place |
| "tasks" | `vig-dagger tasks -c configs/run.yaml` (the numbered list, as in TASKS.md) |
| "status" | `vig-dagger status`, `tail run.log`, `df -h <work_dir>`; summarize: queue alive, failed jobs, progress per collection, which task numbers are done here |
| "report" | `vig-dagger report`, then send `results/REPORT.md` and `results/figures/collections.png` and summarize them (step 7) |
| "stop" | `vig-dagger stop` (every job pauses; the same "run" again resumes) |

If an earlier session already set up and launched this machine (there is a `configs/run.yaml` and a work directory),
"run plugcharger" means: start the queue again if it is not running (step 4), then report the status. Never
reinstall or reconfigure a machine that has started.

## The goal and what "done" means

- **The run:** PlugCharger in ManiSkill. Three 7-level MoE policies that differ only in their seed: seed 0 is
  downloaded from Hugging Face, seeds 1 and 2 are trained here (60 epochs on the 1000 planner demos, 188,580 updates
  each). Each MoE gets VIG and flow-loss thresholds (p90, p95, p99 of its score on demo frames). Then 36 DAgger
  collections of 500 successful episodes each: 3 MoE seeds × {VIG, flow loss} × {p90, p95, p99} × {plain, random
  move}. One threshold decides both directions (take over after 2 scores above, hand back after 16 below).
- **The results:** per collection, the collection success rate, the expert share of the DAgger data and the expert
  share of the failed attempts (with the random move's share separate), and per setting their mean ± std over the
  three seeds.
- **Done when** (on one machine; with several, each machine is done when its assigned tasks are, and the experiment
  when the summary task has run):
  - `vig-dagger status` shows every job `done`;
  - `<work_dir>/results/summary.json` says 36 collections finished;
  - `<work_dir>/results/REPORT.md` has the tables filled in and `results/figures/collections.png` exists.

Every result is written to files in `<work_dir>/results`; W&B is not used (`wandb.enabled: false`).
- **Then report** the numbers listed in [When it finishes](#7-when-it-finishes).

Expect about 3 days on one 64-core machine with 2 GPUs (most of it the 36 collections, ~20–60 worker-hours each),
less on several machines.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `moe`,
  `scorers`, `thresholds`, `dagger`. These define the experiment. A changed value either silently mixes two
  experiments, or the pipeline refuses to continue.
- **Delete or move anything in `work_dir`**, and above all `dagger/`, `policies/`, `data/`. Collections take a day
  or more; partial outputs are what lets a run resume.
- **Upload** anything to Hugging Face, **push** to git, or turn on W&B.
- **Run tasks the humans did not assign to this machine**, or copy files between machines yourself: the humans
  assign the tasks and move the archives.
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

**Everything has a default.** Use it unless the human's message says otherwise, and say in your first report which
values you used:

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

Copy `configs/plugcharger.yaml` to `configs/run.yaml`. Set `work_dir` (step 0). If the human named GPUs or a CPU
budget, set `resources.gpus` / `resources.sim_slots`. Leave everything else as it is: with several machines, every
machine must use the same settings, or the task numbers and their outputs won't match.

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable (a disk WARN means the disk holds only a few collections:
fine for a machine that runs a few tasks, not for one that runs everything). It checks:
- free disk, `ffmpeg`, the GPUs, and that torch sees CUDA;
- rendering (Vulkan);
- that the demos and the seed-0 MoE exist on Hugging Face, and the DINOv2 weights are reachable;
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

For assigned tasks, add `--tasks` (and use a log file of their own, e.g. `run_tasks_2_16-27.log`):
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
| MoE seeds 1, 2 trainings | 4–12 h each, in parallel on two GPUs | `policies/moe/sd{1,2}/metrics.jsonl`, `run_config.json` → `training.max_updates`; the Trainings table of `vig-dagger report` | exactly 188,580 updates; validation loss falling to about seed 0's (its final value is in the same table) |
| thresholds | ~30 min after each MoE | `policies/moe/sd<s>/thresholds.json` | all positive, p90 < p95 < p99. For scale, seed 0 measured earlier: VIG p90 ≈ 0.026, p95 ≈ 0.032, p99 ≈ 0.044; flow p90 ≈ 0.027, p95 ≈ 0.035, p99 ≈ 0.056 (seeds 1 and 2 should be similar) |
| collections | seed 0's from ~1 h on; seeds 1 and 2 after their trainings | `dagger/<collection>/settings.json`; `status` progress | take and return thresholds both equal the collection's percentile; successes rising steadily. For scale, an earlier run of seed 0 with a faster expert: VIG success rate 63–95 % per attempt, flow 35–41 %; lower thresholds give higher expert shares |
| collection metrics | as each collection finishes | `results/collections/<collection>.json`; the tables and `figures/collections.png` in `results/REPORT.md` | the experiment's new numbers |
| summary | at the end | `results/summary.json` | `collections_finished` 36 |

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
| "was collected with different settings" in a `dagger_prepare_*` log | the config or the thresholds changed after the collection started | restore the original config; ask if unsure |
| collection progress stalls (successes not rising for hours while its workers run) | something is wrong in the rollouts | read the workers' logs (`logs/dagger_<collection>_*.log`) and ask |
| `run --tasks` says "cannot run yet; copy these outputs here first" | another task's outputs are not on this machine | tell the humans exactly what it lists; they bring the archive (`vig-dagger pack … --for <this task>` on the other machine), then `vig-dagger unpack` it and run again |
| `unpack` refuses ("unsafe path") | the archive was not made by `vig-dagger pack` | ask |
| queue ends with "failed or blocked" | a failure stopped the jobs that depend on it | fix the failure, then `vig-dagger run` again; finished work is kept |

**Reading a DAgger worker's log** (`logs/dagger_<collection>_<w>.log`): one line per attempt, e.g.
`[seed 1000123] SUCCESS (success_expert) steps=231 expert=150 move=14 takeovers=1 returns=0 96.2s`. `move` counts the
random move's steps (always 0 in the `plain` variant).

## 7. When it finishes

Run `vig-dagger report -c configs/run.yaml` once more (with several machines: on the machine that ran the summary
task) and send the humans `<work_dir>/results/REPORT.md`, `results/figures/collections.png`, the random-move figures
`results/figures/moves/` and `results/summary.csv` (or the whole `results/` folder, tens of MB). In your message, summarize:

1. **Per setting** (the report's first table, mean ± std over the 3 MoE seeds), for VIG and flow loss, plain and
   random move, at p90, p95 and p99:
   - collection success rate;
   - expert share of the DAgger data (and the random move's share);
   - expert share of the failed attempts (and the random move's share).
2. **What the seeds say:** settings where the three seeds disagree by more than ~10 points.
3. **The thresholds** of the three MoEs.
4. **Anything unusual:** failures you fixed, restarts, jobs that needed several attempts, collections that hit their
   attempt limit.

## Several machines

The humans may split the experiment over several machines and assign the work themselves. The work is a numbered
list of 40 tasks, [TASKS.md](TASKS.md) (`vig-dagger tasks` prints the same list):

| tasks | what | needs |
|---|---|---|
| 1 | MoE seed 0: download, thresholds (under an hour, 1 GPU) | – |
| 2, 3 | MoE seeds 1, 2: training (8–14 h on one GPU), thresholds | – |
| 4–15 | seed 0's 12 collections | task 1 |
| 16–27 | seed 1's 12 collections | task 2 |
| 28–39 | seed 2's 12 collections | task 3 |
| 40 | the summary | the metrics of tasks 4–39 |

- Every machine has its own clone, environment and work directory, and the same `configs/run.yaml` settings (only
  `work_dir` and `resources` may differ), so the task numbers mean the same everywhere.
- "Run task 16" means `vig-dagger run -c configs/run.yaml --tasks 16` (step 4). It refuses if an input from another
  task is missing, and names the task and the files.
- **Moving outputs** is the humans' job. On the machine that ran a task: `vig-dagger pack -c configs/run.yaml --tasks
  2 --for 16` writes `<work_dir>/packs/<name>.tar` with what task 16 needs of task 2's outputs (the MoE checkpoint,
  weights only, ~180 MB, and its thresholds). On the machine that runs task 16: `vig-dagger unpack -c
  configs/run.yaml <that file>`. Without `--for`, `pack` takes all of the task's outputs (for a collection: its
  ~15 GB of episodes too). Every file keeps its path relative to the work directory, so copying the same files by any
  other means works too.
- Every machine downloads the demos (and the DINOv2 weights) from Hugging Face itself when one of its tasks needs
  them. Nothing is uploaded.
- **The summary** (task 40) needs every collection's metrics, e.g. from another machine
  `vig-dagger pack … --tasks 4-39 --for 40` (it packs what that machine has). Its report on that machine is the
  experiment's final report (step 7); add `pack --tasks 1-3 --for 40` (a few MB: the training logs and thresholds) to
  fill the report's threshold and training tables.

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

**Useful commands:**

- `vig-dagger run -c … --dry-run`: every job with its priority and dependencies (`done` marks finished ones).
- `vig-dagger run -c … --only REGEX`: run a subset and what it needs.
- `vig-dagger stop -c …`: stop everything. With `--queue-only`, only the queue stops.
- `vig-dagger show-config -c …`: the resolved configuration.

**Where things are:** see "Output directory" in the README. A training's progress is in
`policies/moe/sd<s>/metrics.jsonl` (`train/step`, `train/loss`, `val/loss`), summarized by `vig-dagger report`.
