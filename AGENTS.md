# Running PlugCharger round 2: instructions for an agent

These instructions are for an agent (or a person) who runs `configs/plugcharger_round2.yaml` (branch
`plugcharger-round2`) on one of several machines that share the work through Hugging Face. The README explains what
the pipeline does; this file says how to operate it: what to run, what a healthy run looks like, when to stop and
ask, and what never to change.

This branch runs **round 2** (`per_resolution`) of the PlugCharger experiment only. Round 1 runs elsewhere, on
branch `main`; round 2's results are merged into that run at the end (see [When it finishes](#7-when-it-finishes)).

## Short commands

The humans start and check the run with short messages. The wording varies ("start round 2", "how is it going");
go by the intent, and ask only if you can't tell which of these is meant:

| the human says | do |
|---|---|
| "run round 2" | set up, check and launch this machine (steps 0–4), then report. Every machine gets the same message: the machines share the work through Hugging Face, however many there are (see [Several machines](#several-machines)) |
| "status" | `vig-dagger status`, `tail run.log`, `df -h <work_dir>`; summarize: this machine's queue, round 2's tasks (done, running and where, failed), every machine's heartbeat |
| "report" | `vig-dagger report` (it collects every machine's results from the shared repo), then send `results/REPORT.md` and summarize it (step 7) |
| "stop" | `vig-dagger stop` (this machine's jobs pause, and its tasks wait for it; "run round 2" again resumes) |
| "release *TASK*" | `vig-dagger release TASK`: another machine may then take the task (see [Several machines](#several-machines)) |

If an earlier session already set up and launched this machine (there is a `configs/run.yaml` and a work directory),
"run round 2" means: start the queue again if it is not running (step 4), then report the status. Never reinstall
or reconfigure a machine that has started.

## The goal and what "done" means

- **The run:** round `per_resolution` of the PlugCharger experiment, with exactly the settings of
  `configs/plugcharger.yaml`. Seven dense policies are trained on the 1000 planner demos, one per image resolution
  (full resolution: `base` seed 0; `res2` … `res64`), 60 epochs each. The full-resolution one drives DAgger, the
  others give VIG its degraded-level samples. Their thresholds come from 50 demos. Then 500 successful DAgger
  episodes with VIG and 500 with flow loss, 3 retrainings on each (`base_plus_per_resolution_{vig,flow}`), and 200
  evaluation rollouts per policy (`res2` … `res64` at their own resolution, the 6 retrainings at full resolution)
  on the same scenes as round 1.
- **Done when** (on any machine):
  - `vig-dagger status` shows every task `done` (its queue then ends by itself: "every task of the experiment is
    done");
  - `vig-dagger report` writes `<work_dir>/results/REPORT.md` with round 2's evaluation and DAgger tables.

Every result is written to files in `<work_dir>/results` and uploaded to the shared repo; W&B is not used.
- **Then report** the numbers listed in [When it finishes](#7-when-it-finishes).

Expect about 2 days on two machines with two RTX 5090s each, 3–4 days on one such machine. Most of the time goes
into 13 trainings (7 of 1000 demos, 6 of 1500) and the two DAgger collections.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `rounds`,
  `split`, `drivers`, `scorers`, `thresholds`, `dagger`, `retrain`, `evaluation`, `sharing`. These define the
  experiment, and round 2's results can only be merged into round 1's if they match `configs/plugcharger.yaml`.
- **Delete or move anything in `work_dir`**, and above all `data/`, `dagger/`, `policies/`. Collections and
  trainings take days; partial outputs are what lets a run resume.
- **Upload** anything to Hugging Face by hand (the queue uploads the run's outputs to the shared repo itself),
  **push** to git, or turn on W&B.
- **Release a task** (`vig-dagger release`) unless the humans confirm that the machine running it is down.
- **Run two queues on the same `work_dir`**, or start workers by hand outside the queue.
- **Edit the code.**

**You may:**

- set `work_dir` before the first start;
- log in to Hugging Face with the token the humans give you (`hf auth login --token <token>`). Never write the token
  into the repository, the config, a log or a message;
- change `resources.*` settings, e.g. fewer `sim_slots` when the machine is shared. To apply them without
  interrupting jobs: `vig-dagger stop --queue-only`, edit the config, then `vig-dagger run` again;
- stop and restart the queue (`vig-dagger stop`, `vig-dagger run`);
- rerun `vig-dagger analyze --round per_resolution` and `vig-dagger summarize` at any time;
- read any file or log;
- regenerate the report with `vig-dagger report` at any time.

**Stop and ask the humans when:**

- the human's message has no Hugging Face token and this machine is not logged in (step 0);
- `vig-dagger check` reports a failure you cannot fix by installing something: no Vulkan, no write access to the
  shared repo, not enough disk;
- a job is `FAILED` again after you fixed what its log showed and restarted once, or you can't tell why it failed;
- `status` shows a task `FAILED`, or a machine `SILENT` (no heartbeat for 90 minutes);
- a DAgger collection is heading for its attempt limit (10,000 attempts per collection). Concretely: after 300
  attempts it has fewer than 15 successes (a success rate under 5 %);
- free disk in `work_dir` drops below 30 GB;
- a training's loss is NaN, or its validation loss rises steadily over the second half of training;
- a milestone value is far outside the expected range below.

## 0. Before you start

**Ask only for the Hugging Face token**, and only if this machine is not logged in yet (`hf auth whoami` fails) and
the human's message doesn't contain one: every machine needs write access to the shared repo
(`chengenwu/vig-dagger-plugcharger-round2`). Log in with `hf auth login --token <token>` and never repeat the token.

**Everything else has a default.** Use it unless the human's message says otherwise, and say in your first report
which values you used:

| setting | default |
|---|---|
| work directory | `~/vig-dagger-runs/plugcharger_round2` if that disk has at least 150 GB free; otherwise a directory you can write to on a disk that has; if there is none, ask |
| GPUs | every GPU (`resources.gpus: auto`); one training per GPU |
| CPU budget | the default (`resources.sim_slots: auto`, half the cores this machine may use; in a container, its own limit, not the host's). If other users' jobs already load the machine, halve it and say so |
| reporting | a message after the launch, after each milestone you see, and the final report (step 7) |

Say in your first report that the run takes days and keeps going after your session ends; the humans check on it
with "status".

## 1. Setup

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout plugcharger-round2
uv sync
source .venv/bin/activate
hf auth whoami || hf auth login --token <token>    # the token from the humans (step 0)
```

`uv`, `git` and `ffmpeg` are the programs to install; rendering needs the Vulkan loader (`libvulkan1`). If one is
missing and installing it needs administrator rights, ask. In a Docker container (e.g. a rented cloud GPU), the
container needs `NVIDIA_DRIVER_CAPABILITIES=all` for Vulkan: `vulkaninfo --summary` must list the GPU. If it doesn't,
tell the humans; it is the container's setting, not something to fix here.

## 2. Configure

Copy `configs/plugcharger_round2.yaml` to `configs/run.yaml`. Set `work_dir` (step 0). If the human named GPUs or a
CPU budget, set `resources.gpus` / `resources.sim_slots`. Leave everything else as it is: every machine must use the
same settings, or it refuses to join the shared repo.

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable. It checks:
- the GPUs, and that torch sees CUDA;
- rendering (Vulkan);
- `ffmpeg`;
- free disk;
- that the Hugging Face files exist (the demos);
- the Hugging Face login with write access, and that the shared repo holds this experiment (the first machine
  creates it);
- the W&B setting (it should say `W&B disabled`).

It prints the CPU and GPU budget the queue will use, and the container's limits if it runs in one.

On a new machine, also run the smoke test (a few minutes, on this machine alone, writes only to the given
directory):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

It must end with `[queue] finished: 80 of 80 jobs done`. Delete `/tmp/vig-dagger-smoke` afterwards.

## 4. Launch

```bash
setsid nohup vig-dagger run -c configs/run.yaml > run.log 2>&1 < /dev/null &
```

`run.log` is the queue's log: every task it claims, every job it starts, finishes, retries or gives up on, and every
upload. Each job writes its own log to `<work_dir>/logs/<job name>.log`.

## 5. Monitor

About every hour:

```bash
vig-dagger status -c configs/run.yaml    # round 2's tasks, where they run, every machine's heartbeat
tail -n 30 run.log                       # this machine's queue
df -h <work_dir>
```

About once a day, or after a milestone, run `vig-dagger report -c configs/run.yaml`. It refreshes
`<work_dir>/results/REPORT.md` and its figures with everything finished so far, on every machine.

If this machine's queue is not running (no new lines in `run.log`, no `vig-dagger run` process) and tasks remain,
start it again with the command from step 4. It takes over jobs that are still running.

### Milestones and expected values

| milestone | when | what to check | expected |
|---|---|---|---|
| demos downloaded | when this machine first needs them | `<work_dir>/data/raw/planner/` | 20 `part_*.h5` (26 GB) |
| dataset converted | right after | `num_demos` in `data/datasets/base/dataset_manifest.json` | 1000 |
| the seven trainings | at once, one per GPU over the machines | `policies/retrain/base/seed0/`, `policies/drivers/per_resolution/res{2..64}/`; the Trainings table of `vig-dagger report` | 188,580 updates each; validation loss falling to ~0.02 or below |
| thresholds | after all seven | `policies/drivers/per_resolution/thresholds.json` (on the machine that computed them, and in the repo) | VIG and flow p95 well above p50, all positive. For scale: round 1's MoE gave VIG p50 0.0112 / p95 0.0327, flow 0.0127 / 0.0341 |
| DAgger | right after, each scorer on one machine | `status` progress; `dagger/per_resolution/<scorer>/settings.json` there | take/return thresholds equal the p95/p50 above; successes rising steadily |
| DAgger metrics | when a collection finishes | `results/dagger_per_resolution_<scorer>.json` and `_page.json.gz` (the data for a page of the collection); the DAgger tables and figures in `REPORT.md` | collection success rate, expert share, takeovers: new numbers. Round 1 for scale: 92 % success, 62–65 % expert share |
| retraining | after each collection | `policies/retrain/base_plus_per_resolution_{vig,flow}/seed*/` | about 280–330k updates |
| evaluations | as each policy finishes | `eval/<policy>/level<f>/*.jsonl` | 200 episodes per policy, 2,400 in all |
| summary | at the end | `results/eval_summary.json`, `results/REPORT.md` | mean ± std over 3 seeds per training set |

A training ran at 4.5–13 updates per second on a shared RTX PRO 6000 machine; an RTX 5090 should be in the upper part
of that range.

## 6. When something goes wrong

| symptom | likely cause | what to do |
|---|---|---|
| `status` lists a FAILED task | see the job's log on the machine that ran it (`run.log` names it) | fix the cause if it is in the "you may" list, restart that machine's queue, then ask the humans before `release` |
| a job retries again and again | the same error each time | read the log; don't wait for the retries to run out |
| `CUDA out of memory` | other jobs filled the GPU | nothing; the retry runs when memory frees up. If it keeps happening, ask whether to raise `resources.train_gpu_memory_gb` |
| `[board] Hugging Face's commit limit is reached` | 128 commits per hour per repo | nothing; claims and heartbeats pause for 10 minutes, uploads wait and retry |
| `holds an experiment with other settings` | this machine's config differs from the others' | restore `configs/run.yaml` from `configs/plugcharger_round2.yaml` (only `work_dir` and `resources` may differ) |
| `vk::`, "no device", rendering errors | Vulkan, or the container's driver capabilities | ask (needs a system or container fix) |
| `No space left on device` | disk full | stop the queue (`vig-dagger stop`) and ask; never delete experiment outputs to make room |
| machine rebooted | | start it again; trainings resume from their last checkpoint (at most 5000 updates lost) |
| collection progress stalls (successes not rising for hours while its workers run) | something is wrong in the rollouts | read the workers' logs (`logs/dagger_per_resolution_<scorer>_*.log`) and ask |

## 7. When it finishes

Run `vig-dagger report -c configs/run.yaml` once more (on any machine) and send the humans
`<work_dir>/results/REPORT.md` and the `results/figures/` folder. In your message, summarize:

1. **Evaluation:** for `base_plus_per_resolution_vig` and `base_plus_per_resolution_flow`, the success rate of each
   seed and their mean ± std; and `res2` … `res64` at their own resolution. (The comparison with `base` and round 1
   comes after the merge below.)
2. **For each scorer** (the DAgger table): collection success rate, pooled expert share, median expert share and
   median takeovers per episode.
3. **Anything unusual:** failures you fixed, restarts, tasks that were released, jobs that needed several attempts.

**Merging into round 1** (on the round-1 machine, branch `main`; it only needs to download, not upload):

```bash
hf download chengenwu/vig-dagger-plugcharger-round2 --repo-type dataset --local-dir round2_results \
  --include "config.json" "results/*" "results/figures/*" "eval/*" "eval/**" \
            "policies/**/metrics.jsonl" "policies/**/run_config.json" "policies/drivers/per_resolution/*.json" \
            "policies/drivers/per_resolution/*.npz"
vig-dagger merge -c configs/run.yaml round2_results
```

`merge` refuses unless both runs have the same experiment settings; it copies round 2's evaluations, DAgger metrics
and training logs in and refreshes that run's `eval_summary.json` and `REPORT.md`.

## Several machines

Every machine runs the same thing: the same branch, the same `configs/run.yaml` settings (only `work_dir` and
`resources` may differ), the same `vig-dagger run`. One machine works; so do two or five, and a machine can join at
any time. A machine with several GPUs runs one training on each at the same time. The machines share the run through
the public Hugging Face repo in `sharing.repo` (`chengenwu/vig-dagger-plugcharger-round2`):

- The work comes in **tasks**: each training (13), the thresholds, each DAgger collection (2), each evaluation of a
  policy (12), and the summary.
- A machine with room **claims** the most urgent task whose inputs are all in the repo (a claim can't be taken twice:
  it is a commit that fails if another machine committed first), downloads the inputs (`fetch/…` jobs), runs the
  task, and **uploads** its outputs (`publish/…` jobs): checkpoints (weights only), the DAgger training sets,
  attempts, trajectories and metrics, evaluation episodes. Each machine downloads and converts the demos itself when
  one of its tasks needs them. It claims one task at a time, the next once the last one has started, as long as it
  has room (a free GPU for a training, CPU budget for a DAgger collection or an evaluation).
- **What waits:** the thresholds need all seven trainings, a DAgger collection needs the thresholds, a retraining
  needs its collection, an evaluation needs its policy. Such a task runs on one machine while the others take other
  tasks or wait for the upload. A DAgger collection uses its machine's whole CPU budget.
- Each machine sends a **heartbeat** every 30 minutes; `status` (on any machine) shows every task and every machine.
- A machine's queue ends by itself when every task is done; `vig-dagger report` (on any machine) then writes the full
  report from the repo.

**A machine that goes `SILENT`** (three heartbeats missed) still holds its tasks, and nobody else takes them. Ask the
humans whether it is down. Once they confirm, run `vig-dagger release <task>` for each task `status` lists as running
on it; any machine then takes the task and starts it from the beginning (the checkpoints for resuming a training
stay on the machine that held it). If the machine comes back, its queue sees that the task is no longer its own
and leaves it.

**A task that `FAILED`** stays with its machine until it is released: fix the cause there, restart that machine's
queue, then (with the humans' OK) `vig-dagger release <task>`.

## Reference

**Tasks and their jobs** (what `status`, `run.log` and `<work_dir>/logs/` use):

| task | jobs | what it does |
|---|---|---|
| (local preparation) | `download_demos`, `convert_planner` | download the planner demos, convert them to the `base` training set |
| `train/retrain/base/seed0` | same name | round 2's full-resolution driver (not evaluated here: round 1's own `base` seed 0 is the evaluated one) |
| `train/drivers/per_resolution/res<k>` | same name | the degraded-level policies |
| `thresholds/per_resolution` | `score_per_resolution_<chunk>`, `thresholds_per_resolution` | demo-frame scores and their percentiles |
| `dagger/per_resolution/<scorer>` | `dagger_prepare_…`, `dagger_per_resolution_<scorer>_<worker>`, `dagger_dataset_…`, `analyze_…` | one DAgger collection: its workers, the first 500 successes as a training set, its metrics |
| (local preparation) | `combine_per_resolution_<scorer>` | the retraining set: base + that DAgger set |
| `train/retrain/base_plus_per_resolution_<scorer>/seed<s>` | same name | the evaluated retrainings |
| `eval/<policy>/level<f>` | `eval/<policy>/level<f>/<first seed>` | 200 evaluation episodes in chunks of 25 |
| `summary` | `summarize` | round 2's tables and figures, uploaded with this run's `config.json` |
| every task T | `publish/T`, `fetch/T` | upload T's outputs / download them where another task needs them |

**Useful commands:**

- `vig-dagger run -c … --dry-run`: every job with its priority and dependencies (`done` marks finished ones).
- `vig-dagger stop -c …`: stop everything. With `--queue-only`, only the queue stops.
- `vig-dagger release -c … TASK`: free a task whose machine is gone (only with the humans' OK).
- `vig-dagger show-config -c …`: the resolved configuration.

**Where things are:** see "Output directory" in the README. A training's progress is in
`policies/.../metrics.jsonl` (`train/step`, `train/loss`, `val/loss`), summarized by `vig-dagger report`.
