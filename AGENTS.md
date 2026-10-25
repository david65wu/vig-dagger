# Running the experiment: instructions for an agent

These instructions are for an agent (or a person) who runs `configs/opendrawer.yaml` (branch `opendrawer`) from
start to finish without supervision. The README explains what the pipeline does; this file says how to operate it:
what to run, what a healthy run looks like, when to stop and ask, and what never to change.

## Short commands

The humans start and check the run with short messages. The wording varies ("start opendrawer", "how is it going");
go by the intent, and ask only if you can't tell which of these is meant:

| the human says | do |
|---|---|
| "run opendrawer" | set up, check and launch this machine (steps 0–4), then report. Every machine gets the same message: the machines share the work through Hugging Face, however many there are (see [Several machines](#several-machines)) |
| "status" | `vig-dagger status`, `tail run.log`, `df -h <work_dir>`; summarize: this machine's queue, the experiment's tasks (done, running and where, failed), every machine's heartbeat |
| "report" | `vig-dagger report` (it collects every machine's results from the shared repo), then send `results/REPORT.md` and summarize it (step 7) |
| "stop" | `vig-dagger stop` (this machine's jobs pause, and its tasks wait for it; "run opendrawer" again resumes) |
| "release *TASK*" | `vig-dagger release TASK`: another machine may then take the task (see [Several machines](#several-machines)) |

If an earlier session already set up and launched this machine (there is a `configs/run.yaml` and a work directory),
"run opendrawer" means: start the queue again if it is not running (step 4), then report the status. Never reinstall
or reconfigure a machine that has started.

## The goal and what "done" means

- **The run:** OpenDrawer in Isaac Lab. The pipeline downloads 100 planner demos (plus 100 for a baseline) from
  Hugging Face and trains every policy itself, 150 epochs each. Then two rounds of DAgger (round `moe`, then round
  `per_resolution`), each collecting 100 successful episodes with VIG and 100 with flow loss. 18 dense
  full-resolution policies are trained and evaluated (6 training sets × 3 seeds), and so are the driving policies:
  the MoE at each of its 7 levels, and round 2's `res2` … `res64` at their own levels. Every evaluation is 200
  rollouts on the same scenes.
- **Done when** (on any machine):
  - `vig-dagger status` shows every task `done` (its queue then ends by itself: "every task of the experiment is
    done");
  - `vig-dagger report` writes `<work_dir>/results/REPORT.md` with the evaluation tables filled in.

Every result is written to files in `<work_dir>/results`; W&B is not used (`wandb.enabled: false`).
- **Then report** the numbers listed in [When it finishes](#7-when-it-finishes).

Expect it to take about a week on one machine with 2 GPUs, and about 1–1.5 days on five or six such machines. Most
of the time goes into 25 trainings of 3–20 h each.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `rounds`,
  `drivers`, `scorers`, `thresholds`, `dagger`, `retrain`, `evaluation`. These define the experiment. A changed
  value either silently mixes two experiments, or the pipeline refuses to continue.
- **Delete or move anything in `work_dir`**, and above all `data/`, `dagger/`, `policies/`. Collections and
  trainings take days; partial outputs are what lets a run resume.
- **Upload** anything to Hugging Face by hand (the queue uploads the run's outputs to the shared repo itself),
  **push** to git, or turn on W&B.
- **Release a task** (`vig-dagger release`) unless the humans confirm that the machine running it is down.
- **Run two queues on the same `work_dir`**, or start workers by hand outside the queue.
- **Edit the code**, including the planner and the environment.

**You may:**

- set `work_dir` before the first start;
- log in to Hugging Face with the token the humans give you (`hf auth login --token <token>`). Never write the token
  into the repository, the config, a log or a message;
- change `resources.*` settings, e.g. fewer `sim_slots` when the machine is shared. To apply them without
  interrupting jobs: `vig-dagger stop --queue-only`, edit the config, then `vig-dagger run` again;
- stop and restart the queue (`vig-dagger stop`, `vig-dagger run`);
- rerun `vig-dagger analyze --round <r>` and `vig-dagger summarize` at any time;
- read any file or log;
- regenerate the report with `vig-dagger report` at any time.

**Stop and ask the humans when:**

- before installing: they have not yet agreed to NVIDIA's Isaac Sim license (see step 0);
- `vig-dagger check` reports a failure you cannot fix by installing something: Isaac Sim does not start or renders
  nothing, the GPU has no RT cores, no internet access (simulator jobs load the robot and cabinet models from
  NVIDIA's asset server), not enough disk;
- a job is `FAILED` again after you fixed what its log showed and restarted once, or you can't tell why it failed;
- `status` shows a task `FAILED`, or a machine `SILENT` (no heartbeat for 90 minutes);
- `download_demos` fails and the log shows a network or permission error you cannot fix;
- a DAgger collection is heading for its attempt limit (2,000 attempts per collection). Concretely: after 200
  attempts it has fewer than 10 successes (a success rate under 5 %);
- free disk in `work_dir` drops below 30 GB;
- a training's loss is NaN, or its validation loss rises steadily over the second half of training;
- a milestone value is far outside the expected range below.

## 0. Before you start

**Ask only for these two**, and only if the human's message doesn't already give them:

- **NVIDIA's license:** whether they agree to the
  [NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
  The install script downloads Isaac Sim, and every simulator job accepts the license non-interactively
  (`OMNI_KIT_ACCEPT_EULA=YES`). The human must agree to it; never accept it on their behalf.
- **A Hugging Face access token with write access:** the machines share the run through a public Hugging Face repo
  (see [Several machines](#several-machines)). First check `hf auth whoami`: if this machine is already logged in
  (the humans may have done it themselves), don't ask. Otherwise log in with the token they give you
  (`hf auth login --token <token>`) and never write it anywhere else.

**Everything else has a default.** Use it unless the human's message says otherwise, and say in your first report
which values you used:

| setting | default |
|---|---|
| work directory | `~/vig-dagger-runs/opendrawer` if that disk has at least 120 GB free; otherwise a directory you can write to on a disk that has; if there is none, ask. The repository needs about 30 GB more for its two Python environments |
| GPUs | every GPU (`resources.gpus: auto`). They must be RTX-class (with RT cores): an A100 or H100 cannot run Isaac Sim's renderer; if that is all there is, ask |
| CPU budget | the default (`resources.sim_slots: auto`, a third of the cores' worth of simulator jobs). If other users' jobs already load the machine (`uptime` load average above half of `nproc`), halve it and say so |
| reporting | a message after the launch, after each milestone you see, and the final report (step 7) |

Say in your first report that the run takes days (about a week on one machine, about 1–1.5 days on five or six) and
keeps going after your session ends; the humans check on it with "status".

## 1. Setup

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout opendrawer
uv sync                          # the main environment (.venv)
bash scripts/install_isaac.sh    # the Isaac environment (.venv-isaac): Isaac Sim 5.1 + Isaac Lab, 10-30 min
source .venv/bin/activate
```

Programs needed: `uv`, `git`, `ffmpeg`. If one is missing and installing it needs administrator rights, ask. The
install script ends with a line like `isaacsim 5.1.0.0 | isaaclab 0.54.2 | torch 2.7.0+cu128 | CUDA True`; if it
fails, read its output, fix what you can (network, disk), rerun it (it is safe to rerun), and otherwise ask.

## 2. Configure

Copy `configs/opendrawer.yaml` to `configs/run.yaml`. Set `work_dir` (step 0). If the human named GPUs or a CPU
budget, set `resources.gpus` / `resources.sim_slots`. Leave everything else as it is: every machine of the experiment
uses the same settings and the same shared repo (`sharing.repo`), and a machine with other settings is refused.

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable. It checks:
- free disk, `ffmpeg`, the GPUs, and that torch sees CUDA;
- the Isaac environment: it starts Isaac Sim once, builds the task, renders the two policy cameras, and checks that
  the policy's state input is the TCP frame. The first start on a machine compiles shaders and can take up to ~15
  minutes; the full output goes to `<work_dir>/logs/check_isaac.log`;
- that the demos exist on Hugging Face (`chengenwu/OpenDrawer-ManagerBased-v0`, public) and the DINOv2 weights
  are reachable;
- the Hugging Face login (write access) and the shared repo (`chengenwu/vig-dagger-opendrawer-v1`, public; the first
  machine creates it): it must hold this experiment's settings;
- the W&B setting (it should say `W&B disabled`).

It prints the CPU and GPU budget the queue will use.

On a new machine, also run the smoke test (about 15 minutes, writes only to the given directory):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

It must end with `[queue] finished: 80 of 80 jobs done`. Its success rates are 0 % (its policies train for a few
dozen updates); that is expected. Delete `/tmp/vig-dagger-smoke` afterwards.

## 4. Launch

```bash
setsid nohup vig-dagger run -c configs/run.yaml > run.log 2>&1 < /dev/null &
```

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
`<work_dir>/results/REPORT.md` and its figures with everything finished so far: training curves, DAgger tables,
evaluations.

`status` shows:
- whether the queue is RUNNING (its heartbeat is less than 2 minutes old);
- every FAILED job with its log path;
- every running job with its GPU, run time and progress (training updates, DAgger successes and attempts);
- the experiment's tasks on every machine: how many are done, which run where (a machine that has missed three
  heartbeats is marked `SILENT`), which `FAILED`, which are ready for any machine, which wait for inputs;
- every machine with its last heartbeat and what it runs.

If the queue is NOT RUNNING and jobs remain, start it again with the command from step 4. It takes over jobs that
are still running.

**Reading a simulator job's log** (`dagger_*`, `eval/*`): Isaac Sim writes hundreds of harmless
lines first (`Failed to create change watch … errno=28/No space left on device`, GLFW, display, DLSS, IOMMU
warnings). Skip them: `grep -v "change watch" <log> | grep -E "^\[seed|^\[worker|Traceback|Error:"` shows the
episodes and any real error.

### Milestones and expected values

| milestone | when | what to check | expected |
|---|---|---|---|
| demos downloaded | when this machine first needs them (a training or a round's thresholds) | `<work_dir>/data/raw/planner/`, `planner_extra/` (links into `hub/dataset/raw/`) | 4 `part_*.hdf5` (7.5 GB) and 4 `part_*.done.json` in each |
| datasets converted | right after | `num_demos` in each `data/datasets/<name>/dataset_manifest.json` | planner 100, base 100, planner_extra 100, base_plus_planner 200 |
| round `moe` driver | 3–10 h | `policies/drivers/moe/metrics.jsonl`, then `latest.pt`; the Trainings table of `vig-dagger report` | ~160k updates (exactly: `run_config.json` → `training.max_updates`); validation loss falling |
| baseline trainings | start with the MoE, on the other GPUs | `policies/retrain/{base,base_plus_planner}/seed*/` | ~160k updates (100 demos) or ~320k (200 demos) |
| round `moe` thresholds | ~30 min after the MoE | `policies/drivers/moe/thresholds.json` | VIG and flow p95 several times p50, all positive. For scale only: an earlier OpenDrawer MoE (300 demos) gave VIG p50 ≈ 0.03, p95 ≈ 0.12, flow p50 ≈ 0.04, p95 ≈ 0.10 |
| round `moe` DAgger | right after | `dagger/moe/<scorer>/settings.json` on the machine that took the collection; `status` progress | take/return thresholds equal the p95/p50 above; each collection uses its machine's whole CPU budget; 3–5 min per attempt per worker; successes rising steadily |
| DAgger metrics | when a round's collections finish | `results/dagger_<round>_<scorer>.json`; the DAgger tables and figures in `results/REPORT.md` | `collection.success_rate`, expert share and takeovers: the experiment's new numbers, no reference value |
| round `moe` retraining | after its collections | `policies/retrain/base_plus_moe_{vig,flow}/seed*/` | ~320k updates each (DAgger episodes are about as long as demos) |
| round `per_resolution` | from the start, alongside round `moe` | `policies/drivers/per_resolution/res{2..64}/`, then `thresholds.json` | 6 trainings of ~160k updates; VIG thresholds positive with p95 > p50 |
| evaluations | as each policy finishes | `eval/<policy>/level<f>/*.jsonl` | 200 episodes per policy and level, 6,200 in total. No reference for this setup; for scale, an earlier MoE trained on 300 planner demos succeeded in 36 / 40 / 54 / 46 / 36 / 16 / 2 % of 50 scenes from full resolution to 1/64 |
| summary | at the end | `results/eval_summary.json`, `results/REPORT.md` | mean ± std over 3 seeds per training set |

The runtimes are for a shared 2-GPU, 64-core machine. A training ran at 4.5–13 updates per second there.

## 6. When something goes wrong

| symptom | likely cause | what to do |
|---|---|---|
| `status` lists a FAILED job | see its log (path in `status`) | fix the cause if it is in the "you may" list, then `vig-dagger run` again; otherwise ask |
| a job retries again and again | the same error each time | read the log; don't wait for the retries to run out |
| a simulator job's log ends without a traceback and the job failed | Isaac Sim crashed (`terminate called …`, a crash dump) | rerun once; if it repeats, ask |
| `CUDA out of memory` in a training or worker log | other users filled the GPU | nothing; the retry runs when memory frees up. If it keeps happening, ask whether to raise `resources.train_gpu_memory_gb` |
| `obs ee_pose is the 'gripper' frame in this process` | a worker ran without `PYTHONHASHSEED=0` | only happens when started by hand; use the queue |
| `download_demos` failed (connection reset, timeout) | the network | rerun; it continues where it stopped. If it keeps failing, ask |
| W&B error when a training starts ("could not verify", 401/403) | `wandb.enabled` was set to true | set it back to false (W&B is not used); then rerun |
| `No space left on device` in a Python traceback (not the `change watch` lines) | disk full | stop the queue (`vig-dagger stop`) and ask; never delete experiment outputs to make room |
| queue NOT RUNNING, jobs remain | the queue process died (SSH session, reboot) | start it again (step 4) |
| machine rebooted | | start it again; trainings resume from their last checkpoint (at most 5000 updates lost) |
| "was collected with different settings" in a `dagger_prepare_*` log | the config changed after a collection started | restore the original config; ask if unsure |
| collection progress stalls (successes not rising for hours while its workers run) | something is wrong in the rollouts | read the workers' logs (`logs/dagger_<round>_<scorer>_*.log`) and ask |
| `status`: a task `FAILED (on <machine>)` | one of its jobs failed there | on that machine: read the job's log, fix the cause if it is in the "you may" list, `vig-dagger run` again, then `vig-dagger release TASK` so a machine can take it again. Otherwise ask |
| `status`: a machine `SILENT` | it is down, lost its network, or its queue stopped | if it is your machine, start the queue again (step 4). Otherwise ask the humans; once they confirm it is down, `vig-dagger release` each task `status` lists as running on it |
| a `publish/…` or `fetch/…` job failed | Hugging Face or the network | it retries by itself; if it keeps failing, check `hf auth whoami` and the network, then ask |
| "Hugging Face: not logged in" when the queue starts | no token on this machine | ask the humans for the token (step 0) |

## 7. When it finishes

Run `vig-dagger report -c configs/run.yaml` once more (on any machine: it collects every machine's results from the
shared repo) and send the humans `<work_dir>/results/REPORT.md` and the
`results/figures/` folder (or the whole `results/` folder, a few MB). In your message, summarize:

1. **Evaluation** (the report's first table): for each training set (`base`, `base_plus_planner`,
   `base_plus_moe_vig`, `base_plus_moe_flow`, `base_plus_per_resolution_vig`, `base_plus_per_resolution_flow`):
   - mean ± std of the success rate over the 3 seeds;
   - against `base`: scenes only this set solved / only `base` solved, and the sign-test p-value.
2. **The driving policies by resolution:** the MoE and round 2's policy at each level.
3. **For each round and scorer** (the DAgger table):
   - collection success rate;
   - pooled expert share;
   - median expert share and median takeovers per episode;
   - the main failure reasons (`expert_failed` with its planner reasons, `time_limit_*`).
4. **Anything unusual:** failures you fixed, restarts, jobs that needed several attempts.

## Several machines

Every machine of the experiment runs the same thing: the same branch, the same `configs/run.yaml` settings (only
`work_dir` and `resources` may differ), the same `vig-dagger run`. One machine works; so do two or six, and a machine
can join at any time. The machines share the run through the public Hugging Face repo in `sharing.repo`
(`chengenwu/vig-dagger-opendrawer-v1`):

- The work comes in **tasks**: each training (25), each round's thresholds (2), each DAgger collection (4: two rounds ×
  two scorers), each evaluation of a policy at a level (31), and the final summary.
- A machine with room **claims** the most urgent task whose inputs are all in the repo (a claim can't be taken twice:
  it is a commit that fails if another machine committed first), downloads the inputs (`fetch/…` jobs), runs the
  task, and **uploads** its outputs (`publish/…` jobs): checkpoints (weights only), DAgger training sets and metrics,
  evaluation episodes. Each machine downloads and converts the demos itself when one of its tasks needs them.
- **What waits:** a round's thresholds need all of its driving policies, a DAgger collection needs its thresholds,
  a retraining needs its collection, an evaluation needs its policy. Such a task runs on one machine while the others
  take other tasks or wait for the upload. Both rounds start at once.
- Each machine sends a **heartbeat** every 30 minutes; `status` (on any machine) shows every task and every machine.
- A machine's queue ends by itself when every task of the experiment is done; `vig-dagger report` (on any machine)
  then writes the full report from the repo.

**A machine that goes `SILENT`** (three heartbeats missed) still holds its tasks, and nobody else takes them. Ask the
humans whether it is down. Once they confirm, run `vig-dagger release <task>` for each task `status` lists as running
on it; any machine then takes the task and starts it from the beginning (the checkpoints for resuming a training
stay on the machine that held it). If the machine comes back, its queue sees that the task is no longer its own
and leaves it.

**A task that `FAILED`** stays with its machine until it is released: fix the cause there, restart that machine's
queue, then `vig-dagger release <task>`.

The time depends on how many machines there are, because a few steps can't run in parallel (each round's chain:
driving policies → thresholds → DAgger collection → retraining → evaluation). With 2 GPUs per machine: about a week
on one machine, 3 days on two, 2 days on four, 1–1.5 days on five or six.

## Reference

**Job names** (what `status`, `run.log` and `<work_dir>/logs/` use):

| job | what it does |
|---|---|
| `download_demos` | download the base and extra planner demos from Hugging Face |
| `convert_planner`, `convert_extra` | convert them to training datasets |
| `planner_baseline` | build `base_plus_planner` |
| `train/drivers/moe` | the 7-level MoE that drives round `moe` |
| `train/drivers/per_resolution/res<k>` | round 2's degraded-level policies |
| `score_<round>_<chunk>`, `thresholds_<round>` | demo-frame scores and their percentiles |
| `dagger_prepare_<round>_<scorer>` | resolve the thresholds, write `settings.json` |
| `dagger_<round>_<scorer>_<worker>` | DAgger workers, sharing one collection |
| `dagger_dataset_<round>_<scorer>` | the first 100 successes → training set `dagger_<round>_<scorer>` |
| `combine_<round>_<scorer>` | the retraining set: base + that DAgger set |
| `analyze_<round>_<scorer>` | DAgger metrics of one collection |
| `publish/<task>`, `fetch/<task>` | upload a task's outputs to the shared repo / download them |
| `train/retrain/<set>/seed<s>` | the evaluated policies |
| `eval/<policy>/level<f>/<first seed>` | 25 evaluation episodes of one policy at one level |
| `summarize` | the final tables |

**Useful commands:**

- `vig-dagger run -c … --dry-run`: every job with its priority and dependencies (`done` marks finished ones).
- `vig-dagger run -c … --only REGEX`: run a subset and what it needs.
- `vig-dagger stop -c …`: stop everything. With `--queue-only`, only the queue stops.
- `vig-dagger show-config -c …`: the resolved configuration.

**Where things are:** see "Output directory" in the README. A training's progress is in
`policies/.../metrics.jsonl` (`train/step`, `train/loss`, `val/loss`), summarized by `vig-dagger report`.
