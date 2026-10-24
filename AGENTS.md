# Running the experiment: instructions for an agent

These instructions are for an agent (or a person) who runs `configs/opendrawer.yaml` (branch `opendrawer`) from
start to finish without supervision. The README explains what the pipeline does; this file says how to operate it:
what to run, what a healthy run looks like, when to stop and ask, and what never to change.

## The goal and what "done" means

- **The run:** OpenDrawer in Isaac Lab. The pipeline collects 100 planner demos (plus 100 for a baseline) and
  trains every policy itself, 150 epochs each. Then two rounds of DAgger (round `moe`, then round
  `per_resolution`), each collecting 100 successful episodes with VIG and 100 with flow loss. 18 dense
  full-resolution policies are trained and evaluated (6 training sets × 3 seeds), and so are the driving policies:
  the MoE at each of its 7 levels, and round 2's `res2` … `res64` at their own levels. Every evaluation is 200
  rollouts on the same scenes.
- **Done when:**
  - `vig-dagger status` shows every job `done`;
  - `<work_dir>/results/eval_summary.json` exists;
  - `<work_dir>/results/REPORT.md` has the evaluation tables filled in.

Every result is written to files in `<work_dir>/results`; W&B is not used (`wandb.enabled: false`).
- **Then report** the numbers listed in [When it finishes](#7-when-it-finishes).

Expect it to take about a week on 2 GPUs, less with more GPUs. Most of the time goes into 25 trainings of 3–19 h
each.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `rounds`,
  `drivers`, `scorers`, `thresholds`, `dagger`, `retrain`, `evaluation`. These define the experiment. A changed
  value either silently mixes two experiments, or the pipeline refuses to continue.
- **Delete or move anything in `work_dir`**, and above all `data/`, `dagger/`, `policies/`. Collections and
  trainings take days; partial outputs are what lets a run resume.
- **Upload** to Hugging Face, **push** to git, or turn on W&B.
- **Run two queues on the same `work_dir`**, or start workers by hand outside the queue.
- **Edit the code**, including the planner and the environment.

**You may:**

- set `work_dir` before the first start;
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
- a demo collection job fails with "only N of 25 demos" (the planner failed on most scenes);
- a DAgger collection is heading for its attempt limit (2,000 attempts per collection). Concretely: after 200
  attempts it has fewer than 10 successes (a success rate under 5 %);
- free disk in `work_dir` drops below 30 GB;
- a training's loss is NaN, or its validation loss rises steadily over the second half of training;
- a milestone value is far outside the expected range below.

## 0. Before you start, get from the human

Ask once, before installing anything, for whatever the request didn't say:

- **NVIDIA's license:** the install script downloads Isaac Sim, and every simulator job accepts the
  [NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
  non-interactively (`OMNI_KIT_ACCEPT_EULA=YES`). The human must agree to it; don't accept it on their behalf.
- **the work directory:** a path on a disk with at least 120 GB free; everything the run writes goes there. The
  repository itself needs about 30 GB more for its two Python environments;
- **the GPUs you may use:** all of them, or a list of indices (`resources.gpus`). They must be RTX-class GPUs
  (with RT cores; an A100 or H100 cannot run Isaac Sim's renderer);
- **whether other people use the machine:** if so, ask whether to leave CPU cores free (`resources.sim_slots`, by
  default a third of the cores' worth of simulator jobs);
- **how they want to hear from you:** e.g. a status message after the launch and after each milestone, and the final
  report.

Also say that the run takes about a week and keeps going after your session ends, and agree on how you'll check on
it: a daily new session, or scheduled check-ins if your tool supports them.

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

Copy `configs/opendrawer.yaml` (e.g. to `configs/run.yaml`). Set `work_dir` to a directory with at least 120 GB
free. If the human named GPUs or a CPU budget, set `resources.gpus` / `resources.sim_slots`. Leave everything else as
it is.

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable. It checks:
- free disk, `ffmpeg`, the GPUs, and that torch sees CUDA;
- the Isaac environment: it starts Isaac Sim once, builds the task, renders the two policy cameras, and checks that
  the policy's state input is the TCP frame. The first start on a machine compiles shaders and can take up to ~15
  minutes; the full output goes to `<work_dir>/logs/check_isaac.log`;
- that the DINOv2 weights are reachable on Hugging Face;
- the W&B setting (it should say `W&B disabled`).

It prints the CPU and GPU budget the queue will use.

On a new machine, also run the smoke test (about 15 minutes, writes only to the given directory):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

It must end with `[queue] finished: 76 of 76 jobs done`. Its success rates are 0 % (its policies train for a few
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
- the milestones.

If the queue is NOT RUNNING and jobs remain, start it again with the command from step 4. It takes over jobs that
are still running.

**Reading a simulator job's log** (`collect_*`, `dagger_*`, `eval/*`): Isaac Sim writes hundreds of harmless
lines first (`Failed to create change watch … errno=28/No space left on device`, GLFW, display, DLSS, IOMMU
warnings). Skip them: `grep -v "change watch" <log> | grep -E "^\[seed|^\[worker|Traceback|Error:"` shows the
episodes and any real error.

### Milestones and expected values

| milestone | when | what to check | expected |
|---|---|---|---|
| demos collected | first ~30–60 min | `<work_dir>/data/raw/planner/part_00{0..3}.done.json`, `planner_extra/` the same | 4 + 4 parts of 25 demos; `attempts` in each `done.json` close to 25 (the planner succeeds on most scenes); ~75 MB per demo |
| datasets converted | right after | `num_demos` in each `data/datasets/<name>/dataset_manifest.json` | planner 100, base 100, planner_extra 100, base_plus_planner 200 |
| round `moe` driver | 3–9 h | `policies/drivers/moe/metrics.jsonl`, then `latest.pt`; the Trainings table of `vig-dagger report` | ~150k updates; validation loss falling |
| baseline trainings | start with the MoE, on the other GPUs | `policies/retrain/{base,base_plus_planner}/seed*/` | ~150k updates (100 demos) or ~300k (200 demos) |
| round `moe` thresholds | ~30 min after the MoE | `policies/drivers/moe/thresholds.json` | VIG and flow p95 several times p50, all positive. For scale only: an earlier OpenDrawer MoE (300 demos) gave VIG p50 ≈ 0.03, p95 ≈ 0.12, flow p50 ≈ 0.04, p95 ≈ 0.10 |
| round `moe` DAgger | right after | `dagger/moe/<scorer>/settings.json`; `status` progress | take/return thresholds equal the p95/p50 above; both scorers collecting at the same time; 3–5 min per attempt per worker; successes rising steadily |
| DAgger metrics | when a round's collections finish | `results/dagger_<round>_<scorer>.json`; the DAgger tables and figures in `results/REPORT.md` | `collection.success_rate`, expert share and takeovers: the experiment's new numbers, no reference value |
| round `moe` retraining | after its collections | `policies/retrain/base_plus_moe_{vig,flow}/seed*/` | ~300k updates each (DAgger episodes are about as long as demos) |
| round `per_resolution` | after every round-`moe` retraining | `policies/drivers/per_resolution/res{2..64}/`, then `thresholds.json` | 6 trainings of ~150k updates; VIG thresholds positive with p95 > p50 |
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
| `only N of 25 demos after 75 attempts` in a `collect_*` log | the planner fails on most scenes | ask (something is wrong with the simulator or the installation) |
| W&B error when a training starts ("could not verify", 401/403) | `wandb.enabled` was set to true | set it back to false (W&B is not used); then rerun |
| `No space left on device` in a Python traceback (not the `change watch` lines) | disk full | stop the queue (`vig-dagger stop`) and ask; never delete experiment outputs to make room |
| queue NOT RUNNING, jobs remain | the queue process died (SSH session, reboot) | start it again (step 4) |
| machine rebooted | | start it again; trainings resume from their last checkpoint (at most 5000 updates lost) |
| "was collected with different settings" in a `dagger_prepare_*` log | the config changed after a collection started | restore the original config; ask if unsure |
| collection progress stalls (successes not rising for hours while its workers run) | something is wrong in the rollouts | read the workers' logs (`logs/dagger_<round>_<scorer>_*.log`) and ask |
| queue ends with "failed or blocked" | a failure stopped the jobs that depend on it | fix the failure, then `vig-dagger run` again; finished work is kept |

## 7. When it finishes

Run `vig-dagger report -c configs/run.yaml` once more and send the humans `<work_dir>/results/REPORT.md` and the
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

## Reference

**Job names** (what `status`, `run.log` and `<work_dir>/logs/` use):

| job | what it does |
|---|---|
| `collect_planner_<k>`, `collect_extra_<k>` | collect 25 planner demos in Isaac Sim (base / extra) |
| `convert_planner`, `convert_extra` | convert them to training datasets |
| `planner_baseline` | build `base_plus_planner` |
| `train/drivers/moe` | the 7-level MoE that drives round `moe` |
| `train/drivers/per_resolution/res<k>` | round 2's degraded-level policies |
| `score_<round>_<chunk>`, `thresholds_<round>` | demo-frame scores and their percentiles |
| `dagger_prepare_<round>_<scorer>` | resolve the thresholds, write `settings.json` |
| `dagger_<round>_<scorer>_<worker>` | DAgger workers, sharing one collection |
| `dagger_dataset_<round>_<scorer>` | the first 100 successes → training set |
| `analyze_<round>` | DAgger metrics |
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
