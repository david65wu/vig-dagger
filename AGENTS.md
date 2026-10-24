# Running the experiment: instructions for an agent

These instructions are for an agent (or a person) who runs `configs/plugcharger.yaml` from start to finish without
supervision. The README explains what the pipeline does; this file says how to operate it: what to run, what a
healthy run looks like, when to stop and ask, and what never to change.

## The goal and what "done" means

- **The run:** two rounds of DAgger (round `moe`, then round `per_resolution`). Each collects 500 successful
  episodes with VIG and 500 with flow loss. 18 dense full-resolution policies are trained and evaluated
  (6 training sets × 3 seeds), and so are the driving policies: the MoE at each of its 7 levels, and round 2's
  `res2` … `res64` at their own levels. Every evaluation is 200 rollouts on the same scenes.
- **Done when** (on one machine; with two, see [Two machines](#two-machines-one-round-on-each)):
  - `vig-dagger status` shows every job `done`;
  - `<work_dir>/results/eval_summary.json` exists;
  - `<work_dir>/results/REPORT.md` has the evaluation tables filled in.

Every result is written to files in `<work_dir>/results`; W&B is not used (`wandb.enabled: false`).
- **Then report** the numbers listed in [When it finishes](#7-when-it-finishes).

Expect it to take several days: about a week on 2 GPUs, less with more GPUs. Most of the time goes into 24
trainings of 4–17 h each and the four DAgger collections.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `rounds`,
  `drivers`, `scorers`, `thresholds`, `dagger`, `retrain`, `evaluation`. These define the experiment. A changed
  value either silently mixes two experiments, or the pipeline refuses to continue.
- **Delete or move anything in `work_dir`**, and above all `dagger/`, `policies/`, `data/`. Collections take days;
  partial outputs are what lets a run resume.
- **Upload** to Hugging Face, **push** to git, or turn on W&B.
- **Run two queues on the same `work_dir`**, or start workers by hand outside the queue.

**You may:**

- set `work_dir` before the first start;
- set `split` when the humans tell you this machine runs only one of the two rounds (see
  [Two machines](#two-machines-one-round-on-each));
- change `resources.*` settings, e.g. fewer `sim_slots` when the machine is shared. To apply them without
  interrupting jobs: `vig-dagger stop --queue-only`, edit the config, then `vig-dagger run` again;
- stop and restart the queue (`vig-dagger stop`, `vig-dagger run`);
- rerun `vig-dagger analyze --round <r>` and `vig-dagger summarize` at any time;
- read any file or log;
- regenerate the report with `vig-dagger report` at any time.

**Stop and ask the humans when:**

- `vig-dagger check` reports a failure you cannot fix by installing something: no Vulkan, not enough disk, no access
  to Hugging Face;
- a job is `FAILED` again after you fixed what its log showed and restarted once, or you can't tell why it failed;
- a DAgger collection is heading for its attempt limit (10,000 attempts per collection). Concretely: after 300
  attempts it has fewer than 15 successes (a success rate under 5 %);
- free disk in `work_dir` drops below 30 GB;
- a training's loss is NaN, or its validation loss rises steadily over the second half of training;
- a milestone value is far outside the expected range below.

## 0. Before you start, get from the human

Ask once, before installing anything, for whatever the request didn't say:

- **one machine or two:** whether this machine runs both rounds, or only round 1 or only round 2 (`split`, see
  [Two machines](#two-machines-one-round-on-each));
- **the work directory:** a path on a disk with at least 150 GB free; everything the run writes goes there;
- **the GPUs you may use:** all of them, or a list of indices (`resources.gpus`);
- **whether other people use the machine:** if so, ask whether to leave CPU cores free (`resources.sim_slots`, by
  default half the cores' worth of simulator jobs);
- **how they want to hear from you:** e.g. a status message after the launch and after each milestone, and the final
  report.

Also say that the run takes several days and keeps going after your session ends, and agree on how you'll check on
it: a daily new session, or scheduled check-ins if your tool supports them.

## 1. Setup

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
uv sync
source .venv/bin/activate
```

Hugging Face downloads need no login (both repositories are public). `uv` and `ffmpeg` are the only programs to
install. If either is missing and installing it needs administrator rights, ask.

## 2. Configure

Copy `configs/plugcharger.yaml` (e.g. to `configs/run.yaml`). If this machine runs only one round, set `split` (see
[Two machines](#two-machines-one-round-on-each)). Set `work_dir` to a directory with at least 150 GB
free. Leave everything else as it is.

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable. It checks:
- the GPUs, and that torch sees CUDA;
- rendering (Vulkan);
- `ffmpeg`;
- free disk;
- that the Hugging Face files exist;
- the W&B setting (it should say `W&B disabled`).

It prints the CPU and GPU budget the queue will use.

On a new machine, also run the smoke test (a few minutes, writes only to the given directory):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

It must end with `[queue] finished: 76 of 76 jobs done`. Delete `/tmp/vig-dagger-smoke` afterwards.

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

### Milestones and expected values

| milestone | when | what to check | expected |
|---|---|---|---|
| demos downloaded | first ~30 min | `<work_dir>/data/raw/planner/` | 20 `part_*.h5` (26 GB); `planner_extra/`: 10 (13 GB) |
| datasets converted | ~1 h | `num_demos` in each `data/datasets/<name>/dataset_manifest.json` | planner 1000, base 1000, planner_extra 500, base_plus_planner 1500 |
| round `moe` driver | ~1 h | `policies/drivers/moe/thresholds.json` (downloaded with the MoE) | VIG p50 0.0112, p95 0.0327; flow p50 0.0127, p95 0.0341 |
| round `moe` DAgger | starts ~1 h | `dagger/moe/<scorer>/settings.json`; `status` progress | take/return thresholds equal the values above; successes rising steadily, both scorers at the same time |
| baseline trainings | start ~1 h, run alongside DAgger | `policies/retrain/{base,base_plus_planner}/seed*/metrics.jsonl`; the Trainings table and `figures/training_*.png` of `vig-dagger report` | 188,580 updates (1000 demos) or ~283k (1500); validation loss falling to ~0.02 |
| DAgger metrics | when a round's collections finish | `results/dagger_<round>_<scorer>.json`; the DAgger tables and figures in `results/REPORT.md` | `collection.success_rate`, expert share and takeovers: the experiment's new numbers, no reference value |
| round `moe` retraining | after its collections | `policies/retrain/base_plus_moe_{vig,flow}/seed*/` | about 280–330k updates (DAgger episodes run up to 400 steps) |
| round `per_resolution` | after every round-`moe` retraining | `policies/drivers/per_resolution/res{2..64}/`, then `thresholds.json` | 6 trainings of 188,580 updates; VIG thresholds positive with p95 > p50 |
| evaluations | as each policy finishes | `eval/<policy>/level<f>/*.jsonl` | 200 episodes per policy and level, 6,200 in total. MoE driver, from full resolution to 1/64, on the earlier scene set: 34.0 / 34.5 / 46.5 / 42.0 / 26.5 / 16.0 / 13.0 %; expect values in that range. No reference for the dense Transformer policies; earlier full-resolution U-Net policies on the same demos reached 31–37 % |
| summary | at the end | `results/eval_summary.json`, `results/REPORT.md` | mean ± std over 3 seeds per training set |

The runtimes are for a shared 2-GPU, 64-core machine. A training ran at about 4.5 updates per second there, so a
1000-demo training takes about 12 h.

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
   - median expert share and median takeovers per episode.
4. **Anything unusual:** failures you fixed, restarts, jobs that needed several attempts.

## Two machines: one round on each

The humans may run the two rounds on two machines. Each machine then has its own clone, environments and work
directory, and its `configs/run.yaml` sets `split` (step 2), before the first `vig-dagger run`:

| machine | `split` | what it runs |
|---|---|---|
| round 1 | `round1` | the demos; the shared baselines (`base` and `base_plus_planner`, 3 seeds each); the MoE and round `moe`; their evaluations; the summary |
| round 2 | `round2` | the demos; `base` seed 0 only (round 2's full-resolution driver); `res2` … `res64`; round `per_resolution`; its evaluations |

The round-2 machine starts at once, without waiting for round 1, and trains no other shared baseline: `base`
seeds 1 and 2 and `base_plus_planner` are trained only on the round-1 machine. Check before launching:
`vig-dagger run -c configs/run.yaml --dry-run | wc -l` lists 212 jobs with `split: round1` and 153 with `split: round2`
(no `train/retrain/base_plus_planner` job on the round-2 machine), and `vig-dagger status` says "this machine runs only
round …".

**Bringing the results together.** When the round-2 machine is done (`status` shows every job `done`):

1. Copy its work directory to the round-1 machine, without the large parts, e.g. on the round-1 machine:

   ```bash
   rsync -a --exclude '*.pt' --exclude 'data/' --exclude 'hub/' --exclude 'dagger/' --exclude 'tmp/' \
     <round-2 machine>:<its work_dir>/ <a directory on this machine>/
   ```

   (`config.json`, `eval/`, `results/` and `policies/` are what is needed.)
2. On the round-1 machine: `vig-dagger merge -c configs/run.yaml <that directory>`. It refuses unless both machines
   ran the same experiment, copies round 2's evaluations, DAgger metrics and training logs into this work directory,
   and refreshes `results/eval_summary.json` and `results/REPORT.md`. It can run before or after the round-1 machine
   finishes (its own `summarize` job includes whatever was merged); run it again if you copy again.

The final report (step 7) then comes from the round-1 machine. The round-2 machine's agent reports its round's
DAgger table and its evaluations (`vig-dagger summarize -c configs/run.yaml`, then `results/REPORT.md` there), and
says when the copy can be made.

**A machine that already runs with `split: none`** (both rounds) and should hand round 2 to another machine:
`vig-dagger stop -c configs/run.yaml --queue-only`, set `split: round1` in `configs/run.yaml`, then start the queue
again (step 4). Running jobs continue, and none of round 2's jobs has started yet (they wait for round 1's
retraining), so nothing is lost. Never switch a machine that has already started round-1 work to `split: round2`
without asking the humans: its round-1 work would be abandoned.

## Reference

**Job names** (what `status`, `run.log` and `<work_dir>/logs/` use):

| job | what it does |
|---|---|
| `download_demos` | download the raw planner demos |
| `convert_planner`, `convert_extra` | convert them to training datasets |
| `planner_baseline` | build `base_plus_planner` |
| `download_driver` | download the MoE driver |
| `train/drivers/per_resolution/res<k>` | round 2's degraded-level policies |
| `score_<round>_<chunk>`, `thresholds_<round>` | demo-frame scores and their percentiles (round 2; round 1's come with the MoE) |
| `dagger_prepare_<round>_<scorer>` | resolve the thresholds, write `settings.json` |
| `dagger_<round>_<scorer>_<worker>` | DAgger workers, sharing one collection |
| `dagger_dataset_<round>_<scorer>` | the first 500 successes → training set |
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
