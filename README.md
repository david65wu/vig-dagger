# VIG DAgger

DAgger experiments that compare two query scores deciding when the expert takes over from the policy:

- **VIG** (visual information gain): how much the policy's plan changes when its camera images are degraded.
- **Flow loss**: the policy's flow-matching loss on its own sampled action chunk.

The first task is **PlugCharger** in ManiSkill: pick up a charger and insert its two pins into a socket that moves
from episode to episode. A motion planner provides the demos and plays the expert during DAgger. The policies are
flow-matching Transformer policies trained with the `robopolicy` library, which is included in this repository.

The experiment has two **rounds**, which differ in the policy that drives the DAgger rollouts:

| round | drives DAgger, gives the flow loss | VIG compares |
|---|---|---|
| `moe` | a 7-level resolution mixture-of-experts (MoE) policy, at full resolution | the MoE at full resolution vs the MoE at each degraded level (1/2 … 1/64) |
| `per_resolution` | a dense full-resolution policy | that policy vs 6 dense policies trained on images degraded by 2 … 64 |

Each round has the same four parts:

1. **Demos.** 1000 planner demos, downloaded from Hugging Face; shared by both rounds.
2. **Driver and thresholds.** The round's driving policy or policies, and VIG and flow-loss percentiles on demo
   frames.
3. **DAgger.** 500 successful episodes with VIG as the query score and 500 with flow loss. A take-over threshold
   and a separate return threshold decide who drives. The collections are analyzed and logged to W&B.
4. **Retraining.** Dense full-resolution policies are trained on the base demos plus the DAgger demos, three seeds
   each, and evaluated on the same 200 scenes.

Two baselines are shared by both rounds and trained once: the base demos alone, and the base demos plus 500 more
planner demos.

`vig-dagger run` turns the whole experiment into one queue of jobs. A job starts when its inputs exist and the
machine has room for it (CPU, RAM, GPU memory), so independent work runs in parallel. Rerunning the same command
after an interruption skips finished jobs and picks up running ones.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The task](#the-task)
- [The job queue](#the-job-queue)
- [Part 1: demos](#part-1-demos)
- [Part 2: driving policies and thresholds](#part-2-driving-policies-and-thresholds)
- [Part 3: DAgger](#part-3-dagger)
- [Part 4: retraining and evaluation](#part-4-retraining-and-evaluation)
- [Weights & Biases](#weights--biases)
- [Hugging Face](#hugging-face)
- [Output directory](#output-directory)
- [Configuration reference](#configuration-reference)
- [Cost estimates](#cost-estimates)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

---

## Installation

Requirements:

- Linux with an NVIDIA GPU. The driver must support CUDA 12.8, and Vulkan must work, because SAPIEN, ManiSkill's
  renderer, uses Vulkan even on a headless machine. Tested on RTX PRO 6000 Blackwell.
- `ffmpeg` on the `PATH`. Training data is stored as H.264 video.
- [`uv`](https://docs.astral.sh/uv/) (Python package manager); it installs Python 3.11 itself.
- About 250 GB of disk for the work directory; see [cost estimates](#cost-estimates).

```bash
git clone <this repository> vig-dagger && cd vig-dagger
uv sync                       # creates .venv with the pinned packages (uv.lock)
source .venv/bin/activate     # or prefix commands with `uv run`
wandb login                   # the runs go to the W&B team vig-dagger-exp
vig-dagger --help
```

The first training downloads the DINOv2-small weights from Hugging Face (`facebook/dinov2-small`, about 90 MB).

**Check the installation** with a short run of every job of both rounds on tiny numbers (4 demos, policies trained
for a dozen updates; the success rates it prints are meaningless):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

The library's own tests: `uv sync --extra dev && pytest tests -m "not cuda"`.

## Quick start

1. In `configs/plugcharger.yaml`, set `work_dir` (the output directory). Everything else is set for the
   experiment; see the [configuration reference](#configuration-reference).
2. Start the queue, detached so it survives a closed SSH session:

   ```bash
   setsid nohup vig-dagger run -c configs/plugcharger.yaml > plugcharger.log 2>&1 < /dev/null &
   ```

3. Watch it:

   ```bash
   vig-dagger status -c configs/plugcharger.yaml   # done / running / ready / waiting per step
   tail -f plugcharger.log                         # the queue's own log: every job started, finished, retried
   ```

   Each job's output is in `<work_dir>/logs/<job>.log`.
4. After a crash, a reboot or a fix, run the same command again.

Useful options:

- `--set key=value` overrides any config value, e.g. `--set dagger.num_demos=200`.
- `--only REGEX` runs only matching jobs and what they need, e.g. `--only 'eval/'` or `--only '^dagger_moe'`.
- `--dry-run` lists the jobs with their dependencies.
- `vig-dagger show-config -c …` prints the resolved configuration.

## The task

Environment `PlugCharger-Hole4mm-SideCam-RandSocket-v1` (`src/vig_dagger/envs.py`). This is ManiSkill's
`PlugCharger-v1` with four changes:

- 4 mm clearance per side in the socket holes;
- the socket moves every episode: ±3 cm sideways, ±2 cm in height, ±5° in yaw;
- a charcoal-gray socket body;
- two extra fixed cameras.

| | |
|---|---|
| Robot / control | Franka Panda, absolute joint-position targets (`pd_joint_pos`, 7 joints + gripper) |
| Policy inputs | joint state (7 arm + 2 finger positions); RGB 256×256 from `hand_camera` (wrist), `third_person_camera`, `side_camera` (26 cm from the socket face) |
| Policy outputs | 16-step action chunks; 8 steps are executed before the next prediction |
| Success | plug within 5 mm and 0.2 rad of the inserted pose |
| Step limit | 300 for planner demos, 400 for DAgger and evaluation |
| Physics | CPU (one environment per process); rendering on the GPU |

**Scene seeds.** A seed fixes the charger and socket placement, so every policy is evaluated on the same scenes.
The socket position comes from torch's random generator, so a seed gives a different scene under a different torch
version. Inside this repository torch is pinned (`uv.lock`), so seeds are consistent across machines that use it.

The seed ranges never overlap:

| Use | Seeds |
|---|---|
| Base demos | 0–40,049 and 70,000–210,052 |
| Extra planner demos | 320,000–410,050 |
| DAgger | from 1,000,000 |
| Evaluation | 60,000–60,199 |

## The job queue

`vig-dagger run` builds every job of the experiment (`src/vig_dagger/plan.py`) and hands them to a queue
(`src/vig_dagger/jobs.py`). Each job is one process with its own log.

**When a job starts.** Once the jobs it depends on are done, and the machine has room:

- **CPU:** simulator jobs (collection, DAgger, evaluation, threshold scoring) and conversions use a CPU budget,
  `resources.sim_slots` (default: one slot per 2 cores; a simulator job uses about one core). A training counts as `resources.train_slots` (its data
  loading).
- **Load:** nothing new starts while the 1-minute load average is above `resources.max_load` (default: the number
  of cores), so other users' work is respected.
- **RAM:** the available memory must cover the job's estimate.
- **GPU:**
  - a simulator job goes to the GPU with the most free memory, if it has at least `resources.sim_gpu_memory_gb`;
  - a training needs `resources.train_gpu_memory_gb` free on a GPU that runs none of our other trainings.

Jobs started in the last 3 minutes count with their estimates.

**Order.** Lower priority numbers go first: part 1, then round 1's driver and DAgger, round 1's retraining, the
shared baselines, round 2, and the evaluations.

**What runs in parallel**, if the machine has room:

- the shared baselines (`base`, `base_plus_planner`, 3 seeds each) train on the GPUs while round 1 collects DAgger
  episodes on the CPUs;
- every evaluation runs as soon as its policy is trained.

Round 2's driving policies start only after all of round 1's retraining.

**Failures and restarts:**

- A job that fails is retried a few times (trainings up to `training.max_restarts` times). Every job continues
  where it stopped: trainings from their last checkpoint (every 5000 updates), DAgger workers from their last
  finished episode, evaluations from their last finished scene.
- A job that still fails stops only the jobs that depend on it. The rest of the queue keeps going.
- If the queue process dies, the jobs it started keep running. A restarted queue adopts them instead of starting
  them twice. Only one queue runs per work directory.

## Part 1: demos

`download_demos` fetches from `hub.dataset_repo`:

- `raw/planner`: 20 files of 50 motion-planner demos. These are the 1000 demos the MoE driver was trained on.
- `raw/planner_extra`: 10 files, 500 more demos from the same planner and environment, for the planner baseline.

**The planner** (`src/vig_dagger/expert/planner.py`) samples a strategy for each demo:

- grasp tilt 5–25° and offset ±8 mm;
- a lift waypoint;
- an off-axis approach that is corrected before a straight insertion;
- joint speeds;
- screw or RRT transit.

So no single joint path solves the task. Only successful demos that finish within 300 steps are kept.

**Converted datasets.** `convert_planner` turns the raw demos into the library's training format (H.264 videos plus
HDF5 with actions and joint state):

- `data/datasets/planner`: every demo;
- `data/datasets/base`: the first `demos.num_demos` (1000), the original training set.

`convert_extra` and `planner_baseline` build `base_plus_planner` = base + the first 500 extra demos.

With `demos.source: collect`, the pipeline collects the demos itself instead (seeds from 2,000,000 and 3,000,000).

## Part 2: driving policies and thresholds

**Policy recipe** (`src/vig_dagger/configs/policy.json`; every training in the task uses it):

- frozen DINOv2-small encoder per camera, cross-attention pooling to one token per camera;
- an 8-layer Transformer action head (width 256) with adaptive LayerNorm on the flow time and the current joint
  state;
- linear-path flow matching with 10 Euler sampling steps;
- batch 64, AdamW, lr 1e-4 with cosine decay;
- random 224 px crops during training.

A policy degrades its own input images by area-downsampling and bilinear upsampling by an integer factor.

**Training length.** Every training runs `training.epochs` = 60 passes over its training demos:

max_updates = 60 × ceil(training samples / 64)

A training sample is any step with a full 16-step future. 2 % of the demos are held out for the validation loss.

**Round `moe`.** `download_driver` fetches the 7-level MoE from `hub.model_repo`. It was trained 60 epochs on the
1000 base demos:

- every training sample is degraded to a random level (factor 1, 2, 4, …, 64);
- every head layer's feed-forward block has 8 experts (top-2) with one router per level.

Its thresholds come with it. With `drivers.moe.source: train`, the pipeline trains it instead.

**Round `per_resolution`.** The full-resolution policy is the `base` retraining of seed 0: the same data, recipe
and seed, so it isn't trained twice. `train/drivers/per_resolution/res<k>` trains a dense policy on images degraded
by *k*, for *k* = 2, 4, …, 64.

**Thresholds** (`score_<round>_*`, `thresholds_<round>`):

- **Frames scored:** every 4th frame of the first 50 base demos.
- **Scored like DAgger:** VIG with 32 samples per level; flow loss with 512 draws.
- **Output:** the percentiles (p50 … p99) go to `policies/drivers/<round>/thresholds.json`.

For the downloaded MoE, the thresholds reproduce the values measured when it was trained to within 2 %.

### The two scores

Both are computed on the full-resolution policy's normalized actions (`src/vig_dagger/scoring.py`).

- **VIG.** Draw 32 chunks at full resolution and at every degraded level, all from the same 32 starting noises.
  For each degraded level, take the 1-D Wasserstein distance between the two sample sets: sort each of the
  16 × 8 chunk cells, compare, and average over the cells. VIG is the largest of these distances over the levels.
  In round `per_resolution`, each level's samples come from its own policy and are first mapped into the
  full-resolution policy's action normalization.
- **Flow loss.** Draw one chunk from the full-resolution policy and average the policy's flow-matching loss on
  that chunk over 512 random (noise, time) draws. It is noisy from frame to frame: two evaluations of the same frame
  differ by about 50 % on average.

## Part 3: DAgger

Each round collects with **both** scorers from the same driving policy, on the same scene sequence.

**Each episode** (`src/vig_dagger/workers/dagger.py`, gate in `src/vig_dagger/gate.py`):

1. The full-resolution driving policy drives. Every observation is scored.
2. **Take over:** the expert takes over after **2** consecutive scores above the **take threshold (p95** of the
   score on demo frames).
3. **Return:** the score keeps being computed after every expert step. Control returns to the policy after **16**
   consecutive scores below the **return threshold (p50)**, for the rest of the same episode. Without a return, the
   expert finishes the task.
4. **End:** the episode ends at success, at the 400-step limit (policy and expert alike), or when the planner
   fails.

**The expert** (`resume_and_solve`, `dagger.expert: demo`) moves like the planner demos: the strategy's sampled
speeds and the same stage sequence, with a fresh strategy at each takeover. It starts from the current state:

- **Plug not held:** reach, grasp and lift. It skips the reach detour if the gripper is already over the charger.
- **Taking over right after a grasp:** it first does the demo's lift.
- **Plug held but still far from the socket:** the demo's approach, alignment and insertion.
- **Plug held near the socket** (closer than the strategy's approach distance): it corrects in place instead of
  pulling back to the approach pose. It aligns onto the socket axis at the plug's current distance, and backs off
  only to 5 mm in front of the face when the pins are against the face but not in the holes. Then it inserts.

In tests from a held plug 30 mm in front of the socket and 8 mm off its axis, the in-place correction inserted 6/6
times in a median of 63 steps without backing off. Returning to the approach pose instead backed off 40 mm and took
95 steps. From the pins pressed at the face it backed off at most 8 mm, against 54 mm.

**Saved output** (`dagger/<round>/<scorer>/`):

- **Successful episodes are the DAgger demos.** Their full trajectories, policy and expert steps alike, with the
  executed actions as labels, go to `trajectories/`.
- **Every attempt**, failed ones included, gets a per-step record in `attempts/worker_<w>.h5`:
  - who drove each step;
  - the score of every observation;
  - VIG's distance per level;
  - the plug, gripper and goal poses.

  It also gets a summary line in `episodes/worker_<w>.jsonl`. `settings.json` records the thresholds and settings
  used.

**Stopping and workers:**

- `dagger.workers_per_scorer` processes per scorer (default: half the CPU budget each) share the scene sequence.
- They stop once the collection has 500 successful episodes, or after 20 × 500 attempts.
- The training set takes the first 500 successes to finish.
- A collection refuses to continue with changed thresholds or settings.

### DAgger metrics (`analyze_<round>`)

**Per attempt:**

- steps and expert steps;
- expert share (expert steps / steps);
- takeovers and returns;
- the step and the task stage of the first takeover;
- who finished a successful episode;
- why a failed one ended:
  - `time_limit_policy`: the step limit under the policy;
  - `time_limit_expert`: the step limit under the expert;
  - `expert_error`: the planner failed;
  - `expert_finished_failed`: the expert's plan ended without success;
- the furthest stage reached and failure flags;
- the share of policy-driven steps above the take threshold, and of expert-driven steps below the return threshold.

**Per takeover:** its start, length, stage, triggering score, and whether it ended in a return or at the episode
end.

**Stages** (`src/vig_dagger/stages.py`) are computed from simulator state:

Reach → Grasp → Lift → Transport → Align → Insert → Seated

**Failure flags:** dropped, misaligned at the socket face, tilted, off-center grasp, stalled.

**Collection metrics**, each for successful, failed and all attempts:

- attempts, successes, **collection success rate**, failure reasons;
- **pooled over all steps:** expert share, takeovers per 100 policy-driven steps;
- **distributions over episodes** (mean, std, min, p10, p25, median, p75, p90, max): steps, expert steps,
  **expert share**, **takeovers**, returns, first-takeover step; and over takeovers, the expert-segment length;
- the share of episodes without a takeover;
- **per stage:** steps, expert share, takeovers starting there, episodes with a takeover there.

**Outputs:**

- `results/dagger_<round>_<scorer>.json` (all of the above), `results/dagger_<round>_<scorer>_episodes.csv` and
  `_takeovers.csv`;
- the W&B run `<name>/dagger/<round>/<scorer>`: summary scalars, both tables, histograms of every per-episode
  quantity and of the scores on policy- and expert-driven steps, the per-stage table, and score timelines of six
  episodes;
- the W&B run `<name>/dagger/<round>/compare`: the two scorers side by side.

## Part 4: retraining and evaluation

Every retrained policy is a **dense full-resolution policy** (no experts, no image degradation), trained 60 epochs
with seeds 0, 1 and 2.

**Training sets.** All start with the same base demos, keeping their identities, so their train/validation
assignment is the same:

| training set | demos | used by |
|---|---|---|
| `base` | the 1000 base demos | both rounds (baseline) |
| `base_plus_planner` | base + 500 extra planner demos | both rounds (baseline) |
| `base_plus_moe_vig`, `base_plus_moe_flow` | base + round `moe`'s 500 VIG / flow-loss DAgger episodes (whole episodes: policy and expert steps) | round `moe` |
| `base_plus_per_resolution_vig`, `…_flow` | base + round `per_resolution`'s DAgger episodes | round `per_resolution` |

**Evaluation** (`eval/<policy>/…`):

- every retrained policy, and the MoE driver at full resolution;
- scenes 60,000–60,199, one episode at a time, CPU physics, 400 steps, 8 executed actions per prediction.

Each episode records success, steps, the furthest stage, the stage a failure ended in, failure flags, and the
plug's closest and final distance.

**Summary** (`summarize`), per training set:

- the success rate of each seed, their mean ± std;
- a paired comparison with `base`: scenes solved by only one of the two, over all seeds (seed *i* against seed
  *i*), with a two-sided sign test.

Outputs: `results/eval_summary.json`, `results/eval_episodes.csv`, W&B runs `<name>/eval/<policy>` and
`<name>/eval/summary`.

## Weights & Biases

All runs go to the team `vig-dagger-exp` (`wandb.entity`), project `wandb.project`, grouped by the experiment
`name`:

| run | content |
|---|---|
| `<name>/train/retrain/<set>/seed<s>` | training and validation loss, learning rate |
| `<name>/train/drivers/per_resolution/res<k>` (and `drivers/moe` if trained) | the same, for driving policies |
| `<name>/dagger/<round>/<scorer>`, `<name>/dagger/<round>/compare` | part 3 metrics (above) |
| `<name>/eval/<policy>`, `<name>/eval/summary` | part 4 metrics (above) |

Runs carry tags (`train`, `retrain`, the set, the seed, `dagger`, the round, `eval`, …) for filtering. Log in once
per machine with `wandb login`. Set `wandb.enabled: false` to run without W&B; every number is also written to
`<work_dir>/results`.

## Hugging Face

| repository | content |
|---|---|
| [chengenwu/PlugCharger-Hole4mm-SideCam-RandSocket-v1](https://huggingface.co/datasets/chengenwu/PlugCharger-Hole4mm-SideCam-RandSocket-v1) (dataset) | `raw/planner` (1000 demos), `raw/planner_extra` (500 demos) |
| [chengenwu/sd0_plug_charger_…_resmoe64_sd0](https://huggingface.co/chengenwu/sd0_plug_charger_hole4mm_sidecam_randsocket_vargrasp_1000mp_full_resmoe64_sd0) (model) | `base/moe/latest.pt` (the MoE driver), `base/thresholds.json` |

The queue downloads both itself. The same can be done by hand with `vig-dagger download-demos` and
`vig-dagger download-policy`.

`vig-dagger upload {demos, policy, dagger}` uploads to the repos in `hub.*`. It needs `huggingface-cli login` with
write access; new repositories are created private. Uploaded checkpoints are weights only: about half the size,
they load for inference but cannot resume training.

Existing files can be used instead of downloading:

- `vig-dagger import-demos part0.h5 …` links planner trajectory files (with their `.json`) as the raw demos, in
  argument order (`--extra` for the extra demos).
- `vig-dagger import-policy moe.pt --thresholds thresholds.json` installs a MoE checkpoint as round `moe`'s
  driver.

## Output directory

```
<work_dir>/
  config.json                                      the resolved configuration every job reads
  data/raw/planner/, planner_extra/                planner demos (ManiSkill format; links into hub/)
  data/datasets/{planner,base,planner_extra,base_plus_*,dagger_*}/   training datasets
  policies/drivers/moe/                            the MoE driver (latest.pt, thresholds.json)
  policies/drivers/per_resolution/res<k>/          the degraded-level policies of round per_resolution, thresholds.json
  policies/retrain/<set>/seed<s>/                  every retrained policy (latest.pt, run_config.json, metrics.jsonl)
  dagger/<round>/<scorer>/                         settings.json, trajectories/, attempts/, episodes/
  eval/<policy>/level<f>/seeds_<first>.jsonl       one line per evaluation episode
  results/                                         every metric as JSON / CSV
  logs/<job>.log, logs/pids/                       one log per job; PIDs of running jobs
  hub/                                             downloaded Hugging Face files
```

## Configuration reference

The defaults live in `src/vig_dagger/config.py`; `configs/plugcharger.yaml` holds the experiment's values.

| key | value | meaning |
|---|---|---|
| `name`, `work_dir` | | experiment name (W&B group) and output directory |
| `resources.gpus` | `auto` | every GPU, or a list of indices |
| `resources.sim_slots`, `max_load` | `auto`, `auto` | CPU budget (cores / 2) and load ceiling (cores) |
| `resources.sim_gpu_memory_gb`, `train_gpu_memory_gb` | 4, 8 | free GPU memory a simulator job / training needs (measured: 1.3 / 2.6–2.9 GB) |
| `resources.sim_ram_gb`, `train_ram_gb`, `train_slots` | 4, 24, 2 | RAM and CPU-budget estimates (measured: 2.5 / 19 GB; 1 / 1.3 cores) |
| `resources.dataloader_workers` | 16 | data-loading processes per training |
| `demos.source`, `num_demos` | `download`, 1000 | where the demos come from; the base set's size |
| `training.epochs` | 60 | passes over the training demos, for every training |
| `training.batch_size`, `learning_rate`, `warmup_updates`, `val_fraction` | 64, 1e-4, 1000, 0.02 | |
| `training.checkpoint_every`, `keep_checkpoints`, `max_restarts` | 5000, 1, 5 | rolling checkpoints and automatic restarts |
| `rounds` | `[moe, per_resolution]` | the rounds, in order |
| `drivers.moe.source` | `download` | `download` or `train` |
| `drivers.per_resolution.factors` | 1 … 64 | the levels VIG compares (1 = the base retraining of seed 0) |
| `scorers.vig.num_samples` | 32 | samples per level |
| `scorers.flow.num_loss_samples`, `num_action_samples` | 512, 1 | |
| `thresholds.num_demos`, `frame_stride`, `percentiles` | 50, 4, p50 … p99 | demo frames for the thresholds |
| `dagger.num_demos` | 500 | successful episodes per scorer and round |
| `dagger.thresholds.<scorer>.take` / `.return` | `p95` / `p50` | `p<q>` or a number |
| `dagger.take_count`, `return_count` | 2, 16 | consecutive steps needed to switch |
| `dagger.expert` | `demo` | `demo` (planner-demo motion) or `fast` (the earlier, faster expert) |
| `dagger.atomic_grasp` | false | the expert finishes a started grasp before handing back |
| `dagger.max_episode_steps`, `policy_chunk_steps`, `seed_start` | 400, 8, 1e6 | |
| `dagger.workers_per_scorer`, `max_attempts` | `auto`, 20 × 500 | |
| `retrain.seeds`, `planner_baseline` | `[0, 1, 2]`, true | |
| `evaluation.seed_start`, `num_episodes`, `max_episode_steps`, `chunk` | 60000, 200, 400, 25 | `chunk`: episodes per job |
| `evaluation.evaluate_drivers`, `driver_all_levels` | true, false | also evaluate the MoE driver (at every level) |
| `hub.dataset_repo`, `hub.model_repo` | see [Hugging Face](#hugging-face) | |
| `wandb.enabled`, `entity`, `project` | true, `vig-dagger-exp`, `plugcharger` | |

## Cost estimates

Measured on a shared machine (2× RTX PRO 6000, both GPUs also busy with other jobs). Expect better on a dedicated
machine.

| step | measured | for this experiment |
|---|---|---|
| download | 26 GB + 13 GB | |
| convert | ~1 s per demo; ~1.2 MB per demo | 1500 demos: a few minutes with 8 processes |
| one training, base set (1000 demos) | 4.5 updates/s on a busy GPU, 8–13 on a free one; ~190k updates | 4–12 h |
| one training, base + 500 demos | ~280k updates | 6–17 h |
| trainings in total | | 6 shared + 6 per round + 6 per-resolution drivers = 24 |
| thresholds | ~0.5 s per frame (VIG + flow) | ~2800 frames: ~25 min on one GPU (round 2 only; round 1's come with the MoE) |
| DAgger | per step: VIG ~0.35 s, flow ~0.15 s; ~30 MB per saved episode | 500 successes per scorer and round: probably 600–1500 attempts of 1–4 min; ~15 GB per collection |
| evaluation | ~10–25 s per episode | 19 policies × 200 episodes ≈ 20 CPU-hours |

Each training keeps one rolling checkpoint and the final one (~360 MB each).

Resources per job, measured:
- a training: 2.6–2.9 GB of GPU memory, ~19 GB of RAM (with its 16 data-loading workers), ~1.3 CPU cores;
- a simulator job: ~1.3 GB of GPU memory, 2.5 GB of RAM, about one core.

So the CPU cores, not the GPUs, limit how many DAgger workers and evaluations run at once.

## Troubleshooting

- **A job failed.** `vig-dagger status` and the queue log name it; its output is in `<work_dir>/logs/<job>.log`.
  Fix the cause and run the same `vig-dagger run` again; only unfinished jobs run.
- **Nothing starts.** The machine may be full: the queue waits for CPU budget, the load ceiling, RAM, or a GPU with
  enough free memory. On a machine shared with other users, lower `resources.train_gpu_memory_gb` or raise
  `resources.max_load` only if you know the jobs fit.
- **Slow simulation.** Each job limits itself to 4 OpenMP threads (`OMP_NUM_THREADS=4`). Many threads per process
  make parallel CPU physics several times slower.
- **Rendering errors** (`vk::`, "no device found"): SAPIEN needs a working Vulkan driver; see ManiSkill's
  installation notes.
- **Changed settings after jobs ran.**
  - A training keeps the run config it started with (`<run>/run_config.json`); delete the run directory to retrain
    it with new settings.
  - A DAgger collection refuses to continue with different thresholds or settings (only `num_demos` and
    `max_attempts` may change).

  Use a new `work_dir` or `name` for a new variant of the experiment.
- **W&B errors at startup** (a training checks its run online): run `wandb login`, or set `wandb.enabled: false`.

## Repository layout

```
configs/plugcharger.yaml        the experiment
configs/smoke.yaml              a short end-to-end check
src/vig_dagger/
  cli.py                        the vig-dagger command
  plan.py                       the experiment's job graph
  jobs.py                       the job queue (dependencies, resources, retries, adoption)
  steps.py                      short steps: downloads, conversion, thresholds, DAgger datasets, analysis, summary
  config.py                     defaults, rounds, and the output layout
  envs.py                       the PlugCharger environment
  expert/planner.py             the motion-planning expert (demos and DAgger takeovers)
  expert/interruptible_planner.py  stops the planner mid-motion when control returns to the policy
  policy.py                     running checkpoints: one policy, or the per-resolution set
  scoring.py, gate.py           VIG / flow loss, and the two-threshold gate
  recording.py, stages.py       episode recording, task stages and failure flags
  dataset.py, train.py          training datasets, run configs and the training worker
  analysis.py, evaluation.py    part 3 and part 4 metrics
  tracking.py, hub.py           W&B and Hugging Face
  workers/                      collect, convert, score_demos, dagger, evaluate (one process each)
  configs/policy.json           the policy recipe
src/robopolicy/                 the robopolicy training library
tests/robopolicy/               the library's tests
```

`src/robopolicy` is a copy of the robopolicy library (`refactor/policy-core`, branch `moe`, commit `ce03ad2`, with
the input-mode additions). It has two small additions: W&B `group`/`tags` (`training/logging.py`) and
`training.keep_checkpoints` (rolling checkpoint pruning, `training/trainer.py`).
