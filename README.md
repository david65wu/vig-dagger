# VIG DAgger: PlugCharger, three MoE seeds

DAgger collections on **PlugCharger** (ManiSkill: pick up a charger and insert its two pins into a socket that moves
from episode to episode) with two query scores deciding when the expert takes over from the policy:

- **VIG** (visual information gain): how much the policy's plan changes when its camera images are degraded.
- **Flow loss**: the policy's flow-matching loss on its own sampled action chunk.

The policy that drives DAgger is a **7-level resolution mixture-of-experts (MoE)**, a flow-matching Transformer
policy trained with the `robopolicy` library (included in this repository). There are three MoEs, trained the same
way on the same 1000 planner demos and differing only in their **seed**: seed 0 is downloaded from Hugging Face,
seeds 1 and 2 are trained here (60 epochs, like seed 0).

For each MoE, scorer, threshold and variant there is one **collection** of 500 successful DAgger episodes:

| | values |
|---|---|
| MoE seed | 0, 1, 2 |
| scorer | VIG, flow loss |
| threshold | p90, p95, p99 of the MoE's own score on demo frames. **The same value decides both directions:** the expert takes over after 2 scores above it, the policy takes back after 16 scores below it |
| variant | `plain`; `random_move`: at every takeover the gripper first moves to a random pose within 5 cm and 5° (unscored, gripper held), then the expert replans |

That makes 3 × 2 × 3 × 2 = **36 collections**. The results are three numbers per collection, and their mean and
spread over the three MoE seeds:

1. **collection success rate:** successful episodes / all attempts;
2. **expert share of the DAgger data:** the share of the steps in the 500 successful episodes that the expert drove
   (the random move's share shown separately);
3. **expert share of the failed attempts:** the same over the failed attempts.

`vig-dagger run` turns the whole experiment into one queue of jobs. A job starts when its inputs exist and the
machine has room for it (CPU, RAM, GPU memory), so independent work runs in parallel. Rerunning the same command
after an interruption skips finished jobs and picks up running ones. The work is also numbered as 40 **tasks**
([TASKS.md](TASKS.md)) that can be assigned to different machines.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The task](#the-task)
- [The job queue](#the-job-queue)
- [Part 1: demos](#part-1-demos)
- [Part 2: the MoEs and their thresholds](#part-2-the-moes-and-their-thresholds)
- [Part 3: DAgger](#part-3-dagger)
- [Results](#results)
- [Several machines](#several-machines)
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
- Disk: about 15 GB per collection, so about 580 GB for the whole experiment on one machine; see
  [cost estimates](#cost-estimates).
- Nothing else: every result is written to files in the work directory (`results/REPORT.md`, JSON, CSV,
  figures). Logging to W&B as well is optional (`wandb.enabled`).

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout plugcharger-moe-seeds
uv sync                       # creates .venv with the pinned packages (uv.lock)
source .venv/bin/activate     # or prefix commands with `uv run`
vig-dagger --help
```

The first training downloads the DINOv2-small weights from Hugging Face (`facebook/dinov2-small`, about 90 MB).

**Check the installation** with a short run of every kind of job on tiny numbers (4 demos, two MoEs trained for a
few dozen updates, 8 collections of one success each; about 15–30 minutes):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

The tests: `uv sync --extra dev && pytest tests -m "not cuda"` (the library's tests and this experiment's,
`tests/vig_dagger/`).

## Quick start

An agent operating the experiment without supervision should follow [AGENTS.md](AGENTS.md): the same steps,
plus what to check at each milestone, when to stop and ask, and what never to change.

1. In `configs/plugcharger.yaml`, set `work_dir` (the output directory). Everything else is set for the
   experiment; see the [configuration reference](#configuration-reference).
2. Check that the machine is ready: GPUs, rendering, `ffmpeg`, disk, Hugging Face access. Every line must
   say `ok`:

   ```bash
   vig-dagger check -c configs/plugcharger.yaml
   ```

3. Start the queue, detached so it survives a closed SSH session:

   ```bash
   setsid nohup vig-dagger run -c configs/plugcharger.yaml > plugcharger.log 2>&1 < /dev/null &
   ```

   With several machines, each runs only the tasks assigned to it: `vig-dagger run -c … --tasks 1 4-9` (see
   [Several machines](#several-machines)).
4. Watch it:

   ```bash
   vig-dagger status -c configs/plugcharger.yaml   # queue alive? failed / running jobs with progress; milestones
   tail -f plugcharger.log                         # the queue's own log: every job started, finished, retried
   ```

   Each job's output is in `<work_dir>/logs/<job>.log`.
5. After a crash, a reboot or a fix, run the same command again.
6. To stop: `vig-dagger stop -c configs/plugcharger.yaml`. Every job resumes where it stopped on the next `run`.
   With `--queue-only`, only the queue stops: the running jobs continue, and the next `run` takes them over, e.g. to
   change `resources` settings.
7. Results: `<work_dir>/results/REPORT.md` and `results/figures/collections.png` (see [Results](#results)). Every
   collection's analysis refreshes them; `vig-dagger report -c configs/plugcharger.yaml` refreshes them at any time.

Useful options:

- `--set key=value` overrides any config value, e.g. `--set resources.sim_slots=16`.
- `--only REGEX` runs only matching jobs and what they need, e.g. `--only '^dagger_sd0_'`.
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
| Step limit | 300 for planner demos, 400 for DAgger |
| Physics | CPU (one environment per process); rendering on the GPU |

**Scene seeds.** A seed fixes the charger and socket placement. Every collection runs the same scene sequence,
from seed 1,000,000 on, so the collections are paired scene by scene. The socket position comes from torch's random
generator, so a seed gives a different scene under a different torch version; inside this repository torch is
pinned (`uv.lock`), so seeds are consistent across machines that use it. The base demos used seeds 0–40,049 and
70,000–210,052; collected demos (`demos.source: collect`) start at 2,000,000.

## The job queue

`vig-dagger run` builds every job of the experiment (`src/vig_dagger/plan.py`) and hands them to a queue
(`src/vig_dagger/jobs.py`). Each job is one process with its own log.

**When a job starts.** Once the jobs it depends on are done, and the machine has room:

- **CPU:** simulator jobs (DAgger workers, threshold scoring, demo collection) and conversions use a CPU budget,
  `resources.sim_slots` (default: one slot per 2 cores; a simulator job uses about one core). A training counts as
  `resources.train_slots` (its data loading).
- **Load:** nothing new starts while the 1-minute load average is above `resources.max_load` (default: the number
  of cores), so other users' work is respected.
- **RAM:** the available memory must cover the job's estimate.
- **GPU:**
  - a simulator job goes to the GPU with the most free memory, if it has at least `resources.sim_gpu_memory_gb`;
  - a training needs `resources.train_gpu_memory_gb` free on a GPU that runs none of our other trainings.

Jobs started in the last 3 minutes count with their estimates.

**Order.** Lower priority numbers go first: the demos, the MoE trainings (they take hours, so they start first),
the thresholds, then the collections in task order (MoE seed, variant, scorer, threshold). Each collection has
`dagger.workers_per_collection` workers (default: half the CPU budget), so on one machine about two collections run
at a time while the MoEs of seeds 1 and 2 train on the GPUs. Seed 0's collections start as soon as its thresholds
are computed (about an hour after the start); seed 1's and 2's after their trainings.

**Failures and restarts:**

- A job that fails is retried a few times (trainings up to `training.max_restarts` times). Every job continues
  where it stopped: trainings from their last checkpoint (every 5000 updates), DAgger workers from their last
  finished episode.
- A job that still fails stops only the jobs that depend on it. The rest of the queue keeps going.
- If the queue process dies, the jobs it started keep running. A restarted queue adopts them instead of starting
  them twice. Only one queue runs per work directory.

## Part 1: demos

`download_demos` fetches `raw/planner` from `hub.dataset_repo`: 20 files of 50 motion-planner demos, the 1000 demos
every MoE trains on (seed 0 was trained on exactly these). The demos are needed for the trainings of seeds 1 and 2
and for every MoE's thresholds.

**The planner** (`src/vig_dagger/expert/planner.py`) samples a strategy for each demo:

- grasp tilt 5–25° and offset ±8 mm;
- a lift waypoint;
- an off-axis approach that is corrected before a straight insertion;
- joint speeds;
- screw or RRT transit.

So no single joint path solves the task. Only successful demos that finish within 300 steps are kept.

**Converted dataset.** `convert_planner` turns the raw demos into the library's training format (H.264 videos plus
HDF5 with actions and joint state): `data/datasets/planner` (every demo) and `data/datasets/base` (the first
`demos.num_demos`, 1000). It runs only when a MoE is trained here. The conversion reproduces the demo lengths of
the dataset seed 0 was trained on.

With `demos.source: collect`, the pipeline collects the demos itself instead (seeds from 2,000,000).

## Part 2: the MoEs and their thresholds

**Policy recipe** (`src/vig_dagger/configs/policy.json`):

- frozen DINOv2-small encoder per camera, cross-attention pooling to one token per camera;
- an 8-layer Transformer action head (width 256) with adaptive LayerNorm on the flow time and the current joint
  state;
- linear-path flow matching with 10 Euler sampling steps;
- batch 64, AdamW, lr 1e-4 with cosine decay;
- random 224 px crops during training.

**The MoE:**

- every training sample is degraded to a random level (factor 1, 2, 4, …, 64: area-downsampling, then bilinear
  upsampling);
- every head layer's feed-forward block has 8 experts (top-2) with one router per level.

**Seed 0** (`download_moe_sd0`) comes from `moe.download.sd0` on Hugging Face: its checkpoint, training log and run
config. **Seeds 1 and 2** (`train/moe/sd1`, `train/moe/sd2`) are trained here with the same recipe for
`training.epochs` = 60 passes over the 1000 demos:

max_updates = 60 × ceil(training samples / 64) = 188,580

(a training sample is any step with a full 16-step future; 2 % of the demos are held out for the validation loss).
That is exactly seed 0's training length. Only the seed differs: it sets the initialization, the training noise and
the data order (sampling seed 42 + seed, so 43 and 44; seed 0 used 42). The validation demos are the same for
every seed.

**Thresholds** (`score_sd<s>_*`, `thresholds_sd<s>`), for every MoE alike (seed 0's own thresholds on Hugging Face
are not used):

- **Frames scored:** every 4th frame of the first 50 base demos (about 2,800 frames).
- **Scored like DAgger:** VIG with 32 samples per level; flow loss with 512 draws.
- **Output:** the percentiles (p50 … p99) go to `policies/moe/sd<s>/thresholds.json`; the scores themselves to
  `demo_scores.npz`.

### The two scores

Both are computed on the MoE's normalized actions at full resolution (`src/vig_dagger/scoring.py`).

- **VIG.** Draw 32 chunks at full resolution and at every degraded level, all from the same 32 starting noises.
  For each degraded level, take the 1-D Wasserstein distance between the two sample sets: sort each of the
  16 × 8 chunk cells, compare, and average over the cells. VIG is the largest of these distances over the six
  degraded levels.
- **Flow loss.** Draw one chunk at full resolution and average the policy's flow-matching loss on that chunk over 512
  random (noise, time) draws. It is noisy from frame to frame: two evaluations of the same frame differ by about 50 %
  on average.

## Part 3: DAgger

**A collection** (`dagger/<collection>/`, e.g. `dagger/sd1_vig_p95_random_move/`) is one MoE seed, one scorer, one
threshold and one variant. Its workers share the scene sequence and stop once it has 500 successful episodes (or
after 20 × 500 attempts).

**Each episode** (`src/vig_dagger/workers/dagger.py`, gate in `src/vig_dagger/gate.py`):

1. The MoE drives at full resolution. Every observation is scored.
2. **Take over:** the expert takes over after **2** consecutive scores above the **threshold** (p90, p95 or p99 of
   the MoE's score on demo frames).
3. **Random move** (variant `random_move` only): before the expert plans, the gripper moves to a random pose (below).
4. **Return:** the score keeps being computed after every expert step. Control returns to the policy after **16**
   consecutive scores below the **same threshold**, for the rest of the same episode. Without a return, the
   expert finishes the task.
5. **End:** the episode ends at success, at the 400-step limit (policy, random move and expert alike), or when the
   planner fails.

**The expert** (`resume_and_solve`, `dagger.expert: demo`) moves like the planner demos: the strategy's sampled
speeds and the same stage sequence, with a fresh strategy at each takeover. It starts from the current state:

- **Plug not held:** reach, grasp and lift. It skips the reach detour if the gripper is already over the charger.
- **Taking over right after a grasp:** it first does the demo's lift.
- **Plug held but still far from the socket:** the demo's approach, alignment and insertion.
- **Plug held near the socket** (closer than the strategy's approach distance): it corrects in place instead of
  pulling back to the approach pose. It aligns onto the socket axis at the plug's current distance, and backs off
  only to 5 mm in front of the face when the pins are against the face but not in the holes. Then it inserts.

**The random move** (`src/vig_dagger/expert/random_move.py`, `dagger.random_move`; the same code as the earlier
PlugCharger experiments in maniskill):

- **Target:** a position uniform inside a 5 cm ball around the gripper tip, the orientation turned by 0–5°
  (uniform) about a uniformly random axis.
- **Gripper:** held at the policy's last command, so a held charger moves with the hand.
- **Collision-free:** a target is redrawn (up to 10,000 times) if the screw plan fails (IK, joint limits) or if the
  hand, fingers or held charger would get deeper into the socket walls, the table or the free charger (grown by
  4 mm) than they already are. A plug with its pins in the holes can therefore only be pulled straight out. If no
  target is valid, the takeover has no move (logged as skipped).
- **Speed:** a screw motion at the transit speed of the strategy the expert drew for this takeover.
- **Recorded, not scored:** its steps count toward the 400-step limit, are not scored, and are recorded with their
  own driver code (2), so they are told apart from the expert's (1) and the policy's (0) everywhere. The hand-back
  count starts after the move.

On identical simulator states and random draws, this code accepts and rejects the same targets and plans the same
paths as the earlier maniskill version (45 of 45 cases checked: free hand, charger in transport, pins at and in the
holes).

**Saved output** (`dagger/<collection>/`):

- **Successful episodes are the DAgger demos.** Their full trajectories, policy, random-move and expert steps alike,
  with the executed actions as labels and the driver of every step (`traj_<n>/dagger/controller`), go to
  `trajectories/`.
- **Every attempt**, failed ones included, gets a per-step record in `attempts/worker_<w>.h5`: who drove each step,
  the score of every observation, VIG's distance per level, the plug, gripper and goal poses. Failed attempts keep
  only this record, not their images (`dagger.save_failed_trajectories`).
- Every attempt also gets a summary line in `episodes/worker_<w>.jsonl`, with each random move (tries, distance,
  rotation, steps). `settings.json` records the threshold and settings; a collection refuses to continue with
  changed ones.

### Collection metrics (`analyze_<collection>`)

For four groups of attempts: **data** (the first 500 successful episodes to finish: the collection's DAgger data),
successful, failed and all:

- **pooled over steps:** expert share and random-move share (the policy drove the rest), takeovers per 100
  policy-driven steps;
- **distributions over episodes** (mean, std, min, p10, p25, median, p75, p90, max): steps, expert share,
  random-move share, takeovers, returns, random moves, first-takeover step; over takeovers, their length;
- per task stage (Reach, Grasp, Lift, Transport, Align, Insert, Seated): steps, expert share, takeovers;
- **collection:** attempts, successes, collection success rate, failure reasons.

A takeover is every step from the takeover to the hand-back: the random move, then the expert.

## Results

Everything is written to `<work_dir>/results`, no account needed. `results/REPORT.md` collects it in one file:

- **Progress**, and how many collections are finished.
- **The figure** `results/figures/collections.png`: three panels (success rate per attempt; expert share of the
  DAgger data; expert share of the failed attempts), one bar per scorer, threshold and variant. A bar is the mean
  over the MoE seeds, a dot each seed; the random move's share is stacked in orange on the expert's (blue).
- **Per setting** (scorer × variant × threshold): the three numbers as mean ± std over the seeds, successes / attempts
  summed over the seeds, and the mean over seeds of the per-episode median expert share.
- **Every collection:** its threshold value, attempts, successes, success rate, the expert and random-move shares of
  its data and of its failed attempts (pooled / per-episode median), takeovers, random moves (skipped), failure
  reasons.
- **Random moves** (`results/figures/moves/`): for every MoE seed and threshold, two figures of the random-move
  collections' moves, VIG and flow loss side by side, seen from the third-person camera. Each move in a collection's
  DAgger data is drawn as the outline of the 5 cm sphere its target was drawn from (centred on the gripper tip at the
  take-over), a dot at the centre and an arrow to where the tip ended, on the first frame of scene 1,000,000
  (grayscale). `spheres_held_sd<s>_p<q>.png` colours them by whether the plug was held at the take-over,
  `spheres_progress_sd<s>_p<q>.png` by when in the episode the take-over happened. The same plotting as the earlier
  maniskill random-move figures (it reproduces them from that run's moves).
- **Thresholds** of each MoE, and the **trainings** (updates, losses; `figures/training_moe.png`).

Only finished collections enter the per-setting means; a collection analyzed by hand mid-run is listed as running.

| file | content |
|---|---|
| `results/summary.json`, `summary.csv` | the tables above: one row per collection, one per setting |
| `results/collections/<collection>.json` | every metric of one collection |
| `results/collections/<collection>_episodes.csv`, `_takeovers.csv` | one row per attempt, per takeover |
| `results/collections/<collection>_scores.npz`, `_steps.npz` | the scores on policy- and expert-driven steps; every attempt's stage, driver, score and flags per step |
| `results/collections/<collection>_page.json.gz` | the data of an interactive page of the collection (format in `src/vig_dagger/page_data.py`) |
| `results/collections/<collection>_moves.json`, `_background.png` | random-move collections: every move (start, target, end, plug held, take-over step, episode), the third-person camera's parameters, and the background frame |
| `policies/moe/sd<s>/metrics.jsonl` | each training's log: `train/loss`, `train/lr`, `val/loss` per `train/step` |

With `wandb.enabled: true` each collection's metrics also go to a W&B run `<name>/dagger/<collection>` (team
`wandb.entity`, project `wandb.project`), and trainings log their curves there.

## Several machines

The work is a numbered list of 40 tasks, [TASKS.md](TASKS.md) (`vig-dagger tasks`):

| tasks | what | needs |
|---|---|---|
| 1–3 | the MoE of seed 0, 1, 2 and its thresholds (1: download, under an hour; 2, 3: a training of 8–14 h on one GPU) | – |
| 4–39 | the 36 collections, in the order MoE seed, variant, scorer, threshold | the task of its MoE (1, 2 or 3) |
| 40 | the summary: the figure, the tables, `REPORT.md` | the metrics of every collection |

- Every machine has its own clone, environment and work directory, and the same `configs/run.yaml` settings (only
  `work_dir` and `resources` may differ), so the task numbers mean the same everywhere.
- `vig-dagger run -c … --tasks 2 16-27` runs those tasks on this machine, with the preparation they need
  (downloading and converting the demos). It refuses if an input from another task is missing, and names it.
- **Moving outputs:** on the machine that ran a task, `vig-dagger pack -c … --tasks 2 --for 16` writes one archive
  of what task 16 needs of task 2's outputs (the MoE checkpoint, weights-only, ~180 MB, and its thresholds). On the
  machine that runs task 16: `vig-dagger unpack -c … <that file>`. For the summary: `pack --tasks 4-39 --for 40` (a
  few MB of metrics per collection). How the archives travel is up to the humans.
- The DAgger demos themselves (`dagger/<collection>/trajectories/`, ~15 GB per collection) stay on the machine that
  collected them; `pack --tasks N` without `--for` packs a collection completely.

## Hugging Face

| repository | content |
|---|---|
| [chengenwu/PlugCharger-Hole4mm-SideCam-RandSocket-v1](https://huggingface.co/datasets/chengenwu/PlugCharger-Hole4mm-SideCam-RandSocket-v1) (dataset, public) | `raw/planner`: the 1000 demos the MoEs train on (24 GB) |
| [chengenwu/sd0_plug_charger_hole4mm_sidecam_randsocket_vargrasp_1000mp_full_resmoe64_sd0](https://huggingface.co/chengenwu/sd0_plug_charger_hole4mm_sidecam_randsocket_vargrasp_1000mp_full_resmoe64_sd0) (model, public) | `base/moe/latest.pt`: the seed-0 MoE (386 MB), its training log and run config |

Downloads need no login. `vig-dagger download-demos` and `download-moe --seed 0` do by hand what the queue does.
`vig-dagger upload demos` / `upload dagger` upload the demos / every collection to `hub.dataset_repo` (needs
`hf auth login` with write access; new repositories are created private). Existing files can be used instead of
downloads: `vig-dagger import-demos part0.h5 …` (in part order) and `import-moe CKPT --seed S`.

## Output directory

```
<work_dir>/
  config.json                                the resolved configuration every job reads
  data/raw/planner/                          planner demos (ManiSkill trajectories, part_<k>.h5; links into hub/)
  data/datasets/{planner,base}/              the training dataset (only when a MoE is trained here)
  policies/moe/sd<s>/                        a MoE: latest.pt, run_config.json (or train_config.json when
                                             downloaded), metrics.jsonl, thresholds.json, demo_scores.npz
  dagger/<collection>/                       settings.json, trajectories/, attempts/, episodes/
  results/                                   REPORT.md, summary.*, collections/, figures/
  logs/<job>.log, logs/pids/                 one log per job; PIDs of running jobs
  packs/                                     archives written by `vig-dagger pack`
  hub/                                       downloaded Hugging Face files
```

## Configuration reference

The defaults live in `src/vig_dagger/config.py`; `configs/plugcharger.yaml` holds the experiment's values.

| key | value | meaning |
|---|---|---|
| `name`, `work_dir` | | experiment name and output directory |
| `resources.gpus` | `auto` | every GPU, or a list of indices |
| `resources.sim_slots`, `max_load` | `auto`, `auto` | CPU budget (cores / 2) and load ceiling (cores) |
| `resources.sim_gpu_memory_gb`, `train_gpu_memory_gb` | 4, 8 | free GPU memory a simulator job / training needs |
| `resources.sim_ram_gb`, `train_ram_gb`, `train_slots` | 4, 24, 2 | RAM and CPU-budget estimates |
| `resources.dataloader_workers` | 16 | data-loading processes per training |
| `demos.source`, `num_demos`, `demos_per_part` | `download`, 1000, 50 | `download` or `collect`; the MoEs' training set; demos per collection job |
| `training.epochs` | 60 | passes over the training demos (= 188,580 updates on the 1000 demos) |
| `training.batch_size`, `learning_rate`, `warmup_updates`, `val_fraction` | 64, 1e-4, 1000, 0.02 | |
| `training.checkpoint_every`, `keep_checkpoints`, `max_restarts` | 5000, 1, 5 | rolling checkpoints and automatic restarts |
| `moe.seeds`, `factors` | `[0, 1, 2]`, 1 … 64 | the MoE seeds; their resolution levels |
| `moe.download` | `{sd0: …}` | seeds taken from a Hugging Face model repo instead of trained here |
| `scorers.vig.num_samples` | 32 | samples per level |
| `scorers.flow.num_loss_samples`, `num_action_samples` | 512, 1 | |
| `thresholds.num_demos`, `frame_stride`, `percentiles` | 50, 4, p50 … p99 | demo frames for the thresholds |
| `dagger.scorers`, `thresholds`, `variants` | `[vig, flow]`, `[p90, p95, p99]`, `[plain, random_move]` | the collections (with `moe.seeds`) |
| `dagger.num_demos` | 500 | successful episodes per collection |
| `dagger.take_count`, `return_count` | 2, 16 | consecutive scores above / below the threshold needed to switch |
| `dagger.random_move.radius`, `rot_deg`, `max_tries`, `margin`, `tol` | 0.05, 5, 10000, 0.004, 0.0005 | the random move (m, deg) |
| `dagger.expert` | `demo` | `demo` (planner-demo motion) or `fast` (the earlier, faster expert) |
| `dagger.max_episode_steps`, `policy_chunk_steps`, `seed_start` | 400, 8, 1e6 | |
| `dagger.workers_per_collection`, `max_attempts` | `auto`, 20 × 500 | `auto`: half the CPU budget |
| `dagger.save_failed_trajectories` | false | also keep failed attempts' images |
| `hub.dataset_repo`, `revision` | see [Hugging Face](#hugging-face), `main` | |
| `wandb.enabled`, `entity`, `project` | false, `vig-dagger-exp`, `plugcharger` | optional W&B logging |

## Cost estimates

Measured on a shared machine (2× RTX PRO 6000 Blackwell, 64 cores, both GPUs also busy with other jobs). Expect
better on a dedicated machine.

| step | measured | for this experiment |
|---|---|---|
| download | 24 GB demos, 0.4 GB MoE | minutes to an hour |
| convert | ~1 s per demo; ~1.2 MB per demo | 1000 demos: a few minutes with 8 processes |
| one MoE training (1000 demos, 188,580 updates) | 4.5 updates/s on a busy GPU, 6.5–13 on a free one | 4–12 h; two of them, in parallel on two GPUs |
| thresholds | ~0.5 s per frame (VIG + flow) | ~2800 frames per MoE: ~25 min on one GPU |
| DAgger | per step: VIG ~0.35 s, flow ~0.15 s; 1–4 min per attempt; ~30 MB per saved episode | per collection ~550–1,300 attempts (VIG at p90 ~1.1 attempts per success, flow ~2.5), roughly 20–60 worker-hours and ~15 GB |
| in total | | 36 collections: roughly 1,200–1,500 worker-hours, so about 3 days on one machine with ~20 workers at a time, less on several |

Disk in total, about 580 GB on one machine:
- the 36 collections, about 540 GB (~30 MB per saved successful episode; failed attempts keep only their per-step
  record, a few MB per collection);
- the downloaded demos, 24 GB; the converted dataset, ~1.2 GB; the checkpoints, ~2 GB;
- logs and results, a few GB.

Resources per job, measured:
- a training: 2.6–2.9 GB of GPU memory, ~19 GB of RAM (with its 16 data-loading workers), ~1.3 CPU cores;
- a simulator job: ~1.3 GB of GPU memory, 2.5 GB of RAM, about one core.

So the CPU cores, not the GPUs, limit how many DAgger workers run at once.

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
- **`run --tasks` says "cannot run yet"**: another task's outputs are not in this work directory; bring them with
  `vig-dagger pack … --for <this task>` on the machine that has them and `vig-dagger unpack` here.
- **Changed settings after jobs ran.**
  - A training keeps the run config it started with (`<run>/run_config.json`); delete the run directory to retrain
    it with new settings.
  - A DAgger collection refuses to continue with a different threshold or settings (only `num_demos` and
    `max_attempts` may change).

  Use a new `work_dir` or `name` for a new variant of the experiment.
- **W&B errors at startup** (only with `wandb.enabled: true`; a training checks its run online): run `wandb login`, or
  set `wandb.enabled: false`.

## Repository layout

```
configs/plugcharger.yaml        the experiment
configs/smoke.yaml              a short end-to-end check
TASKS.md                        the numbered tasks of configs/plugcharger.yaml (several machines)
src/vig_dagger/
  cli.py                        the vig-dagger command
  plan.py                       the experiment's job graph
  tasks.py                      numbered tasks, run --tasks, pack / unpack
  jobs.py                       the job queue (dependencies, resources, retries, adoption)
  steps.py                      short steps: downloads, conversion, thresholds, collection settings, analysis
  config.py                     defaults, the collections, and the output layout
  envs.py                       the PlugCharger environment
  expert/planner.py             the motion-planning expert (demos and DAgger takeovers)
  expert/random_move.py         the random end-effector move at a takeover (variant random_move)
  expert/interruptible_planner.py  stops the planner mid-motion when control returns to the policy
  policy.py                     running checkpoints
  scoring.py, gate.py           VIG / flow loss, and the gate (one threshold for both directions here)
  recording.py, stages.py       episode recording, task stages and failure flags
  dataset.py, train.py          the training dataset, run configs and the training worker
  analysis.py, dagger_stats.py  collection metrics
  summary.py, report.py         every collection side by side, the figure, results/REPORT.md
  move_figures.py               the random moves seen from the third-person camera
  page_data.py                  the data of an interactive page per collection
  tracking.py, hub.py           optional W&B logging, and Hugging Face
  workers/                      collect, convert, score_demos, dagger (one process each)
  configs/policy.json           the policy recipe
src/robopolicy/                 the robopolicy training library
tests/robopolicy/               the library's tests
tests/vig_dagger/               this experiment's tests (gate, collections, tasks, random move, metrics, figures)
```

`src/robopolicy` is a copy of the robopolicy library (`refactor/policy-core`, branch `moe`) with two small additions:
W&B `group`/`tags` (`training/logging.py`) and `training.keep_checkpoints` (rolling checkpoint pruning,
`training/trainer.py`).
