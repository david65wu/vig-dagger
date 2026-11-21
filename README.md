# VIG DAgger: PlugCharger, hand-back threshold sweep

DAgger collections on **PlugCharger** (ManiSkill: pick up a charger and insert its two pins into a socket that moves
from episode to episode). The **VIG** score (visual information gain: how much the policy's plan changes when its
camera images are degraded) decides when the expert takes over from the policy and when it hands control back.

The policy that drives DAgger is the **seed-0 7-level resolution mixture-of-experts (MoE)**, a flow-matching
Transformer policy trained with the `robopolicy` library (included in this repository) on the 1000 planner demos,
downloaded from Hugging Face. The expert is the slow **'demo' expert**: its strategies (grasp, standoff, lift,
approach, planner) and speeds are drawn exactly as the planner demos' were, which the pipeline checks against all 1000
demos before the first collection.

The expert takes over after 2 VIG scores above **p90** of the MoE's own score on demo frames. The sweep is over the
**hand-back threshold**, below which the policy takes back control after 16 scores in a row:

| | values |
|---|---|
| take-over threshold | p90 (fixed) |
| hand-back threshold | **p70, p75, p80, p85** (one collection each) |
| variant | `random_move`: at every takeover the gripper first moves to a random pose within 5 cm and 5° (unscored, gripper held), then the expert replans |

That makes **4 collections** of 500 successful DAgger episodes. The results per collection:

1. **collection success rate:** successful episodes / all attempts;
2. **expert share of the DAgger data:** the share of the steps in the 500 successful episodes that the expert drove;
3. **expert share of all attempts:** the same over every attempt, successful or failed;
4. **expert share of the failed attempts:** the same over the failed attempts (and the expert steps per failed attempt);

each with the random move's share shown separately. Then each collection's DAgger data is used for training: per
collection one **dense full-resolution policy** (no experts) on the 1000 base demos + the collection's 500 DAgger
episodes **with the random-move steps cut out**, 60 epochs, training seed 0, evaluated on the same 200 scenes as the
MoE, with paired comparisons.

**The reference** (`reference/sd0_vig_p90_random_move`): the same collection with hand-back at p90 (one threshold
both ways), made earlier with the same MoE, expert and settings, and its retrained policy, shown next to the sweep
as its p90 end. It is used only while its settings and the collection code match (`src/vig_dagger/reference.py`).

`vig-dagger run` turns the whole experiment into one queue of jobs. A job starts when its inputs exist and the
machine has room for it (CPU, RAM, GPU memory), so independent work runs in parallel. Rerunning the same command
after an interruption skips finished jobs and picks up running ones. The work is also numbered as 12 **tasks**
([TASKS.md](TASKS.md)) for several machines: assigned by hand, or claimed through a task board.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The task](#the-task)
- [The job queue](#the-job-queue)
- [Part 1: demos](#part-1-demos)
- [Part 2: the MoEs and their thresholds](#part-2-the-moes-and-their-thresholds)
- [Part 3: DAgger](#part-3-dagger)
- [Part 4: retraining and evaluation](#part-4-retraining-and-evaluation)
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

   With several machines, set `sharing.board` (the same on every machine) and run the same command on each: they
   share the tasks among themselves (see [Several machines](#several-machines)).
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

**Order.** Lower priority numbers go first: the demos, the MoE (downloaded), the thresholds, then the collections in
task order (hand-back p70, p75, p80, p85). Each collection has `dagger.workers_per_collection` = 4 workers, so all four
collections run at once on a machine with a CPU budget of 16; they start as soon as the thresholds are computed
(about an hour after the start). Each retraining starts when its collection is finished, one per GPU at a time.

**Failures and restarts:**

- A job that fails is retried a few times (trainings up to `training.max_restarts` times). Every job continues
  where it stopped: trainings from their last checkpoint (every 5000 updates), DAgger workers from their last
  finished episode.
- A job that still fails stops only the jobs that depend on it. The rest of the queue keeps going.
- If the queue process dies, the jobs it started keep running. A restarted queue adopts them instead of starting
  them twice. Only one queue runs per work directory.

## Part 1: demos

`download_demos` fetches `raw/planner` from `hub.dataset_repo`: 20 files of 50 motion-planner demos, the 1000 demos
the seed-0 MoE was trained on. They are needed for its thresholds, as the base of every retraining, and for the
expert check (their recorded strategies).

**The planner** (`src/vig_dagger/expert/planner.py`) samples a strategy for each demo:

- grasp tilt 5–25° and offset ±8 mm;
- a lift waypoint;
- an off-axis approach that is corrected before a straight insertion;
- joint speeds;
- screw or RRT transit.

So no single joint path solves the task. Only successful demos that finish within 300 steps are kept.

**Converted dataset.** `convert_planner` turns the raw demos into the library's training format (H.264 videos plus
HDF5 with actions and joint state): `data/datasets/planner` (every demo) and `data/datasets/base` (the first
`demos.num_demos`, 1000). It runs only when a MoE is trained here. The demos are named as in the dataset seed 0 was
trained on (`demos.demo_names`: `sd0_plug_charger_…_part<k>__traj_<n>`), because a demo's name is part of what decides
whether it is held out for validation: with these names the converted dataset is identical to seed 0's, demo for
demo (shard, name, length), and every seed holds out the same 25 demos. A dataset converted with other names (an
earlier version of this branch named them after the files, `part_000__traj_0`) is refused, with what to delete.

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

max_updates = 60 × ceil(training samples / 64) = 60 × ceil(201,099 / 64) = 188,580

(a training sample is any step with a full 16-step future; about 2 % of the demos, 25 of the 1000, are held out for
the validation loss). That is exactly seed 0's training length, on exactly seed 0's 975 training demos. A training
whose saved `run_config.json` was written for another dataset refuses to resume and names the run directory to delete. Only the seed differs: it sets the initialization, the training noise and
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

**A collection** (`dagger/<collection>/`, e.g. `dagger/sd0_vig_p90_r75_random_move/`: take over above p90, hand back
below p75) is one MoE seed, one scorer, one take-over and one hand-back threshold and one variant. Its workers share the scene sequence and stop once it has 500 successful episodes (or
after 20 × 500 attempts).

**Each episode** (`src/vig_dagger/workers/dagger.py`, gate in `src/vig_dagger/gate.py`):

1. The MoE drives at full resolution. Every observation is scored.
2. **Take over:** the expert takes over after **2** consecutive scores above the **take-over threshold** (p90 of the
   MoE's score on demo frames).
3. **Random move** (variant `random_move` only): before the expert plans, the gripper moves to a random pose (below).
4. **Return:** the score keeps being computed after every expert step. Control returns to the policy after **16**
   consecutive scores below the **hand-back threshold** (p70, p75, p80 or p85; between the two thresholds nobody
   hands over), for the rest of the same episode. Without a return, the
   expert finishes the task.
5. **End:** the episode ends at success, at the 400-step limit (policy, random move and expert alike), or when the
   planner fails.

**The expert** (`resume_and_solve`, `dagger.expert: demo`) moves like the planner demos: the strategy's sampled
speeds and the same stage sequence, with a fresh strategy at each takeover, drawn by the planner demos' own sampler.
`dagger_prepare` checks this before a collection starts: it regenerates every planner demo's recorded strategy (all
16 fields, both speeds included) from its seed and refuses to start if any differs. It never resets or rewinds the
simulation; it starts from the current state:

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
  the score of every observation, VIG's distance per level, the plug, gripper and goal poses. Failed attempts are
  also saved in full to `trajectories_failed/` (`dagger.save_failed_trajectories: true`, as in the reference).
- Every attempt also gets a summary line in `episodes/worker_<w>.jsonl`, with each random move (tries, distance,
  rotation, steps). `settings.json` records the threshold and settings; a collection refuses to continue with
  changed ones.

### Collection metrics (`analyze_<collection>`)

For four groups of attempts: **data** (the first 500 successful episodes to finish: the collection's DAgger data),
successful, failed and **all** (every attempt; the report's "expert share of all attempts"):

- **pooled over steps:** expert share and random-move share (the policy drove the rest), takeovers per 100
  policy-driven steps;
- **distributions over episodes** (mean, std, min, p10, p25, median, p75, p90, max): steps, expert share,
  random-move share, takeovers, returns, random moves, first-takeover step; over takeovers, their length;
- per task stage (Reach, Grasp, Lift, Transport, Align, Insert, Seated): steps, expert share, takeovers;
- **collection:** attempts, successes, collection success rate, failure reasons.

A takeover is every step from the takeover to the hand-back: the random move, then the expert.

## Part 4: retraining and evaluation

**One retraining per collection** (task `retrain/<collection>`), once the collection is finished:

1. `dagger_dataset_<collection>` builds `data/datasets/dagger_<collection>` from the collection's DAgger data (its
   first 500 successful episodes to finish, the same ones its metrics call "the data"), with the executed actions as
   labels and **the random-move steps cut out** (`retrain.cut_random_moves`): every episode is split at its random
   moves into the pieces before, between and after them (`<demo>_s<start>-<end>`); since a training sample needs its
   full 16-step action chunk inside a demo, the 15 steps before each move leave training too, while the expert's
   recovery after each move stays in; pieces under 16 steps are dropped. The log and the dataset manifest count the
   moves and steps cut. (The same cut as maniskill's `split_dagger_dataset_at_random_moves.py`: identical pieces and
   actions.) Then `data/datasets/base_plus_<collection>` = the 1000 base demos (first, keeping their identities, so
   the same 25 are held out for validation as for the MoE) + those pieces.
2. `train/retrain/<collection>/seed0` trains a **dense full-resolution policy**: the policy recipe without the
   experts and without image degradation, `training.epochs` = 60 passes over the training demos (about 250–320k
   updates), training seed `retrain.seeds` (0) → `policies/retrain/<collection>/seed0/`.
3. `eval/retrain/<collection>/seed0/<first scene>` evaluates it: 200 scenes (`evaluation.seed_start` 60,000 –
   60,199), one episode at a time with CPU physics, 400 steps, 8 executed actions per prediction, in jobs of 25.

The retraining needs the collection's episodes (~15 GB), so it runs on the machine that collected them.

**The MoE** (task `eval/moe/sd0`) is evaluated on the same 200 scenes at full resolution: every retrained policy is
compared with it, and with the reference's retrained policy, scene by scene.

**Evaluation summary** (`final_report`, also refreshed by every report): per policy the success rate, the steps of
successful episodes, where failures end and the failure flags; per retrained policy the paired comparison with its
MoE and with the reference's policy (scenes only one of the two solved, two-sided exact sign test); per setting
(scorer, variant, take-over and hand-back threshold) the retrained policies' success. `results/eval_summary.json`,
`results/eval_episodes.csv`, `results/figures/retraining.png`.

## Results

Everything is written to `<work_dir>/results`, no account needed. `results/REPORT.md` collects it in one file:

- **Progress**, and how many collections are finished.
- **The figure** `results/figures/collections.png`: four panels as horizontal bars, one row per hand-back threshold
  and, below a dashed line, the reference: success rate per attempt; expert share of the DAgger data; expert share of
  all attempts; expert share of the failed attempts (with how many failed and their expert + move steps each). The
  random move's share is stacked in orange on the expert's (blue), and labelled as removed before retraining.
- **Reference collections:** which reference is shown, or why it is not (`reference.py`).
- **Per setting** (scorer × variant × take-over × hand-back threshold): the numbers as mean ± std over the seeds (one
  seed here), successes / attempts, and the expert + move steps per failed attempt.
- **Every collection** (and the reference): its two threshold values, attempts, successes, success rate, the expert
  and random-move shares of its data, of all attempts and of its failed attempts (pooled / per-episode median),
  takeovers, random moves (skipped), failure reasons.
- **Random moves** (`results/figures/moves/`): for every collection's thresholds, two figures of its moves, seen from the third-person camera. Each move in a collection's
  DAgger data is drawn as the outline of the 5 cm sphere its target was drawn from (centred on the gripper tip at the
  take-over), a dot at the centre and an arrow to where the tip ended, on the first frame of scene 1,000,000
  (grayscale). `spheres_held_sd<s>_p<q>_r<h>.png` colours them by whether the plug was held at the take-over,
  `spheres_progress_sd<s>_p<q>_r<h>.png` by when in the episode the take-over happened. The same plotting as the earlier
  maniskill random-move figures (it reproduces them from that run's moves).
- **Retrained policies** (`results/figures/retraining.png`, titled as trained with the random-move steps removed):
  success of the MoE, of each retrained policy and of the reference's, with Wilson 95 % intervals and the paired sign
  tests in the footnote; the per-setting table (only this / only MoE, p) and every retrained policy's success next to
  the MoE's and against the reference's policy.
- **Thresholds** of each MoE, and the **trainings** (updates, losses; `figures/training_moe.png`,
  `figures/training_retrain.png`).

Only finished collections enter the per-setting means; a collection analyzed by hand mid-run is listed as running.

| file | content |
|---|---|
| `results/summary.json`, `summary.csv` | the tables above: one row per collection, one per setting, one per reference |
| `results/collections/<collection>.json` | every metric of one collection |
| `results/collections/<collection>_episodes.csv`, `_takeovers.csv` | one row per attempt, per takeover |
| `results/collections/<collection>_scores.npz`, `_steps.npz` | the scores on policy- and expert-driven steps; every attempt's stage, driver, score and flags per step |
| `results/collections/<collection>_page.json.gz` | the data of an interactive page of the collection (format in `src/vig_dagger/page_data.py`) |
| `results/eval_summary.json`, `eval_episodes.csv` | every evaluation number; one row per evaluation episode |
| `results/collections/<collection>_moves.json`, `_background.png` | random-move collections: every move (start, target, end, plug held, take-over step, episode), the third-person camera's parameters, and the background frame |
| `policies/moe/sd<s>/metrics.jsonl` | each training's log: `train/loss`, `train/lr`, `val/loss` per `train/step` |

With `wandb.enabled: true` each collection's metrics also go to a W&B run `<name>/dagger/<collection>` (team
`wandb.entity`, project `wandb.project`), and trainings log their curves there.

## Several machines

**With a task board (recommended; the machines need to reach one another).** Every machine runs the same config
and the same command, whatever the number of machines; they divide the work among themselves (`src/vig_dagger/board.py`).

- **The board** is a directory, `sharing.board` in the config: `host:/path` on one of the machines, which the others
  reach with `ssh` and `rsync` (passwordless: an SSH key), or a path on a filesystem every machine mounts. Set the same
  value on every machine (the machine that hosts it may give the plain path). The first machine to start creates it.
- **Claiming.** The work is the 40 numbered tasks of [TASKS.md](TASKS.md). A machine with room claims the most urgent
  task whose inputs are on the board (a claim is an `mkdir` on the board, which succeeds for one machine only),
  fetches those inputs, runs the task in its own queue, and publishes its outputs to the board. A machine with
  nothing to claim waits; every machine's queue ends when the summary is done.
- **What moves:** the MoE checkpoints (weights only, ~180 MB) and thresholds, and every collection's metrics (a few
  MB). A collection's DAgger episodes (`dagger/<collection>/trajectories/`, ~15 GB) stay on the machine that collected
  them (`sharing.publish_trajectories: true` copies them to the board too).
- **Order.** The MoE first (downloaded in minutes, then its thresholds), then the collections in task order, then the
  summary. A machine runs as many collections at once as its CPU budget
  holds (`dagger.workers_per_collection` workers each), and claims no new collection with less than 25 GB free in
  its work directory.
- **Watching:** `vig-dagger status` on any machine lists every task (done, running and where, failed, ready, waiting)
  and every machine's last heartbeat. `vig-dagger report` on any machine copies every collection's metrics from the
  board and writes the report there; the summary task also leaves the final `results/` on the board
  (`<board>/files/results/REPORT.md`).
- **A machine that dies** keeps its tasks claimed (`status` shows it SILENT). Restarted with the same work directory,
  it takes them up again where they stopped. If it won't come back, `vig-dagger release <task>` frees each of its
  tasks for another machine (a collection then starts again from its first episode).
- Machines can join at any time, and leave once they have no running task (`vig-dagger stop`).

**Without a network between the machines**, the tasks can be assigned by hand instead:

- `vig-dagger run -c … --tasks 2 16-27` runs those tasks on this machine, with the preparation they need. It refuses
  if an input from another task is missing, and names it.
- `vig-dagger pack -c … --tasks 2 --for 16` writes one archive of what task 16 needs of task 2's outputs;
  `vig-dagger unpack -c … <that file>` puts it in place on the machine that runs task 16. For the summary:
  `pack --tasks 4-39 --for 40`.
- The tasks: 1–3 the MoEs, 4–39 the collections, 40 the collections' summary, 41–76 the retrainings (task N + 37
  retrains collection task N's data, on the machine that collected it), 77–79 the MoE evaluations (where task 1, 2, 3
  ran), 80 the final report (`pack … --for 80` from every other machine: metrics, evaluation episodes and training
  logs, under a MB per machine).

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
  data/datasets/{dagger,base_plus}_<collection>/   a retraining's data
  policies/retrain/<collection>/seed<s>/     a retrained policy (latest.pt, run_config.json, metrics.jsonl)
  eval/<policy>/level1/seeds_<first>.jsonl   one line per evaluation episode (policy: retrain/<c>/seed<s>, moe/sd<s>)
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
| `moe.seeds`, `factors` | `[0]`, 1 … 64 | the MoE seeds; their resolution levels |
| `moe.download` | `{sd0: …}` | seeds taken from a Hugging Face model repo instead of trained here |
| `scorers.vig.num_samples` | 32 | samples per level |
| `scorers.flow.num_loss_samples`, `num_action_samples` | 512, 1 | |
| `thresholds.num_demos`, `frame_stride`, `percentiles` | 50, 4, p50 … p99 | demo frames for the thresholds |
| `dagger.scorers`, `thresholds`, `variants` | `[vig]`, `[p90]`, `[random_move]` | the collections (with `moe.seeds` and `return_thresholds`); `thresholds`: take over above |
| `dagger.return_thresholds` | `[p70, p75, p80, p85]` | hand back below; one collection per take-over × hand-back threshold; `same`: the take-over threshold (one threshold both ways); never above the take-over threshold |
| `dagger.num_demos` | 500 | successful episodes per collection |
| `dagger.take_count`, `return_count` | 2, 16 | consecutive scores above the take-over / below the hand-back threshold needed to switch |
| `dagger.random_move.radius`, `rot_deg`, `max_tries`, `margin`, `tol` | 0.05, 5, 10000, 0.004, 0.0005 | the random move (m, deg) |
| `dagger.expert` | `demo` | `demo` (planner-demo motion) or `fast` (the earlier, faster expert) |
| `dagger.max_episode_steps`, `policy_chunk_steps`, `seed_start` | 400, 8, 1e6 | |
| `dagger.workers_per_collection`, `max_attempts` | 4, 20 × 500 | worker w runs scenes `seed_start` + w, + 4, …: with 4, the reference's scene sequence; `auto`: half the CPU budget |
| `dagger.save_failed_trajectories` | true | also keep failed attempts' images (as the reference did) |
| `retrain.seeds` | `[0]` | training seeds of the dense policy retrained on each collection |
| `retrain.cut_random_moves`, `min_piece_length` | true, 16 | cut the random-move steps (and the 15 before each) out of the retraining data; drop shorter pieces |
| `reference.dirs` | `[sd0_vig_p90_random_move]` | reference collections under `reference/`, shown when they match (`reference.py`) |
| `evaluation.seed_start`, `num_episodes`, `max_episode_steps`, `chunk`, `evaluate_moes` | 60000, 200, 400, 25, true | the evaluation scenes; episodes per job; also evaluate the MoEs |
| `sharing.board` | null | several machines: the task board, `host:/path` or a shared directory |
| `sharing.poll_s`, `heartbeat_s`, `publish_trajectories` | 60, 600, false | how often a machine reads the board and sends a heartbeat; whether DAgger episodes are copied to the board too |
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
| DAgger | per step ~0.15–0.35 s (VIG on a shared GPU); ~1.5 min per attempt; ~30 MB per saved episode | per collection ~600 attempts (the reference: 622 for 500 successes), roughly 15–25 worker-hours and ~15 GB; with 4 workers about 4–6 h |
| in total | | 4 collections, all at once with 4 workers each: about half a day |
| one retraining (~2,500 training demos, ~250–320k updates) | 5.5 updates/s on a busy GPU | 8–24 h on one GPU; 4 of them, one per GPU at a time |
| evaluation | ~10–25 s per episode | 5 policies × 200 = 1,000 episodes ≈ 5 CPU-hours |

Disk in total, about 120 GB on one machine:
- the 4 collections, about 80 GB (~30 MB per saved episode; failed attempts are saved too, as in the reference,
  about 5 GB per collection);
- the downloaded demos, 24 GB; the converted dataset, ~1.2 GB; the checkpoints, ~2 GB;
- per retraining ~2 GB (its training set, which shares the base demos' files, and its checkpoints);
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
- **`vig-dagger run` says "cannot use the board"**: this machine can't run commands on the board's host or copy files
  there. Check `ssh -o BatchMode=yes <host> true` (it needs an SSH key, no password prompt) and that `rsync` is
  installed on both machines.
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
UPDATES.md                      what a machine that started before a `git pull` has to do after it
src/vig_dagger/
  cli.py                        the vig-dagger command
  plan.py                       the experiment's job graph
  tasks.py                      numbered tasks, run --tasks, pack / unpack
  board.py                      several machines: the task board (claims, publish / fetch, heartbeats)
  jobs.py                       the job queue (dependencies, resources, retries, adoption)
  steps.py                      short steps: downloads, conversion, thresholds, collection settings, analysis
  config.py                     defaults, the collections, and the output layout
  envs.py                       the PlugCharger environment
  expert/planner.py             the motion-planning expert (demos and DAgger takeovers)
  expert/random_move.py         the random end-effector move at a takeover (variant random_move)
  expert/interruptible_planner.py  stops the planner mid-motion when control returns to the policy
  policy.py                     running checkpoints
  scoring.py, gate.py           VIG / flow loss, and the gate (take-over and hand-back thresholds)
  recording.py, stages.py       episode recording, task stages and failure flags
  dataset.py, train.py          the training dataset, run configs and the training worker
  analysis.py, dagger_stats.py  collection metrics
  evaluation.py                 the evaluation summary (retrained policies and MoEs)
  summary.py, report.py         every collection side by side, the figure, results/REPORT.md
  reference.py                  reference collections (reference/): loaded, and checked against this experiment
  move_figures.py               the random moves seen from the third-person camera
  page_data.py                  the data of an interactive page per collection
  tracking.py, hub.py           optional W&B logging, and Hugging Face
  workers/                      collect, convert, score_demos, dagger, evaluate (one process each)
  configs/policy.json           the policy recipe
src/robopolicy/                 the robopolicy training library
tests/robopolicy/               the library's tests
tests/vig_dagger/               this experiment's tests (gate, collections, tasks, random move, metrics, figures)
```

`src/robopolicy` is a copy of the robopolicy library (`refactor/policy-core`, branch `moe`) with two small additions:
W&B `group`/`tags` (`training/logging.py`) and `training.keep_checkpoints` (rolling checkpoint pruning,
`training/trainer.py`).
