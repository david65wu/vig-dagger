# VIG DAgger: PlugCharger, DAgger iteration with an adaptive threshold

DAgger on **PlugCharger** (ManiSkill: pick up a charger and insert its two pins into a socket that moves from
episode to episode), starting from **50 planner demos** and improving the policy in **4 rounds**. The **VIG** score
(visual information gain: how much the policy's plan changes when its camera images are degraded) decides when the
expert takes over from the policy and when it hands control back.

Each round:

1. trains a **7-level resolution mixture-of-experts (MoE)**, a flow-matching Transformer policy (the `robopolicy`
   library, included), on the 50 base demos plus the DAgger data of the earlier rounds (random-move steps cut);
2. scores every 4th frame of all its training demos with VIG: the round's **reference**;
3. collects DAgger episodes with that MoE driving: **100, 200, 400, 800** successful episodes in rounds 1–4 (1500 in
   all), with a random 5 cm / 5° gripper move at every takeover;
4. retrains a **dense full-resolution policy** on the base demos plus every round's DAgger data so far (random-move
   steps cut, 60 epochs) and evaluates it on 200 scenes next to the round's MoE;
5. as the comparison, the same dense policy on **as many planner demos** (150, 350, 750, 1550: the base demos plus
   more planner demos collected like them), evaluated on the same scenes: on another machine
   (`configs/planner_reference.yaml`, [Planner-only reference](#planner-only-reference)), or after each round here
   with `iterate.planner_baselines: true`.

**The threshold adapts.** One threshold serves both ways (the expert takes over after 2 VIG scores above it, the
policy takes back control after 16 below). After every attempt, successful or failed, it moves so that the VIG of the
steps the policy drives matches the VIG of the MoE's own training frames. It is kept as a percentile p of the
round's reference:

    p <- clip(p + (n / N) * (q(M) - q(m)), 0, 100)

with m the mean VIG of the attempt's n policy steps (the 2 scores that trigger a take-over left out: they are above
the threshold by construction), N the policy steps of all attempts so far, M the reference's mean and q(x) the
percentile of x in the reference. Nothing is tuned: in percentile units the policy steps' mean moves about one
percentile per percentile of threshold, so the gain is 1, and each attempt's step is its share of all policy steps so
far. The target q(M) is recomputed for every round's MoE. Round 1 starts at **p80**; every later round starts where
the previous round ended. See [Iteration mode](#iteration-mode).

**The expert** is the fast expert of the earlier maniskill collections (`dagger.expert: fast_maniskill`): transit
speed 1.5, insert 1.0, no refine steps, faster acceleration before the grasp, no in-place correction near the socket,
the random move at 1.5. At each takeover it draws a fresh strategy (grasp, standoff, lift, approach, planner) with the
planner demos' own sampler. **The 50 base demos are collected here** by the planner moving like that expert
(`demos.motion: fast_maniskill`), so the demos and the expert's corrections move alike; every demo keeps the planner's
30-step hold with the plug in the socket at the end. Every attempt is saved, failed ones included.

`vig-dagger run` turns the whole experiment into one queue of jobs. A job starts when its inputs exist and the
machine has room for it (CPU, RAM, GPU memory), so independent work runs in parallel (round r's retraining next to
round r+1's MoE and collection). Rerunning the same command after an interruption skips finished jobs and picks up
running ones. The experiment runs on one machine (the rounds follow each other); the work is also numbered as tasks
([TASKS.md](TASKS.md)).

The sections from [Part 1](#part-1-demos) on describe the pipeline's pieces as the earlier experiments on other
branches used them (a downloaded 1000-demo MoE, fixed thresholds, a sweep of collections); iteration mode chains the
same pieces per round.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The task](#the-task)
- [The job queue](#the-job-queue)
- [Iteration mode](#iteration-mode)
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
2. Check that the machine is ready: GPUs, rendering (Vulkan on the NVIDIA GPU; in a container
   `NVIDIA_DRIVER_CAPABILITIES=all`), `ffmpeg`, disk. Every line must say `ok`:

   ```bash
   vig-dagger check -c configs/plugcharger.yaml
   ```

3. Start the queue, detached so it survives a closed SSH session:

   ```bash
   setsid nohup vig-dagger run -c configs/plugcharger.yaml > plugcharger.log 2>&1 < /dev/null &
   ```

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
7. Results: `<work_dir>/results/REPORT.md`, `results/figures/collections.png`, `retraining.png`, `thresholds.png`,
   `rounds.png`, `distributions.png` and `moves/spheres_{held,progress}_rounds.png`
   (see [Iteration mode](#iteration-mode)). Every collection's analysis refreshes them; `vig-dagger report -c
   configs/plugcharger.yaml` refreshes them at any time. Videos of some attempts of each round:
   `<work_dir>/videos/<collection>/` (`vig-dagger videos -c configs/plugcharger.yaml` renders them during a
   collection too).

Useful options:

- `--set key=value` overrides any config value, e.g. `--set resources.sim_slots=16`.
- `--only REGEX` runs only matching jobs and what they need, e.g. `--only '^dagger_round1_'`.
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

## Iteration mode

On when `iterate.rounds` is set (this branch's `configs/plugcharger.yaml`). The jobs, per round r = 1 … R:

```
collect_planner_<k> -> convert_planner                              the base demos (round 1's training set)
round_dataset_<r>          r > 1: round r-1's training set + round r-1's DAgger data, cut at its random moves
train/moe/round<r>         the round's MoE (training seed moe.seeds[0]) -> policies/moe/round<r>/
score_round<r>_<chunk>     VIG of every thresholds.frame_stride-th frame of every training demo -> thresholds_round<r>
                           (thresholds.json with the target, demo_scores.npz = the reference)
dagger_prepare_<c>         settings.json: the reference, the target q(M), the start percentile (round 1:
                           iterate.start_percentile; later: where round r-1's collection ended), seed block
dagger_<c>_<w>             the workers; they share the threshold (threshold_state.json, under a file lock)
analyze_<c>                the collection's metrics
videos_<c>                 videos of videos.per_kind successful and failed attempts -> videos/<c>/ (nothing waits for it)
round_dataset_<r+1> -> train/retrain/<c>/seed0 -> eval/retrain/<c>/seed0/<first scene>   the dense policy after round r
eval/moe/round<r>/<first scene>                                                          the round's MoE
collect_baseline_<k> -> planner_dataset_<r> -> train/planner/round<r>/seed0 -> eval/planner/round<r>/seed0/<first scene>
                           the planner-only comparison for round r (iterate.planner_baselines): the extra planner
                           demos it needs (two parts collecting at a time), planner_<N> = the first N planner demos with
                           N what round r's dense policy trains on, a dense policy on them (after round r's dense
                           policy, so it never holds the GPU while a round's MoE waits), its evaluation
summarize, final_report    results/REPORT.md (a table per round), iterate_summary.json, figures
```

with `<c>` = `round<r>_sd0_vig_adaptive_random_move`. Round r+1's MoE and collection outrank round r's retraining and
evaluations in the queue.

**The reference** is the round MoE's VIG on every 4th frame of all its training demos: the base demos and the
earlier rounds' DAgger episodes, each cut at its random moves exactly like the training set
(`dataset.training_pieces`), scored on the raw renders of the recorded trajectories (as DAgger scores them, not on the
dataset's compressed videos), with torch reseeded per frame. M is its mean, q(M) the target.

**The update** (`src/vig_dagger/adaptive.py`, `workers/dagger.py`): before each attempt a worker reads p and sets
both thresholds to the reference's p-th percentile; during the attempt it keeps every score computed while the policy
drives, minus the 2 that trigger each take-over (`iterate.exclude_trigger`); after the attempt it applies
`p <- clip(p + n/N (q(M) - q(m)), 0, 100)` to the shared state under the lock and appends a line to
`threshold_log.jsonl` (threshold used, n, m, q(m), p before and after). An attempt without policy steps leaves p
unchanged. Several workers update the same state, each from the p it read at its attempt's start. The episode line
(`episodes/worker_<w>.jsonl`) also records the threshold and the update, and the collection's analysis uses each
attempt's own threshold.

Why the trigger scores are left out: below about p65 the expert takes over within a few steps, so the trigger
scores are a large share of the few policy steps, and their mean happens to equal the training mean: a false
equilibrium near p50 (seen in a maniskill run of the earlier VIG-unit rule, which fell to p50 after one long failed
attempt and stayed there). Without them the policy steps' mean rises steadily with the threshold.

**Failed attempts** (`iterate.failed_last_stretch`, on in `configs/plugcharger.yaml`): in a failed attempt, the policy
steps after its last take-over (the policy drove into the failure without the gate firing; all its policy steps if the
expert never took over) count as if their VIG were at the threshold the attempt ran with: steps the expert should have
taken. Successful attempts, a failed attempt's earlier policy steps, and a failed attempt that ends with the expert
driving count as without it. Why: a MoE on few demos fails in states that look like its training frames, so without
this its failed attempts push p up as well, and round 1 of a 50-demo run climbed to about p98, where 4 % of the
attempts succeeded. At 1000 demos, where failures are rare, it barely matters (estimated p88 instead of p89–90).
`threshold_log.jsonl` records `n_last_stretch` and the raw mean of those scores.

**Every attempt counts the same** (`iterate.update_weight: attempts`, on in `configs/plugcharger.yaml`; the default
`policy_steps` is the n/N step above): `p <- clip(p + (q(M) - q(m)) / (u + 2), 0, 100)`, u = the collection's earlier
updates, so p is the average of the start and of each attempt's p + q(M) − q(m), and the first attempt moves p half
way. Why: with n/N, round 1 of the 50-demo run (with failed attempts counted) jumped from p80 to p100 on its first
attempt (a success whose policy steps were easy), then a few 400-step failures at p100, where the expert cannot take
over, pulled it to p35; there the expert takes over within a few steps, an attempt has ~20 policy steps against
~2,300 so far, and p crept to p41 over 100 attempts: 88 % of that round's data was the expert's.

**Scenes:** the base demos use scene seeds from 2,000,000; round r's collection from `dagger.seed_start` +
(r − 1) × `iterate.round_seed_stride` (1,000,000, 1,100,000, …), so no two rounds share scenes; evaluation 60000–60199.

**The planner-only comparison** (`iterate.planner_baselines`: `true` here after each round, `only` on its own
machine, see below): the dense policy after round r trains on N = 50 + the
DAgger episodes of rounds 1 … r (150, 350, 750, 1550). Its comparison trains the same way (dense, full resolution,
60 epochs, training seed 0) on the first N planner demos: the 50 base demos, then demos collected exactly like them
(same planner, `demos.motion`, part k from scene seed 2,000,000 + 1000 k with strategy seed 1 + k; parts 1–30 in
`data/raw/planner_baselines/`), so the four sets are nested like the DAgger data. Demos are counted as episodes; the
DAgger episodes are longer than planner demos and lose their random-move steps, so the two sides differ in training
samples (`run_config.json` → `_info.training_samples` has both).

**Results** (`results/REPORT.md`, `results/iterate_summary.json`):
- per round: the MoE's training demos, the reference (frames, M, q(M)), the threshold's start and end (and its median
  over the collection's second half), attempts, successes, the expert and random-move shares of the DAgger data;
- per round: the dense policy's and the MoE's success on the 200 scenes, the dense policy against the previous
  round's and against the MoE (paired, exact sign test);
- per round: the planner-only dense policy's success, and the DAgger dense policy against it (paired);
- `figures/collections.png`, one row per round's collection, as bars: success rate per attempt; expert share of the
  DAgger data, of all attempts and of the failed attempts (random-move steps stacked in orange; failed attempts:
  how many, steps each); takeovers per kept episode (± std); each row names the round's MoE and where its threshold
  started and ended;
- `figures/retraining.png`, per round, as bars with Wilson 95 % intervals: the dense policy retrained after the round,
  the dense policy on as many planner demos (when its evaluation is in `eval/planner/`, trained here or copied from
  the machine that ran `configs/planner_reference.yaml`) and the round's MoE; paired sign tests in the footnote;
- `figures/thresholds.png`: the threshold attempt by attempt, all rounds; `figures/rounds.png`: the same numbers as
  lines over the rounds;
- `figures/moves/spheres_{held,progress}_rounds.png`: every random move of each round's DAgger data on the
  third-person camera, one panel per round (with where its threshold started and ended): the 5 cm sphere the target
  was drawn from, its centre and an arrow to where the gripper tip went; colour = plug held at the take-over, or the
  take-over's step / episode length (`move_figures.round_figures`);
- `figures/distributions.png`, one panel per round: VIG histograms of the round MoE's training frames (the reference),
  of its collection's policy steps (every attempt, successful and failed, as the update counts them: trigger scores
  left out, raw scores), of its policy + expert steps and of its expert steps alone (dash-dot), with the overlap of the policy-step and training histograms, the
  training mean, the policy steps' mean and the threshold now; the numbers are in `iterate_summary.json` under each
  round's `distribution`.

**Videos** (`videos_<c>`, `src/vig_dagger/videos.py`): `videos.per_kind` (20) successful and as many failed attempts
of each collection, spread evenly over the attempts in the order they finished (first and last included), one mp4
each at real time: the three cameras on top; below, VIG against the step while the policy drives (solid) and while
the expert drives (dashed), the threshold that attempt ran with (and its percentile), expert control shaded grey,
random moves orange, gripper closes, a cursor and a status line. Files `videos/<c>/{success,failed}/attempt<k>_seed<s>.mp4`
(k: the attempt's place in the finishing order) and `videos/<c>/index.json`. `vig-dagger videos -c <config>
[--collection <c>]` renders them any time, also during a collection (videos already there are kept).

**Cost** (rough, one GPU): base demos minutes; round 1–2 MoEs under an hour each, round 3 about an hour or two, round
4 (750 demos, ~127,000 updates) several hours; collections ~1–1.5 min per attempt per worker (4 workers), so round 4's
800 successes take several hours; the last dense policy (1550 demos, ~270,000 updates) 5–20 h depending on the GPU;
evaluations 8 × 200 episodes. About 2–3 days on one dedicated GPU; disk about 100 GB (every attempt saved). The
planner comparison, if run here (`planner_baselines: true`), adds 1500 planner demos (CPU, a few hours alongside the
rounds), 4 dense trainings (150 … 1550 demos; the last as long as the last DAgger one) and 4 × 200 evaluation
episodes: about a day and ~20 GB more.

### Planner-only reference

`configs/planner_reference.yaml` runs only the comparison (`iterate.planner_baselines: only`), on its own machine:
the base demos are the DAgger machine's (copy its `data/raw/planner/part_000.{h5,json,done.json}`, e.g. from the
Hugging Face dataset they were uploaded to, into `<work_dir>/data/raw/planner/` before the first run; the queue then
does not collect them again, and the planner sets refuse a part collected with other settings). The queue collects the
1500 more planner demos (`iterate.baseline_lanes` = 8 parts at a time), builds `planner_150` … `planner_1550`, trains
the 4 dense policies (each as soon as its set exists) and, as round 0, a dense policy on the 50 base demos alone
(`planner/round0`, where the DAgger and the planner-only series both start), and evaluates them on the same 200
scenes; REPORT.md lists their success. Every other value equals `configs/plugcharger.yaml`'s (a test checks this), so
the policies match the DAgger machine's recipe; to compare, copy this machine's `eval/planner/` folder into the DAgger
machine's work directory and run `vig-dagger report` there: `retraining.png` and `rounds.png` then show the
planner-only policies and round 0, and round 1's dense policy is tested against round 0. About a day with two GPUs;
~25 GB.

```
vig-dagger run -c configs/planner_reference.yaml --set work_dir=<dir>
```

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

**The expert** (`resume_and_solve`, `dagger.expert: fast_slowed`) runs the planner demos' stage sequence with a
fresh strategy at each takeover, drawn by the planner demos' own sampler. `dagger_prepare` checks the sampler before a
collection starts: it regenerates every planner demo's recorded strategy (all 16 fields) from its seed and refuses to
start if any differs. It is the expert of the earlier maniskill DAgger collections (`plug_charger_diverse_solver.py`),
slowed down:

- **Speeds:** fixed, transit 0.7 and insert 0.625 (the planner demos sample 0.5–0.9 and 0.4–0.85 per demo; the
  earlier fast expert ran at 1.5 and 1.0). Acceleration is tied to speed, doubled before the grasp (nothing held yet).
- **No stationary steps:** it re-asserts the grasp without stepping and skips the demos' 5 refine steps after aligning.

It never resets or rewinds the simulation; it starts from the current state:

- **Plug not held:** reach, grasp and lift. It skips the reach detour if the gripper is already over the charger.
- **Plug held:** the strategy's approach pose (off the socket axis by up to 15 mm and 8°, 3–8 cm in front of the
  socket), alignment onto the axis, and the straight insertion. This also holds when the plug is already near the
  socket: it pulls back to the approach pose first.

`dagger.expert: demo` (the earlier sweep on this branch) moves at the strategy's sampled speeds with the 5 refine
steps, does the demo's lift first when taking over right after a grasp, and corrects in place near the socket
(aligns at the plug's current distance instead of pulling back); `fast` is that with the fast expert's speeds.

**The random move** (`src/vig_dagger/expert/random_move.py`, `dagger.random_move`; the same code as the earlier
PlugCharger experiments in maniskill):

- **Target:** a position uniform inside a 5 cm ball around the gripper tip, the orientation turned by 0–5°
  (uniform) about a uniformly random axis.
- **Gripper:** held at the policy's last command, so a held charger moves with the hand.
- **Collision-free:** a target is redrawn (up to 10,000 times) if the screw plan fails (IK, joint limits) or if the
  hand, fingers or held charger would get deeper into the socket walls, the table or the free charger (grown by
  4 mm) than they already are. A plug with its pins in the holes can therefore only be pulled straight out. If no
  target is valid, the takeover has no move (logged as skipped).
- **Speed:** a screw motion at speed 1.5, as in the earlier maniskill collections (`dagger.expert: demo`: the transit
  speed of the strategy the expert drew for this takeover).
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
  also saved in full to `trajectories_failed/` (`dagger.save_failed_trajectories: true`).
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
compared with it scene by scene (and with a reference's retrained policy, when one is configured).

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
  policies/moe/round<r>/                     iteration mode: round r's MoE (thresholds.json has the target q(M),
                                             demo_scores.npz is the reference)
  data/datasets/round<r>_train/              iteration mode: what round r's MoE (and round r-1's dense policy) trains on
  data/raw/planner_baselines/                iteration mode: the planner demos beyond the base demos (part_<k>.h5)
  data/datasets/planner_<N>/                 iteration mode: the first N planner demos (a planner-only comparison)
  policies/planner/round<r>/seed<s>/         iteration mode: the dense policy on as many planner demos as round r's
  dagger/<collection>/                       settings.json, trajectories/, attempts/, episodes/ (iteration mode also
                                             threshold_state.json, threshold_log.jsonl)
  data/datasets/{dagger,base_plus}_<collection>/   a retraining's data
  policies/retrain/<collection>/seed<s>/     a retrained policy (latest.pt, run_config.json, metrics.jsonl)
  eval/<policy>/level1/seeds_<first>.jsonl   one line per evaluation episode (policy: retrain/<c>/seed<s>, moe/sd<s>)
  results/                                   REPORT.md, summary.*, collections/, figures/
  videos/<collection>/                       iteration mode: videos of some attempts ({success,failed}/*.mp4, index.json)
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
| `dagger.expert` | `fast_slowed` | `fast_slowed` (the earlier maniskill expert at speeds 0.7 / 0.625), `demo` (planner-demo motion) or `fast` (speeds 1.5 / 1.0); see [Part 3](#part-3-dagger) |
| `dagger.max_episode_steps`, `policy_chunk_steps`, `seed_start` | 400, 8, 1e6 | |
| `dagger.workers_per_collection`, `max_attempts` | 4, 20 × 500 | worker w runs scenes `seed_start` + w, + 4, …: with 4, every collection the same scene sequence; `auto`: half the CPU budget |
| `dagger.save_failed_trajectories` | true | also keep failed attempts' images |
| `retrain.seeds` | `[0]` | training seeds of the dense policy retrained on each collection |
| `retrain.cut_random_moves`, `min_piece_length` | true, 16 | cut the random-move steps (and the 15 before each) out of the retraining data; drop shorter pieces |
| `reference.dirs` | `[]` | reference collections under `reference/`, shown when they match (`reference.py`); none here |
| `evaluation.seed_start`, `num_episodes`, `max_episode_steps`, `chunk`, `evaluate_moes` | 60000, 200, 400, 25, true | the evaluation scenes; episodes per job; also evaluate the MoEs |
| `iterate.update_weight` | `policy_steps` (`attempts` in `configs/plugcharger.yaml`) | how much each attempt moves the threshold: by its policy steps (n/N), or every attempt the same with the start counting as one (1/(u+2)) |
| `iterate.failed_last_stretch` | false (true in `configs/plugcharger.yaml`) | a failed attempt's policy steps after its last take-over count at the threshold in the update (see Iteration mode) |
| `iterate.planner_baselines`, `baseline_lanes` | false, 2 | true: after each round also a dense policy on as many planner demos, evaluated on the same scenes; only: just those (`configs/planner_reference.yaml`); parts of the extra planner demos collected at once |
| `videos.per_kind`, `workers` | 20, 4 | videos per collection of successful and of failed attempts (0: no video jobs); rendering processes. Not an experiment parameter |
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
| DAgger | per step ~0.15–0.35 s (VIG on a shared GPU); ~1.5 min per attempt; ~30 MB per saved episode | per collection ~600 attempts (hand-back at p90 with this expert: 600 for 500 successes), roughly 15–25 worker-hours and ~15 GB; with 4 workers about 4–6 h |
| in total | | 4 collections, all at once with 4 workers each: about half a day |
| one retraining (~2,500 training demos, ~250–320k updates) | 5.5 updates/s on a busy GPU | 8–24 h on one GPU; 4 of them, one per GPU at a time |
| evaluation | ~10–25 s per episode | 5 policies × 200 = 1,000 episodes ≈ 5 CPU-hours |

Disk in total, about 120 GB on one machine:
- the 4 collections, about 80 GB (~30 MB per saved episode; failed attempts are saved too,
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
