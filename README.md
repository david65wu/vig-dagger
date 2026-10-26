# VIG DAgger: OpenDrawer

DAgger experiments that compare two query scores deciding when the expert takes over from the policy:

- **VIG** (visual information gain): how much the policy's plan changes when its camera images are degraded.
- **Flow loss**: the policy's flow-matching loss on its own sampled action chunk.

This branch (`opendrawer`) runs the **OpenDrawer** task in Isaac Lab: a Franka Panda opens the top drawer of a
cabinet, puts a charger into it, and closes the drawer. A motion planner provides the demos and plays the expert
during DAgger. The policies are flow-matching Transformer policies trained with the `robopolicy` library, which is
included in this repository. (The PlugCharger experiment, in ManiSkill, is on the `main` branch.)

The experiment has two **rounds**, which differ in the policy that drives the DAgger rollouts:

| round | drives DAgger, gives the flow loss | VIG compares |
|---|---|---|
| `moe` | a 7-level resolution mixture-of-experts (MoE) policy, at full resolution | the MoE at full resolution vs the MoE at each degraded level (1/2 … 1/64) |
| `per_resolution` | a dense full-resolution policy | that policy vs 6 dense policies trained on images degraded by 2 … 64 |

Each round has the same four parts:

1. **Demos.** 100 planner demos, downloaded from Hugging Face (made with this repository's environment and
   planner); shared by both rounds.
2. **Driver and thresholds.** The round's driving policy or policies (trained here), and VIG and flow-loss
   percentiles on demo frames.
3. **DAgger.** 100 successful episodes with VIG as the query score and 100 with flow loss. A take-over threshold
   and a separate return threshold decide who drives. The collections are analyzed.
4. **Retraining.** Dense full-resolution policies are trained on the base demos plus the DAgger demos, three seeds
   each, and evaluated on the same 200 scenes.

Two baselines are shared by both rounds and trained once: the base demos alone, and the base demos plus 100 more
planner demos. Every training runs 150 epochs.

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
- [Results](#results)
- [Hugging Face](#hugging-face)
- [Output directory](#output-directory)
- [Configuration reference](#configuration-reference)
- [Cost estimates](#cost-estimates)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

---

## Installation

Requirements:

- **Linux x86_64** with glibc 2.35 or newer (Ubuntu 22.04 or 24.04).
- **NVIDIA RTX-class GPUs.** Isaac Sim renders the cameras with RTX ray tracing, so the GPU needs RT cores:
  data-center GPUs without them (A100, H100) cannot run the simulator. Tested on RTX PRO 6000 Blackwell with driver
  580.178. The driver must support CUDA 12.8.
- `ffmpeg` on the `PATH` (training data is stored as H.264 video), `git`, and
  [`uv`](https://docs.astral.sh/uv/) (Python package manager; it installs Python 3.11 itself).
- Internet access, also while the experiment runs: every simulator job loads the robot, table and cabinet models
  from NVIDIA's asset server (only the charger model is in this repository).
- Disk: about 30 GB for the two Python environments (inside the repository) and about 120 GB for the work directory;
  see [cost estimates](#cost-estimates).
- Nothing else: every result is written to files in the work directory (`results/REPORT.md`, JSON, CSV,
  figures). Logging to W&B as well is optional (`wandb.enabled`).

The code runs in **two Python environments**, and the queue picks the right one for each job:

| environment | installed by | used for |
|---|---|---|
| `.venv` | `uv sync` | the `vig-dagger` command, conversion, training (PyTorch 2.9.1), threshold scoring, analysis |
| `.venv-isaac` | `bash scripts/install_isaac.sh` | the simulator jobs: DAgger, evaluation (and demo collection with `demos.source: collect`); Isaac Sim 5.1, Isaac Lab v2.3.2, PyTorch 2.7.0 |

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout opendrawer
uv sync                          # .venv with the pinned packages (uv.lock)
bash scripts/install_isaac.sh    # .venv-isaac and third_party/IsaacLab; 10-30 minutes, mostly downloads
source .venv/bin/activate        # or prefix commands with `uv run`
vig-dagger --help
```

**NVIDIA's license.** `scripts/install_isaac.sh` installs Isaac Sim from NVIDIA's package index, and every simulator
job runs with `OMNI_KIT_ACCEPT_EULA=YES`, i.e. it accepts the
[NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
without asking. Whoever installs and runs the pipeline should read it first.

The first training downloads the DINOv2-small weights from Hugging Face (`facebook/dinov2-small`, about 90 MB). The
first Isaac Sim start on a machine compiles shaders, which can take several minutes; later starts take about 10 s.

**Check the installation** with a short run of every job of both rounds on tiny numbers (2 demos, policies trained
for a few dozen updates; the success rates it prints are meaningless; about 15 minutes):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

The library's own tests: `uv sync --extra dev && pytest tests -m "not cuda"`.

## Quick start

An agent operating the experiment without supervision should follow [AGENTS.md](AGENTS.md): the same steps,
plus what to check at each milestone, when to stop and ask, and what never to change.

1. In `configs/opendrawer.yaml`, set `work_dir` (the output directory). Everything else is set for the
   experiment; see the [configuration reference](#configuration-reference). With several machines, see
   [the job queue](#the-job-queue).
2. Check that the machine is ready: GPUs, the Isaac environment (it starts Isaac Sim once and renders the policy's
   cameras), `ffmpeg`, disk, Hugging Face access. Every line must say `ok`:

   ```bash
   vig-dagger check -c configs/opendrawer.yaml
   ```

3. Start the queue, detached so it survives a closed SSH session:

   ```bash
   setsid nohup vig-dagger run -c configs/opendrawer.yaml > opendrawer.log 2>&1 < /dev/null &
   ```

4. Watch it:

   ```bash
   vig-dagger status -c configs/opendrawer.yaml   # queue alive? failed / running jobs with progress; milestones
   tail -f opendrawer.log                         # the queue's own log: every job started, finished, retried
   ```

   Each job's output is in `<work_dir>/logs/<job>.log`.
5. After a crash, a reboot or a fix, run the same command again.
6. To stop: `vig-dagger stop -c configs/opendrawer.yaml`. Every job resumes where it stopped on the next `run`.
   With `--queue-only`, only the queue stops: the running jobs continue, and the next `run` takes them over, e.g. to
   change `resources` settings.
7. Results: `<work_dir>/results/REPORT.md` and `results/figures/` (see [Results](#results)). The DAgger analysis and
   the final summary refresh them; `vig-dagger report -c configs/opendrawer.yaml` refreshes them at any time.

Useful options:

- `--set key=value` overrides any config value, e.g. `--set dagger.num_demos=200`.
- `--only REGEX` runs only matching jobs and what they need, e.g. `--only 'eval/'` or `--only '^dagger_moe'`.
- `--dry-run` lists the jobs with their dependencies.
- `vig-dagger show-config -c …` prints the resolved configuration.
- `vig-dagger tasks -c …`, `run -c … --tasks N …`, `pack -c … --tasks N --for M`, `unpack -c … FILE`: several machines,
  tasks assigned by hand.

## The task

Environment `OpenDrawer-ManagerBased-v0`, an Isaac Lab manager-based environment included in this repository
(`src/vig_dagger/opendrawer_env/`). A Franka Panda stands in front of a cabinet with a charger on the table. The task
has four stages, each latched once reached:

1. **drawer opened:** the top drawer is pulled out at least 18 cm;
2. **charger picked:** the charger is lifted 4 cm;
3. **charger placed:** the charger lies inside the drawer;
4. **drawer closed:** the drawer is shut (open less than 1 cm) with the charger inside.

Each episode randomizes the cabinet's position (−5 to +13 cm forward, ±12 cm sideways) and the charger's position
(−12 to +8 cm, ±12 cm) and yaw (any).

| | |
|---|---|
| Robot / control | Franka Panda; relative inverse-kinematics control of the gripper's TCP: 7-dimensional actions (3 translation, 3 rotation, gripper), the spacemouse's action space in the original teleoperated demos; 60 Hz |
| Policy inputs | the TCP pose (position and quaternion, 7) and the 2 finger positions; RGB 256×256 from the fixed `left` camera and the `wrist` camera |
| Policy outputs | 16-step action chunks; 8 steps are executed before the next prediction |
| Success | the teleop recorder's rule: the env's success term (all four stages, drawer closed with the charger inside) holds for 10 steps in a row, and 14 more steps are recorded |
| Step limit | 1200 for planner demos, DAgger and evaluation |
| Physics | CPU (one environment per process); rendering on the GPU |

The scene's other three cameras are removed in every job (`src/vig_dagger/isaac.py`); this saves rendering time and
changes nothing else.

**The TCP pose input.** Isaac Lab keeps the hand's frames in a set whose order depends on Python's hash seed, so the
observation `ee_pose` is the TCP in some processes and the bare hand in others. Every simulator job therefore runs
with `PYTHONHASHSEED=0`, which makes it the TCP, and refuses to start otherwise.

**Scene seeds.** A seed fixes the cabinet and charger placement, so every policy is evaluated on the same scenes.
Isaac Lab draws the placement from torch's random generator, so a seed gives a different scene under a different
torch version; the Isaac environment pins torch 2.7.0, so seeds are consistent across machines that use it.

The seed ranges never overlap:

| Use | Seeds |
|---|---|
| Base demos (downloaded) | 0–174 |
| Extra planner demos (downloaded) | 200–377 |
| Demos collected by the pipeline (`demos.source: collect`) | 2,000,000 + 1000·*k* for part *k*, 3,000,000 + 1000·*k* for the extra parts |
| DAgger | from 1,000,000 |
| Evaluation | 60,000–60,199 |

## The job queue

`vig-dagger run` builds every job of the experiment (`src/vig_dagger/plan.py`) and hands them to a queue
(`src/vig_dagger/jobs.py`). Each job is one process with its own log.

**Simulator jobs** (DAgger workers, evaluations, demo collection) run in `.venv-isaac` with `PYTHONHASHSEED=0`,
`OMNI_KIT_ACCEPT_EULA=YES`, one OpenMP thread, and a temporary directory of their own under `<work_dir>/tmp/`.
Every other job runs in `.venv`.

**When a job starts.** Once the jobs it depends on are done, and the machine has room:

- **CPU:** simulator jobs, threshold scoring and conversions use a CPU budget, `resources.sim_slots` (default: one
  slot per 3 cores; an Isaac job uses about 1.5 cores). A training counts as `resources.train_slots` (its data
  loading).
- **Load:** nothing new starts while the 1-minute load average is above `resources.max_load` (default: the number
  of cores), so other users' work is respected.
- **RAM:** the available memory must cover the job's estimate.
- **GPU:**
  - a simulator job goes to the GPU with the most free memory, if it has at least `resources.sim_gpu_memory_gb`. It
    renders and runs its policy on that GPU only (`CUDA_VISIBLE_DEVICES`);
  - a training needs `resources.train_gpu_memory_gb` free on a GPU that runs none of our other trainings.

Jobs started in the last 3 minutes count with their estimates.

**Order.** Lower priority numbers go first: part 1, then round 1's driver and DAgger, round 1's retraining, the
shared baselines, round 2, and the evaluations.

**What runs in parallel**, if the machine has room:

- the shared baselines (`base`, `base_plus_planner`, 3 seeds each) train on the GPUs while the MoE trains and
  round 1 collects DAgger episodes;
- every evaluation runs as soon as its policy is trained.

On one machine without Hugging Face (`sharing.mode: none`), round 2's driving policies start only after all of
round 1's retraining.

**Several machines, tasks assigned by hand.** The experiment's work is a numbered list of 63 tasks,
[TASKS.md](TASKS.md) (`vig-dagger tasks`): 25 trainings, each round's thresholds, the four DAgger collections, 31
evaluations and the summary, each with the task numbers whose outputs it needs (always lower numbers) and the tasks
that use its outputs. `vig-dagger run --tasks 3 17 32-40` runs those tasks on this machine, with the preparation they
need; it refuses if another task's outputs are missing, and names them. Both rounds' tasks are independent of each
other. `vig-dagger pack --tasks 3 --for 15` writes one archive of what task 15 needs of task 3's outputs (checkpoints
weights-only), `vig-dagger unpack FILE` puts it in place on the other machine; how the archive travels is up to the
humans. The summary task, given every evaluation's episodes and every collection's metrics, writes the full report.
[AGENTS.md](AGENTS.md) has the steps.

**Several machines sharing through Hugging Face** (`sharing.mode: hub`, for machines that can upload). Every machine
runs the same config and the same `vig-dagger run`; however many machines there are (one works too), they share the run through one public
Hugging Face repo, `sharing.repo` (`board.py`). The work comes in 63 tasks: each training, each round's thresholds,
each DAgger collection, each evaluation of a policy at a level, and the summary. A machine with room claims the most
urgent task whose inputs are all in the repo (with a commit that fails if another machine committed first, so no
task runs twice), downloads the inputs (`fetch/<task>` jobs), runs the task's jobs in its own queue, and uploads the
outputs (`publish/<task>`: checkpoints weights-only, DAgger training sets and metrics, evaluation episodes). A step
that needs another's output (a round's thresholds, its DAgger collections, their retrainings) runs on one machine
while the others take other tasks or wait for the upload, and both rounds start at once. Machines send a heartbeat
every `sharing.heartbeat_s` seconds; `vig-dagger status` shows every task and machine, `vig-dagger release <task>`
frees a task whose machine is gone, and `vig-dagger report` builds the full report on any machine. With
`sharing.mode: none` one machine runs everything without Hugging Face, in the order above (round 2 after round 1's
retraining).

**Failures and restarts:**

- A job that fails is retried a few times (trainings up to `training.max_restarts` times). Every job continues
  where it stopped: trainings from their last checkpoint (every 5000 updates), DAgger workers from their last
  finished episode, evaluations from their last finished scene, demo collection from the start of its part.
- A job that still fails stops only the jobs that depend on it. The rest of the queue keeps going.
- If the queue process dies, the jobs it started keep running. A restarted queue adopts them instead of starting
  them twice. Only one queue runs per work directory.

## Part 1: demos

`download_demos` fetches from `hub.dataset_repo`
([chengenwu/OpenDrawer-ManagerBased-v0](https://huggingface.co/datasets/chengenwu/OpenDrawer-ManagerBased-v0)):

- `raw/planner/part_000 … 003.hdf5`: the 100 base demos (scene seeds 0–174);
- `raw/planner_extra/part_000 … 003.hdf5`: 100 more demos from the same planner and environment (seeds 200–377), for
  the planner baseline.

The files are Isaac Lab demo files (`data/demo_<i>`: actions, the observations before each step, sim states,
the stage latches), the format of the original teleoperated demos, with the scene seed and the planner's strategy as
attributes; each part has a `part_<k>.done.json` that lists its demos. They hold only the two policy cameras.

These demos were collected before this repository existed, with the same environment and planner (RoboticArmSim's
`collect_open_drawer_demos.py`, Isaac Sim 5.1). Replayed with this repository's code from their scene seed and
strategy, they reproduce bit for bit: the same actions, robot state and episode length at every step. They are the
first 200 of the 300 planner demos an earlier OpenDrawer MoE was trained on, in the same order.

With `demos.source: collect`, the pipeline collects new demos itself instead: `collect_planner_<k>` (4 jobs of 25
demos) and `collect_extra_<k>` (4 more) run the planner in Isaac Sim, from scene seeds 2,000,000 and 3,000,000; their
files also hold the per-step task state used for the stage labels.

**The planner** (`src/vig_dagger/expert/planner.py`) does the task the way the teleoperated demos do: grasp the
handle from the front, pull the drawer open, let go, grasp the charger from above across its short side, lift it
over the drawer's rim, lower it into the drawer and let go, then push the drawer shut with the fingers. It commands
the same relative-IK actions as the policy, clipped to the spacemouse's range, so planner actions stay inside the
policy's action distribution.

It samples a strategy for each demo from ranges measured on 100 teleoperated demos:

- handle approach tilt, yaw and roll, the grasp point on the bar, the pre-grasp standoff;
- how far the drawer is pulled, and how the hand retreats from the handle;
- the charger grasp point, tilt and yaw, the pre-grasp height, the lift detour, carry height, tilt and hand yaw;
- where in the drawer and from how high the charger is released;
- the push point, tilt and hand yaw;
- jittered midpoints on transits, speeds, gains and short pauses after gripper changes.

So no single path solves the task. A demo counts only if it is clean like the teleoperated demos: success within
1200 steps with exactly two grasps (handle, then charger) and no retries; the planner gives up on a missed or
slipped grasp instead of trying again.

**Converted datasets.** `convert_planner` turns the raw demos into the library's training format (H.264 videos of
the two policy cameras plus HDF5 with actions and the policy's state):

- `data/datasets/planner`: every demo;
- `data/datasets/base`: the first `demos.num_demos` (100), the original training set.

`convert_extra` and `planner_baseline` build `base_plus_planner` = base + the first 100 extra demos.

## Part 2: driving policies and thresholds

**Policy recipe** (`src/vig_dagger/configs/policy.json`; every training in the task uses it):

- frozen DINOv2-small encoder per camera, cross-attention pooling to one token per camera;
- an 8-layer Transformer action head (width 256) with adaptive LayerNorm on the flow time and the robot state;
- linear-path flow matching with 10 Euler sampling steps;
- batch 64, AdamW, lr 1e-4 with cosine decay;
- random 224 px crops during training.

A policy degrades its own input images by area-downsampling and bilinear upsampling by an integer factor.

**Training length.** Every training runs `training.epochs` = 150 passes over its training demos:

max_updates = 150 × ceil(training samples / 64)

A training sample is any step with a full 16-step future. 2 % of the demos are held out for the validation loss.
A planner demo has about 700 steps, so a training on the 100 base demos runs about 160,000 updates, and one on 200
demos about twice that.

**Round `moe`.** `train/drivers/moe` trains the 7-level MoE on the 100 base demos:

- every training sample is degraded to a random level (factor 1, 2, 4, …, 64);
- every head layer's feed-forward block has 8 experts (top-2) with one router per level.

**Round `per_resolution`.** The full-resolution policy is the `base` retraining of seed 0: the same data, recipe
and seed, so it isn't trained twice. `train/drivers/per_resolution/res<k>` trains a dense policy on images degraded
by *k*, for *k* = 2, 4, …, 64.

**Thresholds** (`score_<round>_*`, `thresholds_<round>`):

- **Frames scored:** every 4th frame of the first 50 base demos (about 8,000 frames).
- **Scored like DAgger:** VIG with 32 samples per level; flow loss with 512 draws.
- **Output:** the percentiles (p50 … p99) go to `policies/drivers/<round>/thresholds.json`.

### The two scores

Both are computed on the full-resolution policy's normalized actions (`src/vig_dagger/scoring.py`).

- **VIG.** Draw 32 chunks at full resolution and at every degraded level, all from the same 32 starting noises.
  For each degraded level, take the 1-D Wasserstein distance between the two sample sets: sort each of the
  16 × 7 chunk cells, compare, and average over the cells. VIG is the largest of these distances over the levels.
  In round `per_resolution`, each level's samples come from its own policy and are first mapped into the
  full-resolution policy's action normalization.
- **Flow loss.** Draw one chunk from the full-resolution policy and average the policy's flow-matching loss on
  that chunk over 512 random (noise, time) draws. It is noisy from frame to frame.

## Part 3: DAgger

Each round collects with **both** scorers from the same driving policy, on the same scene sequence.

**Each episode** (`src/vig_dagger/workers/dagger.py`, gate in `src/vig_dagger/gate.py`):

1. The full-resolution driving policy drives. Every observation is scored, also while the expert drives.
2. **Take over:** the expert takes over after **2** consecutive scores above the **take threshold (p95** of the
   score on demo frames).
3. **Return:** control returns to the policy after **16** consecutive scores below the **return threshold (p50)**,
   on the very next step, whatever the expert was doing, for the rest of the same episode. Without a return, the
   expert finishes the task.
4. **End:** the episode succeeds by the success rule above within 1200 steps (policy and expert steps alike). It
   fails at the step limit, or when the expert's plan fails.

**The expert** (`OpenDrawerPlanner(..., expert=True)`, `dagger.expert: demo`) is the demo planner with a fresh
strategy at each takeover, moving like the demos (the strategy's speeds, pauses and transit midpoints). It plans
only what is left, from the live scene: the stage latches, what is in the gripper and how far the drawer is open.
If the policy already holds the handle, it keeps pulling; if it holds the charger, it carries it to the drawer; if
the drawer slid shut before the charger went in, it opens it again.

It **corrects in place** instead of backing out to the start of a stage (`EXPERT_IN_PLACE`):

- **near the handle** (in front of the bar within 15 cm, within 5 cm of the approach line, within 30° of the grasp
  orientation): it goes straight for the grasp, at the hand's own point along the bar;
- **hovering over the charger:** it descends straight onto it, keeping the hand's yaw when that fits the grasp;
- **holding the charger over the drawer's release area:** it lowers the charger from where it is instead of
  carrying it up and over to a sampled release point;
- **hand at the drawer front:** it pushes from there, at the hand's own height and point along the face.

Like the teleoperated demos, the expert never retries: a missed or slipped grasp, a stage that does not complete,
a hand stuck far from its target, or a charger out of reach ends the plan, and the episode counts as failed.

Tested from identical mid-stage states (the demo planner driven to the situation, then 8 random actions to imitate
a misaligned policy, then the expert), with the corrections on vs off (`EXPERT_IN_PLACE`), medians over 8 scenes:

| situation | steps until the stage is done | hand travel | stage done |
|---|---|---|---|
| hand within 10 cm of the handle | 106 vs 120 (until the drawer counts as opened) | 27 vs 30 cm | 8/8 vs 7/8 |
| charger held over the drawer (6 scenes) | 22 vs 46 (until placed) | 9 vs 17 cm | 6/6 vs 6/6 |
| hand at the drawer front | 47 vs 94 (until closed) | 33 vs 57 cm | 8/8 vs 8/8 |

From the handle situation the expert then did the rest of the task and finished it in 5 of 8 scenes (4 of 8
without the corrections). The other episodes failed in the expert's own later stages: the charger dropped during
the lift, the hand stuck turning for the charger grasp, a handle slip. These are the planner's known failure modes
(from reset it solved 44 of 48 scenes when it was built), and an expert failure ends a DAgger episode.

**Saved output** (`dagger/<round>/<scorer>/`):

- **Successful episodes are the DAgger demos.** They go to `trajectories/worker_<w>_seg<k>.hdf5` in the planner
  demos' format, policy and expert steps alike, with the executed actions as labels, plus the episode's DAgger
  record (`dagger/`) and task state (`task_state/`).
- **Every attempt**, failed ones included, gets a per-step record in `attempts/worker_<w>.h5`:
  - who drove each step;
  - the score of every observation;
  - VIG's distance per level;
  - the task state: TCP and charger poses, drawer opening, joint positions, what the gripper holds, the stage
    latches.

  It also gets a summary line in `episodes/worker_<w>.jsonl`. `settings.json` records the thresholds and settings
  used.

**Stopping and workers:**

- `dagger.workers_per_scorer` processes per scorer (default: half the CPU budget each) share the scene sequence.
- They stop once the collection has 100 successful episodes, or after 20 × 100 attempts.
- The training set takes the first 100 successes to finish.
- A collection refuses to continue with changed thresholds or settings.

### DAgger metrics (`analyze_<round>`)

**Per attempt:**

- steps and expert steps;
- expert share (expert steps / steps);
- takeovers and returns;
- the step and the task stage of the first takeover;
- who finished a successful episode;
- why it ended:
  - `success_policy`, `success_expert`: who drove the last step;
  - `time_limit_policy`, `time_limit_expert`: the step limit, by who drove last;
  - `expert_failed`: the expert's plan failed (the planner's reason is in `expert_failure`);
- the furthest stage reached and failure flags;
- the share of policy-driven steps above the take threshold, and of expert-driven steps below the return threshold.

**Per takeover:** its start, length, stage, triggering score, and whether it ended in a return or at the episode
end.

**Stages** (`src/vig_dagger/stages.py`) are computed from simulator state:

Reach handle → Handle approach → Grasp handle → Pull → Release → Reach charger → Charger approach → Grasp charger →
Lift → Carry → Lower, release → Retreat → Push approach → Push → Hold

**Failure flags:** missed the handle, handle slipped, missed the charger, knocked the charger, dropped the charger,
stalled (the hand moved less than 2 cm in 50 steps; this also happens briefly in successful episodes while the
fingers close on the charger).

**Collection metrics**, each for successful, failed and all attempts:

- attempts, successes, **collection success rate**, failure reasons;
- **pooled over all steps:** expert share, takeovers per 100 policy-driven steps;
- **distributions over episodes** (mean, std, min, p10, p25, median, p75, p90, max): steps, expert steps,
  **expert share**, **takeovers**, returns, first-takeover step; and over takeovers, the expert-segment length;
- the share of episodes without a takeover;
- **per stage:** steps, expert share, takeovers starting there, episodes with a takeover there.

**Outputs:**

- `results/dagger_<round>_<scorer>.json` (all of the above), `results/dagger_<round>_<scorer>_episodes.csv`,
  `_takeovers.csv`, and `_scores.npz` (the scores on policy- and expert-driven steps);
- in `results/REPORT.md` and `results/figures/`: the collection tables, per-episode distributions, score
  histograms with the thresholds, expert share per stage, and score timelines of six episodes (see
  [Results](#results)).

## Part 4: retraining and evaluation

Every retrained policy is a **dense full-resolution policy** (no experts, no image degradation), trained 150 epochs
with seeds 0, 1 and 2.

**Training sets.** All start with the same base demos, keeping their identities, so their train/validation
assignment is the same:

| training set | demos | used by |
|---|---|---|
| `base` | the 100 base demos | both rounds (baseline) |
| `base_plus_planner` | base + 100 extra planner demos | both rounds (baseline) |
| `base_plus_moe_vig`, `base_plus_moe_flow` | base + round `moe`'s 100 VIG / flow-loss DAgger episodes (whole episodes: policy and expert steps) | round `moe` |
| `base_plus_per_resolution_vig`, `…_flow` | base + round `per_resolution`'s DAgger episodes | round `per_resolution` |

**Evaluation** (`eval/<policy>/level<f>/…`): 200 rollouts per policy and level, on the same scenes
(60,000–60,199) for all of them; one episode at a time, CPU physics, 1200 steps, 8 executed actions per prediction.
Each 25-episode chunk is its own job, so evaluations run in parallel.

| evaluated | how | rollouts |
|---|---|---|
| `retrain/base/seed0,1,2` (the base policies; `seed0` is also round 2's full-resolution driver) | full resolution | 3 × 200 |
| `retrain/base_plus_planner/seed0,1,2` | full resolution | 3 × 200 |
| `retrain/base_plus_moe_{vig,flow}/seed0,1,2` (round 1's DAgger retraining) | full resolution | 6 × 200 |
| `retrain/base_plus_per_resolution_{vig,flow}/seed0,1,2` (round 2's DAgger retraining) | full resolution | 6 × 200 |
| `driver/moe` | at each of its 7 levels | 7 × 200 |
| `driver/per_resolution/res2 … res64` | each on images degraded by its own factor | 6 × 200 |

That makes 6,200 rollouts.

Each episode records success, steps, the stage latches reached, the furthest stage, the stage a failure ended in,
the steps spent in each stage, and the failure flags.

**Summary** (`summarize`):

- per training set:
  - the success rate of each seed, their mean ± std;
  - a paired comparison with `base`: scenes solved by only one of the two, over all seeds (seed *i* against seed
    *i*), with a two-sided sign test;
- per resolution level, the driving policies side by side: the MoE at that level, and round 2's policy for it
  (`base/seed0` at full resolution, `res<k>` at 1/*k*).

Outputs: `results/eval_summary.json`, `results/eval_episodes.csv` (every episode), and the evaluation tables and
figures in `results/REPORT.md` (see [Results](#results)).

## Results

Everything is written to `<work_dir>/results`, no account needed. `results/REPORT.md` collects it in one readable
file:

- **Progress:** how many jobs are done.
- **Evaluation:** success per training set and seed, mean ± std, and the paired comparison with `base`.
- **Driving policies by resolution:** the MoE and round 2's policy at each level, side by side.
- **Where evaluation episodes end:** the furthest stage of the failed episodes and the failure flags, per policy.
- **DAgger collections:** for each round and scorer, the thresholds, attempts, successes, success rate, pooled
  and per-episode expert share, takeovers, returns, takeover lengths and failure reasons; the same split into
  successful and failed attempts; and the expert share per task stage.
- **Trainings:** updates, last training loss, final and best validation loss (also in
  `results/training_summary.csv`).
- **Figures** in `results/figures/`:
  - success per training set (`eval_success.png`);
  - the driving policies by resolution (`eval_drivers_by_level.png`);
  - per round, the DAgger distributions (`dagger_<round>_distributions.png`), the scores with the thresholds
    (`dagger_<round>_scores.png`), the expert share per stage (`dagger_<round>_stages.png`), and score timelines
    (`dagger_<round>_<scorer>_timelines.png`);
  - the training curves (`training_<set>.png`).

The `analyze_<round>` and `summarize` jobs refresh the report; `vig-dagger report -c …` regenerates it at any
time, including from partial results. The raw numbers behind it:

| file | content |
|---|---|
| `results/eval_summary.json`, `eval_episodes.csv` | every evaluation metric; one row per evaluation episode |
| `results/dagger_<round>_<scorer>.json`, `_episodes.csv`, `_takeovers.csv`, `_scores.npz` | every DAgger metric; one row per attempt and per takeover; the scores |
| `policies/**/metrics.jsonl` | each training's log: `train/loss`, `train/lr`, `val/loss` per `train/step` |

### Weights & Biases (optional)

With `wandb.enabled: true` the same results also go to W&B, to the team `wandb.entity` (`vig-dagger-exp`) and project
`wandb.project`, grouped by the experiment `name`:

| run | content |
|---|---|
| `<name>/train/retrain/<set>/seed<s>` | training and validation loss, learning rate |
| `<name>/train/drivers/moe`, `<name>/train/drivers/per_resolution/res<k>` | the same, for driving policies |
| `<name>/dagger/<round>/<scorer>`, `<name>/dagger/<round>/compare` | part 3 metrics (above) |
| `<name>/eval/<policy>`, `<name>/eval/summary` | part 4 metrics (above) |

Runs carry tags (`train`, `retrain`, the set, the seed, `dagger`, the round, `eval`, …) for filtering. This needs
`wandb login` with an account in the entity; `vig-dagger check` tests it.

## Hugging Face

| repository | content |
|---|---|
| [chengenwu/OpenDrawer-ManagerBased-v0](https://huggingface.co/datasets/chengenwu/OpenDrawer-ManagerBased-v0) (dataset, public) | `raw/planner` (100 demos), `raw/planner_extra` (100 demos), 15 GB |
| [chengenwu/vig-dagger-opendrawer-v1](https://huggingface.co/datasets/chengenwu/vig-dagger-opendrawer-v1) (dataset, public) | only with `sharing.mode: hub`: the shared run (the tasks' claims, checkpoints, DAgger sets, metrics, evaluation episodes) |

The queue downloads the demos itself (`download_demos`); `vig-dagger download-demos` does the same by hand. Every
policy, the MoE included, is trained here. Downloads need no login; with `sharing.mode: hub` every machine also needs
a login with write access to the shared repo (`hf auth login`). Outputs can be uploaded by hand to the repositories
set in `hub.*`: Outputs can be shared through the repositories set in `hub.*`:

- `vig-dagger upload demos` uploads `raw/planner` and `raw/planner_extra` (the `part_*.hdf5` files) to
  `hub.dataset_repo`;
- `vig-dagger upload policy` uploads the MoE driver (weights only) and its thresholds to `hub.model_repo`;
- `vig-dagger upload dagger` uploads the DAgger collections to `hub.dataset_repo` (the shared run's repo already holds
  them).

Uploading needs `huggingface-cli login` with write access; new repositories are created private. Uploaded
checkpoints are weights only: about a third of the size, they load for inference but cannot resume training.

A later run can also reuse an uploaded MoE (`drivers.moe.source: download`, `hub.model_repo`), or existing files: `vig-dagger import-demos part0.hdf5 …` links planner demo files as the raw demos, in argument order
(`--extra` for the extra demos); `vig-dagger import-policy moe.pt --thresholds thresholds.json` installs a MoE
checkpoint as round `moe`'s driver.

## Output directory

```
<work_dir>/
  config.json                                      the resolved configuration every job reads
  data/raw/planner/, planner_extra/                planner demos (Isaac Lab demo files, part_<k>.hdf5; links into hub/)
  data/datasets/{planner,base,planner_extra,base_plus_*,dagger_*}/   training datasets
  policies/drivers/moe/                            the MoE driver (latest.pt, thresholds.json)
  policies/drivers/per_resolution/res<k>/          the degraded-level policies of round per_resolution, thresholds.json
  policies/retrain/<set>/seed<s>/                  every retrained policy (latest.pt, run_config.json, metrics.jsonl)
  dagger/<round>/<scorer>/                         settings.json, trajectories/, attempts/, episodes/
  eval/<policy>/level<f>/seeds_<first>.jsonl       one line per evaluation episode
  results/                                         every metric as JSON / CSV, REPORT.md, figures/
  logs/<job>.log, logs/pids/                       one log per job; PIDs of running jobs
  tmp/                                             the simulator jobs' temporary directories
  hub/                                             downloaded Hugging Face files
```

## Configuration reference

The defaults live in `src/vig_dagger/config.py`; `configs/opendrawer.yaml` holds the experiment's values.

| key | value | meaning |
|---|---|---|
| `name`, `work_dir` | | experiment name and output directory |
| `resources.gpus` | `auto` | every GPU, or a list of indices |
| `resources.sim_slots`, `max_load` | `auto`, `auto` | CPU budget (cores / 3) and load ceiling (cores) |
| `resources.sim_gpu_memory_gb`, `train_gpu_memory_gb` | 6, 8 | free GPU memory a simulator job / training needs (measured: 4.7 / 2.6–2.9 GB) |
| `resources.sim_ram_gb`, `train_ram_gb`, `train_slots` | 8, 24, 2 | RAM and CPU-budget estimates (measured: 6.8 / 19 GB; 1.5 / 1.3 cores) |
| `resources.dataloader_workers` | 16 | data-loading processes per training |
| `resources.sim_start_gap_s` | 20 | seconds between two simulator-job starts (Isaac Sim can crash when several start at once) |
| `isaac.python` | `auto` | the Isaac environment's interpreter (`<repo>/.venv-isaac/bin/python`) |
| `demos.source`, `num_demos`, `demos_per_part` | `download`, 100, 25 | `download` or `collect`; the base set's size; demos per collection job |
| `demos.max_episode_steps` | 1200 | |
| `training.epochs` | 150 | passes over the training demos, for every training |
| `training.batch_size`, `learning_rate`, `warmup_updates`, `val_fraction` | 64, 1e-4, 1000, 0.02 | |
| `training.checkpoint_every`, `keep_checkpoints`, `max_restarts` | 5000, 1, 5 | rolling checkpoints and automatic restarts |
| `rounds` | `[moe, per_resolution]` | the rounds, in order |
| `sharing.mode`, `repo` | `none`, `chengenwu/vig-dagger-opendrawer-v1` | `none`: one machine runs everything, or tasks are assigned by hand (`run --tasks`); `hub`: the machines share the run through this public Hugging Face repo (see [the job queue](#the-job-queue)) |
| `sharing.poll_s`, `heartbeat_s`, `upload_trajectories` | 60, 1800, true | how often a machine reads the repo and sends a heartbeat; whether the DAgger trajectories are uploaded too |
| `drivers.moe.source` | `train` | `train` or `download` |
| `drivers.per_resolution.factors` | 1 … 64 | the levels VIG compares (1 = the base retraining of seed 0) |
| `scorers.vig.num_samples` | 32 | samples per level |
| `scorers.flow.num_loss_samples`, `num_action_samples` | 512, 1 | |
| `thresholds.num_demos`, `frame_stride`, `percentiles` | 50, 4, p50 … p99 | demo frames for the thresholds |
| `dagger.num_demos` | 100 | successful episodes per scorer and round |
| `dagger.thresholds.<scorer>.take` / `.return` | `p95` / `p50` | `p<q>` or a number |
| `dagger.take_count`, `return_count` | 2, 16 | consecutive steps needed to switch |
| `dagger.expert` | `demo` | `demo` (planner-demo motion) or `fast` (the earlier, faster expert) |
| `dagger.max_episode_steps`, `policy_chunk_steps`, `seed_start` | 1200, 8, 1e6 | |
| `dagger.workers_per_scorer`, `max_attempts` | `auto`, 20 × 100 | |
| `retrain.seeds`, `planner_baseline` | `[0, 1, 2]`, true | |
| `evaluation.seed_start`, `num_episodes`, `max_episode_steps`, `chunk` | 60000, 200, 1200, 25 | `chunk`: episodes per job |
| `evaluation.evaluate_drivers`, `driver_all_levels` | true, true | also evaluate the driving policies (MoE at every level, `res2` … `res64`) |
| `hub.dataset_repo`, `hub.model_repo` | `chengenwu/OpenDrawer-ManagerBased-v0`, none | see [Hugging Face](#hugging-face) |
| `wandb.enabled`, `entity`, `project` | false, `vig-dagger-exp`, `opendrawer` | optional W&B logging |

## Cost estimates

Measured on a shared machine (2× RTX PRO 6000 Blackwell, 64 cores; both GPUs also busy with other jobs). Expect
better on a dedicated machine.

| step | measured | for this experiment |
|---|---|---|
| demo download | 15 GB (~75 MB per demo, 535–928 steps each, median 704) | a few minutes to an hour, depending on the connection |
| demo collection (only with `demos.source: collect`) | ~25 s per demo | 200 demos in 8 jobs: about 30 minutes |
| convert | a few seconds per demo | 200 demos: a few minutes with 8 processes |
| one training, 100 demos | ~160k updates; 4.5 updates/s on a busy GPU, 8–13 on a free one | 3–10 h |
| one training, 200 demos | ~320k updates | 7–20 h |
| trainings in total | | the MoE, 6 per-resolution drivers, 18 retrained policies = 25; about 6.5 million updates |
| thresholds | ~0.4 s per frame (VIG + flow) | ~8,000 frames per round in 4 jobs: about 15 minutes each |
| DAgger | ~0.2 s per step to score (VIG or flow); 3–5 min per episode; ~75 MB per saved episode | 100 successes per scorer and round: a few hours with 10 workers per scorer |
| evaluation | ~25 s per successful episode, up to ~40 s for one that runs to the step limit | 31 policy/level combinations × 200 = 6,200 episodes ≈ 50 CPU-hours |

The trainings dominate: about 6.5 million updates in total, so roughly 5–8 days on 2 GPUs and 3–4 days on 4 at the
speeds above (one training per GPU; the dependencies between rounds keep some GPUs idle at times).

Disk in total, about 80 GB (120 GB recommended):
- the downloaded planner demos, 15 GB;
- the four DAgger collections, about 35 GB (~75 MB per saved episode);
- the checkpoints, about 20 GB (each training keeps one rolling checkpoint and the final one);
- datasets, logs and results, a few GB.

Resources per job, measured:
- a training: 2.6–2.9 GB of GPU memory, ~19 GB of RAM (with its 16 data-loading workers), ~1.3 CPU cores;
- a simulator job: 4.7 GB of GPU memory (the renderer and the MoE), 6.8 GB of RAM, about 1.5 cores.

## Troubleshooting

- **A job failed.** `vig-dagger status` and the queue log name it; its output is in `<work_dir>/logs/<job>.log`.
  Fix the cause and run the same `vig-dagger run` again; only unfinished jobs run.
- **Nothing starts.** The machine may be full: the queue waits for CPU budget, the load ceiling, RAM, or a GPU with
  enough free memory. On a machine shared with other users, lower `resources.train_gpu_memory_gb` or raise
  `resources.max_load` only if you know the jobs fit.
- **Isaac Sim's log noise.** Every simulator log starts with hundreds of lines like `Failed to create change watch
  … errno=28/No space left on device` (the kernel's file-watch limit, not the disk) and warnings about GLFW, the
  display, DLSS or IOMMU. They are harmless. Real errors show as a Python traceback near the end of the log.
- **`obs ee_pose is the 'gripper' frame in this process`.** A simulator job ran without `PYTHONHASHSEED=0`. The queue sets it; set it
  yourself when you start a worker by hand.
- **Isaac Sim does not start** (`vig-dagger check` fails, a crash dump under `.venv-isaac/…/Kit/Isaac-Sim/5.1/`):
  check the GPU (it needs RT cores) and the driver, then rerun `bash scripts/install_isaac.sh`. The full output of
  the check is in `<work_dir>/logs/check_isaac.log`.
- **Changed settings after jobs ran.**
  - A training keeps the run config it started with (`<run>/run_config.json`); delete the run directory to retrain
    it with new settings.
  - A DAgger collection refuses to continue with different thresholds or settings (only `num_demos` and
    `max_attempts` may change).

  Use a new `work_dir` or `name` for a new variant of the experiment.
- **W&B errors at startup** (only with `wandb.enabled: true`; a training checks its run online): run `wandb login`, or
  set `wandb.enabled: false`.

## Repository layout

```
configs/opendrawer.yaml         the experiment
configs/smoke.yaml              a short end-to-end check
scripts/install_isaac.sh        installs the Isaac environment (.venv-isaac, third_party/IsaacLab)
src/vig_dagger/
  cli.py                        the vig-dagger command
  plan.py                       the experiment's job graph
  jobs.py                       the job queue (dependencies, resources, retries, adoption)
  steps.py                      short steps: conversion, thresholds, DAgger datasets, analysis, summary
  config.py                     defaults, rounds, and the output layout
  task.py                       task constants shared by both environments
  opendrawer_env/               the OpenDrawer Isaac Lab environment and its charger asset
  isaac.py                      simulator side: launching Isaac Sim, the env, success rule, task state, recording
  expert/planner.py             the motion planner (demos and DAgger takeovers)
  policy.py                     running checkpoints: one policy, or the per-resolution set
  scoring.py, gate.py           VIG / flow loss, and the two-threshold gate
  stages.py                     task stages and failure flags
  dataset.py, train.py          training datasets, run configs and the training worker
  analysis.py, evaluation.py    part 3 and part 4 metrics
  report.py                     results/REPORT.md and the figures
  tracking.py, hub.py           optional W&B logging, and Hugging Face
  workers/                      collect, convert, score_demos, dagger, evaluate (one process each)
  configs/policy.json           the policy recipe
src/robopolicy/                 the robopolicy training library
tests/robopolicy/               the library's tests
```

`src/robopolicy` is a copy of the robopolicy library (`refactor/policy-core`, branch `moe`, commit `ce03ad2`, with
the input-mode additions). It has two small additions: W&B `group`/`tags` (`training/logging.py`) and
`training.keep_checkpoints` (rolling checkpoint pruning, `training/trainer.py`).

The environment in `src/vig_dagger/opendrawer_env/` and the planner come from the RoboticArmSim project (Isaac Lab
task `OpenDrawer-ManagerBased-v0`, `scripts/tools/open_drawer_planner.py`); the planner here adds the DAgger
expert's demo-speed motion and in-place corrections.
