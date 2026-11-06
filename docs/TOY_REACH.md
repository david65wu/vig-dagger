# Toy reaching environment

`ToyReachEnv` is a separate Isaac Lab `ManagerBasedEnv` for the A→B
experiment. It contains a Franka Panda on a plain table, a hand-mounted wrist
camera, and a fixed third-person camera. Both provide RGB uint8 images at 256×256.
Blue marks A and green marks B; smaller-looking table projections help locate
the endpoints in the wrist view. Markers are visual only, with no collision.

Defaults, in meters relative to the robot base:

- A = `(0.45, -0.15, 0.35)`, B = `(0.45, 0.15, 0.35)`.
- Downward EEF orientation, quaternion `(w,x,y,z) = (0,1,0,0)`.
- A reset samples a uniform disk of radius 1 cm perpendicular to A→B at A.
- Goal tolerance: 5 mm. Physics: 120 Hz; actions and cameras: 60 Hz.
- Exactly one environment; CPU physics and GPU rendering.

The tube radius specifies the reset distribution. The demonstration planner uses
seeded cubic Bezier curves, with perpendicular control-point offsets up to 8 mm
inside the 1 cm tube. The environment itself does not constrain motion to the
tube, so off-path positions remain available for OOD probes.

## Collect demonstrations

```bash
source /cpfs02/user/caojianuo/vig-dagger-runs/setup/graphics-env.sh
OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=1 .venv-isaac/bin/python scripts/collect_toy_reach.py \
  --gpu 6 --num-demos 20 --seed 100 --out /tmp/toy-reach-20-demos
```

Use a fresh output directory. The planner samples three offset control points
and a fixed endpoint B; the curve remains in the tube by the convex-hull property.
The arm follows a 12 mm look-ahead target with differential IK, a fixed downward
orientation and an open gripper. This is a geometric Cartesian planner for the
clear workspace, without obstacle-search planning. Collection rejects a demo if
its actual TCP leaves the 1 cm tube, orientation drifts more than 0.02 rad, or it
fails to stay within 1.5 mm of B for 15 consecutive control steps.

Outputs include:

- `mp4s/demo_000.mp4` …: labeled side-by-side review videos at 30 fps, real-time
  playback, with a half-second final-frame hold.
- `dataset/`: robopolicy small-files manifests, one HDF5 shard per demo, and
  separate unlabelled `third_person` and `wrist` training MP4s at 60 fps.
- Each HDF5 shard includes lossless RGB, TCP state, joint state, actions,
  next states, timestamps, planner targets/control points, the settled initial
  simulator state, and the final observation.
- `settings.json`, `progress.json`, and `summary.json`: settings and per-demo checks.

Frame/observation `t` precedes action `t`; `next_state[t]` follows that action.
Setup motion is excluded. Finger positions are absolute (not OpenDrawer's relative
finger positions). Training must use this task's camera names and action units.

After collection, validate all stored transitions and decode every video:

```bash
.venv/bin/python scripts/validate_toy_reach_dataset.py /tmp/toy-reach-20-demos
```

This also creates `all_20_demos.mp4`, `individual_demo_mp4s.zip`,
`trajectories.png`, and `validation.json`. It refuses to overwrite an existing
combined MP4 or ZIP. The combined video plays the 20 review clips in order.

## Preview and validation

From the repository, using the existing Isaac environment:

```bash
# This host needs its existing graphics-library configuration:
source /cpfs02/user/caojianuo/vig-dagger-runs/setup/graphics-env.sh
OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=1 .venv-isaac/bin/python scripts/preview_toy_reach.py \
  --gpu 6 --out /tmp/toy-reach-preview
```

Use a fresh output directory. Add `--gui` on a machine with a display. The preview
checks reproducible seeded resets, initial tube membership, motion to B, image
shape/content, and placement at two off-path probes. It saves a side-by-side MP4,
camera PNGs, an EEF position trace, and `validation.json`. This is a pose-servo
smoke check, not the future motion-planner demo collection.

## Environment API

Start Isaac Sim with cameras enabled before importing the task/config:

```python
from vig_dagger.toy_reach_env.config import ToyReachEnvCfg
from vig_dagger.toy_reach_env.environment import ToyReachEnv

cfg = ToyReachEnvCfg()
env = ToyReachEnv(cfg)
obs, info = env.reset(seed=7)
obs, info = env.step(env.action_toward(cfg.point_b))

# Deterministic nominal A, or a physically consistent OOD observation:
obs, info = env.reset(seed=7, options={"randomize_start": False})
obs = env.place_eef((0.50, 0.0, 0.35))
env.close()
```

`step` returns `(observations, extras)`, following `ManagerBasedEnv`, with no
reward, termination, or automatic reset. The caller decides the rollout length.
`obs["task"]["success"]` reports positional goal success.

Observations:

| Key | Shape | Meaning |
|---|---|---|
| `policy` | `(1,9)` | TCP xyz, quaternion wxyz, two absolute finger positions |
| `rgb_camera.third_person_cam` | `(1,256,256,3)` | Fixed camera RGB |
| `rgb_camera.wrist_cam` | `(1,256,256,3)` | Wrist RGB |
| `task.goal_distance` | `(1,1)` | Distance to B, meters |
| `task.line_distance` | `(1,1)` | Distance to the A→B segment, meters |
| `task.success` | `(1,1)` | Within goal tolerance |

Actions have shape `(1,7)`: base-frame delta translation in meters, delta rotation
vector in radians, and a binary gripper command (positive opens). The IK action
scale is **1.0**, so these commands are not interchangeable with OpenDrawer's
scaled action values. The same 0.1034 m TCP offset is used for control and
observation. `action_toward` holds the configured orientation and open gripper;
arbitrary user actions can change them.

`reset` physically settles the robot at its sampled start. `place_eef` settles at
an explicitly requested position with the configured orientation and refreshed
cameras; unreachable poses raise an error. These setup movements do not emit
episode steps or recorder samples. Use the explicit returned observation to start
a later collection; recorder-based collection integration is not included yet.

A/B, tube radius, orientation, camera eye/target, and image sizes are configurable
before constructing the environment. For the initial experiment, keep them fixed
apart from the planned narrow path variation.

## Six-policy training experiment

The first training experiment is in
`/cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_training_v1`.
Each training is an independent process with its own session, no terminal,
stdin from `/dev/null`, and stdout/stderr appended to a log file. It survives
SSH disconnects. A machine reboot still requires relaunching the workers.

| GPU | Run | Training demos | Images | MoE |
|---|---|---:|---|---|
| 0 | `demos5_dense_full` | 5 | Full resolution | No |
| 1 | `demos5_moe_7level` | 5 | Random 7 levels | Yes |
| 2 | `demos5_dense_7level` | 5 | Random 7 levels | No |
| 3 | `demos20_dense_full` | 20 | Full resolution | No |
| 4 | `demos20_moe_7level` | 20 | Random 7 levels | Yes |
| 5 | `demos20_dense_7level` | 20 | Random 7 levels | No |

All six use the existing frozen DINOv2-small / 8-layer Transformer flow recipe,
batch 64, learning rate 1e-4, 100 warmup updates, a cosine schedule, seed 0,
16-step action chunks, and 100 training epochs. The five-demo set is demos
000–004 of the original twenty. Five additional demos (planner seeds 1000–1004)
are used only for validation. Every run uses its own training-set normalization.
The seven-level dense and MoE policies both receive the resolution index as
conditioning; their images use factors 1, 2, 4, 8, 16, 32, 64, sampled uniformly
per observation. Full-resolution dense training retains the common random crop.

Two loader workers yield 38 updates per epoch for 5 demos and 150 for 20 demos
(including each worker's final partial batch), giving 3,800 and 15,000 updates.
Validation runs every five epochs. W&B is disabled.

From the repository:

```bash
.venv/bin/python -m vig_dagger.toy_training status \
  --root /cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_training_v1
```

`logs/<run>.log` contains progress, throughput, and estimated remaining time;
`policies/<run>/metrics.jsonl` contains training/validation losses.
`policies/<run>/status.json` is refreshed every 50 updates. Exact configurations,
dataset identities, and GPU assignments are saved under the experiment root.

Permanent checkpoints are `epoch-025.pt`, `epoch-050.pt`, `epoch-100.pt`, and
`latest.pt`. The 20-demo models additionally retain `matched-3800.pt` for a
comparison at the five-demo runs' final update count. Three rolling recovery
checkpoints are retained, written every 250 updates.

After a machine restart or an interrupted process, this command resumes missing
runs from their newest checkpoint; active or completed runs are skipped:

```bash
.venv/bin/python -m vig_dagger.toy_training launch \
  --root /cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_training_v1
```

Recovery restarts the data iterator at the next logical epoch, as in the
underlying robopolicy trainer, rather than reproducing the interrupted iterator.

## Initial policy evaluation

Results are saved under
`/cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_eval_v1`.
Each final checkpoint has 10 ID rollouts from new planner start seeds 2000–2009
and 8 recovery rollouts from the line midpoint with ±3 and ±5 cm offsets along
X or Z. All policies act at full image resolution, execute 8 actions per
16-action chunk, and have a 1,200-step limit. Success requires 15 consecutive
steps within 5 mm of B. Extreme actions and workspace exits terminate a rollout.
Recovery tests change both the start position along the path and its transverse
offset; they do not independently isolate the effect of transverse offset.

Shared physical probes cover progress 0.25, 0.5, and 0.75, with transverse radii
0, 0.5, 1, 2, 3, 5, and 8 cm. Each nonzero radius has eight random directions,
matched across radii and policies. The robot settles before recording its actual
pose and both cameras. There are 147 probes and 100 calibration frames from the
five held-out demonstrations. `observations.hdf5` contains these inputs.

At each observation, scoring draws 128 action chunks with shared noise across
the seven resolution levels. VIG is the maximum, over degraded levels, of mean
per-cell marginal Wasserstein-1 distance to full-resolution samples in normalized
action space. For full-resolution-only policies, external image degradation
produces a sensitivity diagnostic, not trained-resolution VIG. Flow
self-consistency averages the existing flow-matching loss over eight generated
chunks, each scored with 512 fresh noise/time draws. Action spread is the square
root of the trace of first-action XYZ covariance, in meters of commanded motion.

Warnings use each policy's own 95th percentile on the 100 held-out frames.
Ten recorded rollout starts per policy (two ID and all eight recovery cases)
are also scored, pairing actual outcomes with exact recorded state and the
first decoded RGB frame of the review MP4. These decoded images differ slightly
from the raw simulator images used when acting. Static probe comparisons use
lossless raw images. Direct-to-B reference disagreement is only a diagnostic:
a different valid trajectory can disagree with that reference.

`scripts/evaluate_toy_reach.py` captures probes and runs Isaac Lab rollouts;
`scripts/run_toy_evaluation.py` chains a policy's rollout and offline scoring
jobs. Jobs run in detached sessions, one policy per GPU. Each policy directory
contains `pipeline.json`, `rollouts.json`, `scores.json`, sampled action arrays,
trajectory arrays, and review videos. After all six jobs finish, regenerate the
figures and summary with:

```bash
.venv/bin/python scripts/analyze_toy_reach.py report \
  --root /cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_eval_v1
```

The report includes `ood_scores.png`, `action_distributions.png`,
`failure_warnings.png`, and `summary.json`. Comparison videos arrange five-demo
policies in the top row and twenty-demo policies in the bottom row; columns are
dense full resolution, MoE seven levels, and dense seven levels. Finished panels
hold their last frame while the other policies continue.

## Search for low-flow failures

`/cpfs02/user/caojianuo/vig-dagger-runs/toy_flow_search_v2` is a separate follow-up
experiment focused on flow and action distributions. It retains all six final
policies and does not compute VIG. Its `protocol.json` records the full design.

There are 69 static probes: an on-line A control, plus progress 0.5, 0.9, 0.98,
and 1.02 with zero offset or 2.5, 5, 10, and 20 mm offsets along ±X and ±Z.
At each probe, 128 full-resolution action chunks are sampled, and every chunk
receives its own flow loss from 128 fresh noise/time draws. The loss calculation
matches the underlying linear flow-matching MSE; a numerical check against
`compute_loss` verified agreement. Per-chunk Monte Carlo standard errors are
retained. These standard errors measure noise/time integration variability only.

Calibration uses 20 equally spaced frames, including both endpoints, from each
of the five held-out demonstrations. State thresholds use the p95 of the 100
mean flow values. Individual-chunk thresholds use pooled calibration sample
losses. Eight chunks at each calibration observation are additionally rescored
with 512 draws to match the individually scored rollout chunks.

Before observing outcomes, each policy selects twelve states: five on-line
controls, the lowest-flow offset at each of four progress values, two remaining
offsets with large reference disagreement relative to flow, and the highest-flow
remaining probe. This is a targeted search, not an unbiased failure-prevalence
estimate. Full simulator snapshots preserve joint/root positions and velocities
across repeated trials. Each selected state has eight stochastic rollouts with
the same success and termination rules as the first evaluation. The exact first
chunk executed receives a flow score with 512 draws; later samples can also
cause a failed continuation, so this pairing does not establish first-action
causality. Raw initial RGB, states, actions, and the first chunk are retained.

`scripts/toy_flow_search_sim.py` captures snapshots and runs repeated trials;
`scripts/analyze_toy_flow_search.py` scores, selects, and reports them.
`scripts/run_toy_flow_search.py` runs a detached per-policy pipeline. Completed
scoring and individual rollout records are reused on restart. Jobs use one GPU
per policy and survive an SSH disconnect.

The exploratory candidate criterion is mean flow below the held-out p95 and at
least six failures in eight trials. A separate narrow-distribution flag compares
XYZ spread to its calibration p95. Reports include Wilson 95% intervals for
failure rates and explicitly show on-line controls to expose reset/calibration
effects. No thresholds are tuned using rollout outcomes.

The initial `toy_flow_search_v1` rollout attempt is invalid and retained only for
diagnostics. Isaac Lab's `scene.reset_to` assigns saved physical joint velocities
to desired controller velocities. In this position-controlled task, desired
velocities must remain zero. The v2 restore preserves physical state while
explicitly zeroing desired velocity targets. An open-loop replay check compares
80 identical commands before and after restoration at three progress values;
policy workers wait for this check to pass. Static observations and completed
scores from v1 are reused, but none of its rollout outcomes are reused.

The completed v2 search contains 576 policy rollouts and 72 review MP4s. The
direct-to-B controller succeeded from all 21 unique selected states. Every
policy had at least one selected state with mean flow below its ID p95 and at
least six failures in eight trials: six states per five-demo policy and one per
twenty-demo policy. All of these states also had spread below their calibration
p95. These are counts in a deliberately selected search set, not failure rates
over a task distribution.

For example, observation 129 is a 10 mm negative-X offset at progress 0.90.
The twenty-demo full-resolution policy failed 8/8 trials there, with mean flow
0.028 (0.56 times its ID p95) and XYZ spread 0.188 mm. Observation 146 is a
10 mm negative-X offset at progress 0.98; the five-demo full-resolution policy
failed 8/8 trials with flow ratio 0.41 and spread 0.157 mm. The reference
controller recovered from both states.

Additional trajectory diagnostics use exact recorded states with decoded MP4
RGB, so they are distinguished from raw-input scoring. In trial 0 of the
twenty-demo example, the first sampled flow warning appeared at step 8, before
the goal was missed. In trial 0 of the five-demo example, it appeared at step
144 (2.4 seconds), when the EEF was approximately 86 mm from B. Thus an initial
low-flow failure case need not imply that continuous monitoring stays silent.
`flow_vs_failure.png`, `candidate_action_distributions.png`, `summary.json`,
the focused `counterexample_*.png` figures, and `trajectory_flow_*` diagnostics
are saved in the v2 run directory.

## Random endpoints on a sphere

The separate `SphereReachEnv` / `SphereReachEnvCfg` variant samples A and B on a
sphere centered at `(0.45, 0, 0.35)` m with radius `0.15` m. Directions come from
normalized isotropic Gaussian samples, giving uniform surface samples. Pairs
are rejected when their separation is below `0.10` m. The sphere defines the
sampling region; it is not a collision obstacle. Both markers and their table
projections move to the new endpoints on reset. The wrist and third-person
cameras retain the original arrangement.

The default path is a straight chord with a cubic Bezier variation inside an
8 mm tube. Its first and last control points are exactly A and B, so endpoint
sampling stays on the sphere. The physical reset settles at A within the same
1 mm placement tolerance as the original task. The EEF keeps the same downward
orientation and open fingers.

Because B changes every episode, the policy observation includes its coordinates:
`[TCP xyz, TCP quaternion wxyz, two finger positions, goal xyz]`, shape `(1,12)`.
Goal coordinates are in the robot-base frame. The original task retains its
9-dimensional observation. New sphere policies will therefore need their own
datasets and training configurations.

After starting Isaac Sim:

```python
from vig_dagger.toy_reach_env.sphere_config import SphereReachEnvCfg
from vig_dagger.toy_reach_env.sphere_environment import SphereReachEnv

env = SphereReachEnv(SphereReachEnvCfg())
obs, extras = env.reset(seed=5100)
plan = env.sample_plan(seed=105100)
target = plan.target(obs['policy'][0, :3].cpu().numpy())
obs, extras = env.step(env.action_toward(target, max_translation=.008))
```

Explicit `point_a` and `point_b` may be passed together in reset `options`; both
must lie on the configured sphere and satisfy the minimum separation. The
`eef_position` option can override initial placement for controlled OOD probes.
`scripts/preview_sphere_reach.py` checks seeded resets, goal observations, marker
poses, cameras, and reaching for eight random pairs plus six diameter crossings.
It writes per-episode MP4s, trajectory arrays, and a 3D trajectory plot.

## Sphere training experiment

`/cpfs02/user/caojianuo/vig-dagger-runs/toy_sphere_training_v1` uses the same six
variants, optimizer, architecture, seed, image processing, and 100-epoch update
budget calculation as the fixed-line experiment. The required input change is
`state_dim=12`, appending `obs/goal_xyz` to the original nine state values. Each
training view preserves the goal field and each demo's endpoint metadata.

Training demos are collected in `toy_sphere_20_demos_v1` with seeds 100–119;
independent validation demos are in `toy_sphere_validation_5_v1` with seeds
1000–1004. The first five training demos are the nested five-demo subset.
`scripts/collect_toy_reach.py --task sphere` records aligned RGB, actions, state,
goal XYZ, per-demo endpoints, and initial simulator state. The validator checks
the goal field, sphere constraints, path tube, timestamps, and all videos.

`scripts/start_sphere_training.py` is a detached coordinator. It waits for both
collections, validates them, prepares the six configurations, launches one
training per GPU, and records completion. The workers also have their own
detached sessions. SSH disconnection does not stop this pipeline. Machine
restarts require resuming the workers from checkpoint.

Because randomized chord lengths differ from the original fixed path, the
number of updates per epoch is recomputed from the new dataset. Twenty-demo
runs also save a checkpoint at the five-demo runs' final update count; its name
is `matched-<updates>.pt`. Permanent epoch checkpoints remain 25, 50, and 100.

```bash
.venv/bin/python -m vig_dagger.toy_training status \
  --root /cpfs02/user/caojianuo/vig-dagger-runs/toy_sphere_training_v1
```

Use the same command with `launch` instead of `status` to resume inactive,
unfinished workers. Collection/validation/coordinator progress is in
`pipeline.json` and `logs/coordinator.log`; individual training logs and metrics
follow the same layout as the fixed-line experiment.

## Simple sphere ID/OOD comparison

`toy_sphere_simple_eval_v1` compares only flow loss and sampled actions for all
six epoch-100 sphere policies. The first five training reaches are common to
all policies. Physical ID probes are placed at recorded positions nearest 25%,
50%, and 75% chord progress. OOD probes keep A/B fixed and move 3 or 8 cm along
four perpendicular directions. Both cameras are rendered at the actual pose.
This tests off-path shifts with familiar goals; it does not test unseen goals,
and the ID positions are training-supported rather than held-out examples.

`scripts/capture_sphere_simple.py` saves the 135 physical probes and 15 exact
recorded ID observations used to check placement effects. The simulator needs
the existing `setup/graphics-env.sh` environment. The detached coordinator
`scripts/analyze_sphere_simple.py run --root <output>` waits for capture, scores
one final policy per GPU, and creates the figures. Every policy uses factor 1.

Each observation has 128 generated action chunks. Flow loss is the mean linear
flow-matching MSE over those chunks, with 128 fresh noise/time draws per chunk,
using the policy's normalized 16-step, seven-dimensional actions. It is a
generated-action self-consistency score, not supervised loss against an expert.
Action clouds show the first translation command in millimeters. The main
example is fixed before scoring: demo 0, midpoint, positive first transverse
direction. Examples for all five reaches and all three Cartesian projections
are retained. Axes are matched within comparisons; direct-to-goal crosses are
visual references, not additional outcome metrics.

Outputs: `flow_id_vs_ood.png`, `action_distributions.png`,
`example_observations.png`, `examples/`, and `summary.json`. Per-policy
`scores.json` and `samples.npz` preserve raw flow scores and action samples.
The flow figure includes both exact recorded ID frames and recreated on-path
controls, since placement changes the full observation even at the same TCP
position. Action figures compare recorded ID against the two physical offsets.

## Sphere closed-loop rollouts

`toy_sphere_rollouts_v1` runs all six epoch-100 policies on 25 shared starts:
five familiar A-to-B pairs (the first five training demos), five new pairs
(seeds 6000–6004), and the five plotted midpoint starts at each of ID, +3 cm,
and +8 cm. Each start receives one stochastic trial, with matched seeds across
policies. These are descriptive counts over five cases per group.

Success requires TCP position within 5 mm of B for 15 consecutive 60 Hz steps.
The time limit is 1,200 steps (20 seconds). Policies use full-resolution inputs,
sample 16-step action chunks, and replan after executing eight commands. The
existing extreme-command and workspace-exit guards terminate failed rollouts
without clipping actions. Recorded midpoint ID cases restore the demo's joint
positions and physical velocities, set desired joint velocities to zero, and
assert agreement with the recorded policy state. Cameras are rendered again.

`scripts/run_sphere_rollouts.py run --root <output>` launches detached workers,
one per GPU; `scripts/rollout_sphere_simple.py` executes the episodes. Outputs
are `success_rates.png`, `success_rates.csv`, and `summary.json`, plus per-policy
`rollouts.json`, state/action arrays, and a two-camera MP4 for every episode.

`toy_sphere_rollouts_v2` reuses those 150 episodes and adds five outside-sphere
A/B pairs per policy (180 episodes total). Both endpoints are on a concentric
20 cm sphere, 5 cm beyond the training sphere. `prepare_sphere_video_eval.py`
checks endpoint reachability and calibrates both cameras using the renderer's
`CameraParams` matrices. USD camera pose metadata can remain stale under
Fabric, even with pose updates enabled; the rendered camera matrices are the
validation reference. The wrist camera is a rigid transform from the TCP.

`extend_sphere_rollouts.py` runs the additional cases and then invokes
`annotate_sphere_rollouts.py` on every episode. Annotated videos show both
cameras, orange sampled-action arrows, cyan executed-action arrows, and a flow
trace beneath them. Commands are magnified five times in 3D before projection,
explicitly labeled in the video. Off-screen or behind-camera projections are
clipped or labeled rather than falsely anchored on the image.

Flow is recomputed offline from exact saved states and decoded H264 camera
frames every eight control steps. Each value averages 32 generated chunks with
64 fresh noise/time draws per chunk. These images differ slightly from the raw
images used during policy execution. Orange arrows are fresh samples from that
reconstructed observation; cyan arrows are the exact recorded actions. Videos
play at 7.5 fps in real simulation time. Per-video JSON and NPZ files preserve
the method, flow values, sampling steps, and sampled first translation commands.

### Full chunks and training-flow calibration

The revised `chunk_videos/` overlays show all sixteen translation commands of
each of the 32 sampled chunks. Each path starts at the current TCP and sums the
base-frame commands in sequence, at true scale (1x). These are nominal command
paths, not forward-simulated EEF positions. Cyan shows the up to eight commands
actually executed before replanning; the discarded final eight predictions
were not recorded. `annotate_sphere_chunks.py` regenerates complete chunks with
the original seeds, verifies their first commands against the previously saved
samples, and copies the existing flow curves without rescoring. Full seven-
dimensional chunks are retained in NPZ files.

`calibrate_sphere_flow.py` uses each policy's own training-demo selection. It
samples twenty evenly spaced valid 16-step-window starts per demo, giving 100
observations for five-demo policies and 400 for twenty-demo policies. RGB comes
from decoded training MP4s, and all policies run at full resolution, matching
the rollout video scoring setup. Each observation's score averages 32 generated
chunks, with 64 noise/time draws per chunk. The empirical median and 95th
percentile are saved under `flow_calibration/`; these are in-sample training
references, not held-out guarantees of OOD detection.

`add_sphere_flow_reference.py` produces `calibrated_chunk_videos/`, retaining
the sixteen-command overlays and adding a dashed line for that policy's
training p95 under the flow trace. Original flow values and rollout outcomes
are unchanged. Every delivered MP4 is fully decoded and its frame count checked.

### Sphere continuation to 300 total epochs

`scripts/continue_sphere_training.py` creates an isolated continuation under
`toy_sphere_training_v2_300ep`. Each policy starts from its epoch-100 weights,
with the same training data and normalization. AdamW is reset, with a fresh
100-update warmup to 1e-4 and cosine decay across 200 additional epochs. This is
a second optimization stage, not a single uninterrupted 300-epoch schedule.
Initialization checks verify exact weights, statistics, and training selection.
Checkpoint names count total epochs; logged optimizer steps count the new stage.

The detached coordinator trains on GPUs 0–5, runs evaluations on GPUs 6–7,
and also uses training GPUs for evaluation after their training finishes.
Epochs 100, 150, 200, 250, and 300 are evaluated on each policy's own 5 or 20
training A/B pairs (375 rollouts). Each pair uses one fixed policy seed across
checkpoints. The existing 5 mm / 15-step success criterion, 1200-step limit,
full-resolution inference, and execution of eight actions per 16-action chunk
are retained. Results and raw videos are under `eval/epoch-NNN/<policy>/`;
`trained_pair_success.{json,csv,png}` updates as evaluations finish. These are
training-pair results, not estimates of generalization to new goals.

### Epoch-300 videos with fresh flow calibration

`scripts/run_sphere_policy_videos.py` repeats all 30 prior cases for each of the
six epoch-300 policies under `toy_sphere_rollouts_300ep_v1`. It copies only the
case definitions and unchanged camera geometry from `toy_sphere_rollouts_v2`;
rollouts, sampled actions, flow scores, and training-flow references are new.
Checkpoint paths and SHA-256 hashes are recorded in `protocol.json`.

Each policy first recalibrates on its own training demos using the same twenty
frames per demo and 32-chunk / 64-noise-draw scoring recipe. The renderer then
writes full sixteen-command overlays and the training-p95 reference together
in one pass, directly to `<policy>/calibrated_chunk_videos/`. The NPZ sidecars
retain all 32 x 16 x 7 sampled actions and per-chunk losses for every scored
frame. Every final MP4 is fully decoded; validation checks its frame count,
action horizon, mean flow, and the reference's checkpoint provenance.

### Fixed-goal opposite-hemisphere experiment

`toy_fixed_goal_sphere_v1` uses the same center `(0.45, 0, 0.35)` m and radius
0.15 m. B is fixed at `(0.45, 0, 0.50)` m. A is sampled uniformly on the opposite
hemisphere, satisfying `(A-C) dot (B-C) <= 0`; a normalized Gaussian direction
is reflected across the equatorial plane when necessary. Interior Bezier tube
variation, control, state inputs, and both cameras retain the sphere settings.

`scripts/run_fixed_goal_sphere.py` is a detached serial coordinator pinned to
physical GPU 7 by UUID. CUDA workers see that GPU as `cuda:0`; Isaac's renderer
is explicitly assigned physical device 7 with multi-GPU rendering disabled.
Only one worker runs at a time. It adopts the initial training-demo collector,
collects 20 training and 5 independent validation demos, validates their data
and MP4s, and checks physical probes and camera projection before training.

All six policy variants use the same two-stage recipe as the preceding sphere
experiment: 100 initial epochs, then 200 more with a fresh AdamW optimizer and
warmup/cosine schedule. Each checkpoint at total epochs 100/150/200/250/300 is
tested on the policy's own 5 or 20 training pairs. Final analysis includes the
same simple ID/OOD flow and action-sample plots, fresh per-policy training-flow
calibration, and 180 full-16-command overlay videos. No old policy scores or
calibration values are reused.

Unseen A and sideways-offset tests retain the fixed B. The separately labeled
both-endpoints-outside stress test uses a 0.20 m radius, with B at `(0.45, 0,
0.55)` m and A on its opposite hemisphere; it deliberately changes the goal.
Offsets are measured from the conditioned demonstration path, not asserted to
be the minimum distance from every other training trajectory.

The experiment root contains `experiment.json`, `jobs.json`, `pipeline.json`,
and `logs/`. Data and review videos are in `train_demos/` and `validation_demos/`;
weights and checkpoint success curves are in `training_100ep/` and
`training_300ep/`; analysis outputs are in `static_eval/` and `video_eval/`.
