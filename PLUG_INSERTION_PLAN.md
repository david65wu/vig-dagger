# Plug insertion experiment plan

Updated: 2026-11-02 (Asia/Shanghai).

## Objective

Study whether the policy's flow-loss score separates familiar/in-distribution
(ID) observations from out-of-distribution (OOD) observations, and inspect how
the sampled action distribution changes. Compare the effects of demonstration
count and training duration. Keep the analysis focused on flow loss, action
samples, and rollout success; VIG is not the focus of this experiment.

## Task and environment

- Use the main-branch `vig-dagger` PlugCharger scene in ManiSkill/SAPIEN.
  Baseline commit: `5d2ab39603bbb1269b5129ce6e9705df52ae7add`.
- The charger is already physically grasped when a recorded episode starts.
  Pickup and movement into the starting pose happen during initialization and
  are excluded from the demonstration.
- Start near the socket, with bounded variation in position and orientation.
  Current preview settings, in the socket goal frame:
  - Axial standoff: 3–8 cm from the fully seated charger pose.
  - Lateral offsets: independently sampled within ±1.5 cm on each lateral axis.
  - Roll, pitch, and yaw: independently sampled within ±8 degrees.
  These are requested pose ranges; log the actual settled starting poses too.
- Keep the baseline robot, charger, socket geometry, contact physics, lighting,
  camera placement, and socket randomization. Socket variation remains ±3 cm
  laterally, ±2 cm vertically, and ±5 degrees in yaw; hole clearance remains
  4 mm per side.
- Record alignment, insertion, and the baseline closed-gripper terminal hold.
  No artificial attachment holds the charger.
- Preserve the baseline success condition: position error ≤5 mm and orientation
  error ≤0.2 rad. For collected demos, additionally require a stable grasp and
  successful insertion throughout the last ten steps.
- Control: 20 Hz, absolute joint-position targets plus gripper command
  (`pd_joint_pos`, eight action dimensions).

## Policy observations and architecture

**Training and policy inference use exactly these two cameras, in this order:**

1. `hand_camera` — wrist-mounted camera.
2. `side_camera` — close-up socket view.

The third-person camera may remain in recordings and review videos, but must
not be passed to the policy. Existing preview data already contain both required
views, so this camera choice does not require recollecting the previews.

Use full-resolution inputs from the 256×256 recorded streams, with no seven-level
resolution degradation and no MoE. Any standard encoder resize/crop preprocessing
must be documented and identical across all runs. Retain robot proprioception
(seven arm joints and two gripper joint positions), the dense flow-matching
policy, and the 16-action prediction horizon.

Before training, create an insertion-specific configuration with the two-camera
feature list and `policy.cameras = 2`; make the inference image loader use the
same order. Do not change shared defaults used by other experiments. This
document records the camera decision. The insertion training configurator now
implements it and checks the saved checkpoint's camera list during preflight.

## Demonstration collection

- Target: **500 successful insertion demonstrations total**.
- Use deterministic scene and initialization seeds and retain their metadata,
  including actual starting poses, socket pose, motion stages, and success.
- Keep unsuccessful attempts separate from the training dataset.
- Build deterministic nested subsets: **50 ⊂ 200 ⊂ 500 demos**. Select using a
  fixed shuffled ordering, rather than ordering by task difficulty or outcome.
- The five successful previews can count toward the 500 if the preview task
  settings are retained. Any change to those settings must be recorded before
  combining datasets.
- Retain raw trajectories, converted training data, and camera MP4s. Verify
  action/observation alignment, camera order, grasp state, and terminal success.

## Training duration and comparison

Train one continuous run for each demo count and retain checkpoints along the
way. Use early, middle, and late snapshots from each run for the main comparison:
three demo counts × three training stages = **nine comparison checkpoints**.

The working checkpoint grid is **10, 25, 50, 100, 200, and 300 epochs**. This is a
sampling grid, not a claim that a particular epoch is undertrained or overfit.
Select representative stages using training curves and a fixed set of ID
rollouts, including exact demonstration starting states:

| Stage | Evidence to look for |
|---|---|
| Undertrained | Loss and familiar-task performance are still improving substantially. |
| Moderate | Good ID success and performance approaching a plateau. |
| Late / potentially overtrained | Continued fitting beyond the plateau; call it overfit only if held-out performance deteriorates. |

Record epoch, optimizer-update count, batch size, dataset size, learning-rate
schedule, training seed, and elapsed time for every checkpoint. The same epoch
count entails different update counts for 50, 200, and 500 demos. Keep all other
training choices fixed for this comparison.

Use checkpoints from the same training trajectory when comparing durations;
do not silently substitute separately trained short runs with different
learning-rate schedules. Exact stage selections and whether training needs to
extend beyond the working grid remain to be determined from the learning curves.
Do not select checkpoints using the final OOD evaluation results.

## Simple evaluation and flow calibration

First verify closed-loop insertion success on exact demonstrated initial states.
Also evaluate fresh starts sampled within the collection bounds, reported
separately from exact training starts. This distinguishes memorized starts from
generalization within the intended task distribution.

Proposed OOD cases vary starting translation and/or orientation beyond the
collection bounds while retaining a valid physical grasp and reachable task.
Choose and document the exact OOD ranges and rollout counts before evaluation;
they are not fixed yet. Use shared physical starting states across checkpoints.

For each evaluated checkpoint:

1. Recalibrate its flow score using observations from its own training subset,
   with the same two cameras, preprocessing, scoring settings, and sampling
   cadence used during evaluation. Never reuse another checkpoint's calibration.
2. Use the existing `vig-dagger` flow-loss self-consistency score: sample an action
   chunk from the policy and evaluate flow-matching loss against that generated
   chunk over random noise/time draws. The current baseline uses 512 loss draws
   and scores the first sampled chunk. This differs from supervised loss against
   an expert action and is not a likelihood or a calibrated failure probability.
3. Preserve raw scores and map each score to its training-reference percentile:
   `percentile(s) = 100 * count(training_score <= s) / N`.
4. Plot flow-loss percentile boxplots for each ID/OOD case, using scored states
   across **whole rollouts**, not only the reset state. State the scoring cadence,
   number of rollouts, termination rule, and number of scored observations.
   Pooled observations give longer rollouts more weight; label that explicitly.
5. At selected shared ID and OOD observations, sample action chunks repeatedly
   and plot the action distributions. Use the same sample count and plotting
   scales across policies. Inspect whether OOD actions are concentrated yet
   incorrect, and whether their flow scores remain similar to ID scores.
6. Produce rollout MP4s with projected action trajectories and a bottom row
   showing raw flow score and its training percentile. Draw **all 16 actions**
   of the displayed chunk. Because these are joint targets, use robot forward
   kinematics and camera calibration for projection; label these as commanded
   kinematic paths rather than guaranteed executed trajectories.

Report rollout success alongside these plots. A narrow action distribution by
itself is not evidence of overconfidence; interpret concentration together with
behavioral failure or incorrect actions and a low/ID-like flow score.

## Current status and artifacts

- Five preview demonstrations completed successfully after setup debugging.
  The final collection succeeded on all five tested scene seeds (100–104).
- Charger grasp remained valid at every recorded step. Final position errors
  were approximately 2.0–3.9 mm. There are 414 action/observation pairs total.
- Preview root:
  `/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_preview_v1/`
- Combined preview: `videos/all_5_demos.mp4`.
- Individual previews: `videos/demo_00.mp4` through `videos/demo_04.mp4`.
- Raw data: `raw/preview.h5` and `raw/preview.json`.
- Training-format data: `dataset/`.
- Checks and provenance: `validation.json`, `initial_states.json`,
  `provenance.json`, and `source/`.
- Implementation worktree:
  `/cpfs02/user/caojianuo/vig-dagger-charger-insertion/`.

## Execution and GPU allocation

The preview job is finished. The full collection and automatic training pipeline
were launched on 2026-11-02 following the user's instruction to start. GPU 7 was
the sole GPU used for the previews; the full pipeline uses GPUs 6 and 7.

Run root: `/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_v1/`.
The detached coordinator is `scripts/run_insertion_experiment.py`; its log and
current phase are `coordinator.log` and `pipeline.json` in the run root. Collection
uses ten independently recorded parts (nine of 50 demos and one of 45) plus the
five retained previews. Each part retains unsuccessful attempts separately.

After collection and checks, the pipeline creates the shuffled nested subsets
and runs an eight-update training/resume/inference preflight. It then trains the
500-demo policy on GPU 6, and the 200-demo followed by the 50-demo policy on GPU 7.
Training uses batch size 64, learning rate 1e-4, weight decay 1e-4, 100 warmup
updates, cosine decay to a 1e-4 learning-rate ratio, seed 0, and two loader workers.
All 500 successful demos are available for training; no demonstration holdout is
subtracted from the requested counts. Closed-loop validation is a subsequent step.

Each epoch traverses eligible full-chunk samples once (`k_passes=1`). The worker
computes per-epoch update counts including each loader worker's partial batch,
so scheduled epoch checkpoints correspond to complete passes in uninterrupted
runs. Rolling checkpoints are saved every 1,000 updates, keeping the newest two;
the six named epoch checkpoints are retained. Resume restores model, optimizer,
scheduler, and normalization, with the library's next-data-epoch resume semantics.

The current coordinator ends at `training_complete_analysis_pending`; calibration
and rollout analysis follow after training and are not represented as complete
by that status.

### Analysis launched on 2026-11-03

Collection and all three 300-epoch training runs finished on November 2 at 18:12
Shanghai time. The approved analysis is now running separately at
`/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_analysis_v1/`, with
`pipeline.json` and `coordinator.log` tracking its progress. Its coordinator
automatically proceeds through calibration, rollouts, MP4 validation, summary
tables, and plots; it does not stop after evaluation.

- Evaluate all 18 saved checkpoints, with 10 rollouts in each of five conditions:
  familiar training starts, new ID, position OOD, orientation OOD, combined OOD.
- Position OOD uses signed 3–5 cm socket-frame lateral Y offsets; orientation
  OOD uses a signed 15–25 degree pitch or yaw rotation. Other requested starting
  components retain the collection ranges. The four new-start conditions share
  socket seeds and paired initial components.
- Reuse 50 prepared physical snapshots across checkpoints. Familiar starts use
  recorded demonstration states. A single physics substep rebuilds contact
  impulses after restoration; the observed maximum charger displacement in
  preparation was 0.88 mm. This substep is excluded from rollout scoring.
- Recalibrate using every training demo in each checkpoint's own subset, taking
  raw recorded observations every eight action steps. Use 512 noise/time draws
  per flow score. Pilot-only one-demo calibration is kept in `pilot/` and never
  used for the final figures.
- Execute eight actions per replan, score every eight steps, cap at 300 steps,
  and require ten consecutive successful, grasped steps for rollout success.
- Save 128 initial action chunks per rollout. Plot first-command TCP translation
  and rotation distributions via forward kinematics; retain complete chunks.
- Save the first success and first failure video per condition/checkpoint when
  available. Both camera views show all 16 FK targets; the bottom row displays
  raw flow and a training-percentile trace.
- Pilot validation replayed recorded actions successfully and confirmed expert
  recovery on all five representative starts. The orientation pilot required
  the existing RRT fallback for alignment. All five 50-demo epoch-300 policy
  pilot rollouts failed; these are pilot observations, not full success-rate
  estimates. Raw pilot artifacts remain separate from final evaluation outputs.

The final outputs include `flow_percentiles.png`, `flow_raw.png`, six action
distribution figures, `success_rates.csv`, `video_index.json`, and `index.html`.

The user has authorized using the two idle GPUs and additional idle GPUs when
available. At the 2026-11-02 12:59 Shanghai-time check, the existing OpenDrawer
queue is configured for GPUs 0–5, leaving **GPUs 6 and 7** for insertion. Use those
as the primary allocation. Extra GPUs can accelerate collection and evaluation
when available, but check the existing queue as well as instantaneous utilization
before assigning work; avoid competing with its pending jobs. Do not stop or
reconfigure unrelated jobs to obtain GPUs.

After allocation, run collection/training as detached, logged jobs that survive
SSH disconnection, with checkpoint/resume support for training. Estimate runtime
from measured collection throughput and training updates per second before
reporting a full-run estimate. Preserve the unrelated OpenDrawer queue and jobs.

### Preliminary time estimate

Budget approximately **8–12 hours** for collection, three training runs through
300 epochs, calibration, and an initial set of ID/OOD plots and videos on two
GPUs. This assumes one training seed per demo count and a modest evaluation set;
the rollout counts and exact OOD ranges remain to be fixed.

- Remaining collection, conversion, and checks: roughly 30–60 minutes.
- Training wall time with the 500-demo run on one GPU and the 200-/50-demo runs
  on the other: roughly 5–8 hours, allowing for checkpointing and shared I/O.
- Calibration and initial rollout analysis: provisionally 1–3 hours, depending
  on scoring cadence and evaluation counts.
- Allow some additional setup time for the two-camera training/evaluation path.

Basis: the five previews averaged 82.8 control steps. With full 16-step chunks,
batch size 64, and no withheld training demos, the projected update counts at
300 epochs are approximately 15,900 / 63,600 / 159,000 for 50 / 200 / 500 demos.
The previous two-camera dense sphere run measured about 9.5 updates/second;
insertion throughput has not yet been benchmarked. The preview collection and
export took about 34 seconds for five demos after library startup. These are
extrapolations, not measured full-run timings. Recompute the ETA after collecting
the full dataset and observing initial training throughput. Extra GPUs mainly
shorten collection and evaluation; the longest single-GPU training remains the
critical path.

## Follow-up: seven-resolution MoE experiment

The user requested the same insertion experiment with a 7-level MoE policy on
2026-11-03. Training is running in
`/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_moe7_v1/` and its analysis
will be written to
`/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_moe7_analysis_v1/`.

- Reuse the exact dense experiment's 50/200/500-demo selections. No recollection.
- Retain hand and side cameras, proprioception, action normalization, 16-action
  horizon, seed 0, optimizer, learning-rate schedule, update counts, and epoch
  checkpoints at 10/25/50/100/200/300.
- Use the main-branch MoE settings: eight experts, top-2 routing, resolution
  category in router slot 0, and mutual-information auxiliary loss weight 0.0005
  with 100-update warmup and 0.99 momentum.
- Uniformly sample one resolution factor per training sample from
  `[1, 2, 4, 8, 16, 32, 64]`, shared across its two camera images. The processing
  code downsamples and restores image size and supplies the level index to the
  policy. Seven levels refers to resolution categories, not expert count.
- Evaluate at full resolution (factor 1, category index 0), including flow
  calibration and all action-distribution samples. This comparison changes both
  the model architecture and its training-resolution distribution relative to
  dense-full training; equal total updates do not imply equal full-resolution
  training exposures.
- Recalibrate each MoE checkpoint on its own training observations. Dense flow
  reference distributions are not reused.
- Use byte-identical copies of the dense experiment's 50 physical starting-state
  snapshots and the identical success rule, action execution stride, scoring
  cadence, and 512 flow-loss draws. Evaluate 18 checkpoints and 900 rollouts.
- Produce the same flow, action-distribution, success-rate, and annotated-video
  outputs, plus a dense-versus-MoE success-rate comparison. Videos retain the
  expanded percentile scale with ticks every 20 points.

The detached `scripts/run_insertion_moe.py` coordinator runs the full sequence:
configuration checks, preflight, training, calibration, rollouts, video validation,
plots, and comparison. Initially GPU 6 trained 500 demos and GPU 7 trained 200,
with 50 demos queued. The user subsequently paused the other experiment and
authorized GPUs 0–7. The coordinator now runs with `--all-gpus`: GPU 0 trains
50 demos, GPU 6 continues 500 demos, and GPU 7 continues 200 demos. The two
existing training workers were adopted without restarting. GPUs 1–5 calibrate
and evaluate saved checkpoints as they become available; each training GPU
joins that evaluation pool when its training finishes. Idle evaluation lanes
wait for checkpoints. Checkpoint readiness requires a training progress report
published after the checkpoint save returns, avoiding partially written files.
No separate user instruction is needed to advance from training to analysis.

Preflight passed training/resume, finite inference at all seven factors,
512-draw full-resolution flow scoring, and 128-chunk sampling. The training
worker verifies normalization matches the corresponding dense run exactly.

## Expanded dense-policy video gallery

The user requested more videos with plots for every dense checkpoint. The
detached `scripts/expand_insertion_videos.py` coordinator fills a gallery of
50 videos per checkpoint (10 per condition), 900 total, under the original
dense analysis directory's `full_rollout_videos/`. The 141 existing recordings
are retained; the other 759 videos use fresh policy rollouts from the saved
physical starts. Their flow scores are measured again and converted with the
unchanged checkpoint's original training calibration. Each video includes both
cameras, all 16 action targets, raw flow, and the percentile curve with ticks
at 0/20/40/60/80/100. Gallery entries distinguish original and fresh rollouts.

A recorded-action replay check showed small charger contact differences, so
original flow traces are not overlaid on replayed footage. Fresh rollout traces
and outcomes are saved separately; original evaluation summaries remain intact.
Five video workers share evaluation GPUs 1–5 with the MoE analysis, leaving the
training GPUs 0, 6, and 7 dedicated to their current training runs.

The user also requested corresponding updates to the other plots. The detached
`scripts/update_insertion_video_analysis.py` waits for all videos, validates the
selected traces and calibration, and regenerates success, raw-flow, percentile,
and action-distribution plots in `full_rollout_videos/analysis/`. Exactly one
rollout per checkpoint and case is selected, matching the gallery (141 retained
recordings plus 759 fresh rollouts). There is no double-counting of original and
fresh trials. The main analysis index points to these updated plots, with the
original index and figures preserved. `analysis_selection.json` directs the
later dense-versus-MoE comparison to the same revised dataset.

## MoE: full videos and calibrated VIG

The user requested full MoE rollout videos and calibrated VIG alongside flow.
The revised analysis directory is
`/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_moe7_flow_vig_eval_v1/`.
The previous flow-only MoE evaluation is retained separately. Training workers,
checkpoints, and training hyperparameters are unchanged.

- Evaluate all 50 matched physical starts for every one of 18 checkpoints and
  save every episode: 900 freshly evaluated videos, 10 per condition per checkpoint.
- Compute VIG using `vig_dagger.scoring.VIGScorer(num_samples=32)`: common-noise
  samples across factors 1/2/4/8/16/32/64; per degraded factor, average the
  per-cell 1D Wasserstein distance over normalized 16-by-8 action chunks; take
  the maximum over the six degraded factors.
- Calibrate VIG independently for each checkpoint on all its own training-demo
  observations at stride 8, using precisely the same frames as flow calibration.
  Percentiles are empirical ranks, `100 * searchsorted(sorted_reference, score,
  side="right") / N`. Previously computed flow references are reusable only
  after verifying the checkpoint SHA256; VIG references are newly computed.
- Score flow and VIG together every 8 action steps. The robot still executes
  full-resolution chunks generated by flow scoring. VIG uses a forked RNG
  context, with a runtime check that CPU/GPU random states are preserved.
- MP4s contain hand and side cameras, all 16 action targets, a flow percentile
  plot, and a VIG percentile plot underneath. Each panel includes the raw score
  and ticks at 0/20/40/60/80/100. Output size is 512 by 660 pixels.
- Regenerate success, flow, and action-distribution figures from these same
  900 episodes; add `vig_raw.png` and `vig_percentiles.png`. Each condition's
  box pools all scored observations from its 10 full rollouts. Longer episodes
  contribute more observations. Validate both score-to-percentile mappings and
  the matching calibration frame indices before plotting.
- The detached all-GPU coordinator adopts the existing training workers and
  evaluates new checkpoints as they become available. Prior flow-only workers
  finish before their GPU is reused. The new evaluator resumes completed cases
  and calibration, including the two preflight cases, without duplicating them.

## Every-frame evaluation for dense and MoE

The user subsequently requested scoring on every frame, including the dense
policies. `scripts/run_insertion_every_frame.py` runs both evaluations through
one detached queue on GPUs 0–7 and adopts the remaining MoE training worker.
The evaluation roots are `charger_insertion_dense_every_frame_v1` (flow) and
`charger_insertion_moe7_every_frame_v1` (flow and VIG), under the runs directory.
Each contains 18 checkpoints and 900 full rollout videos. The shared queue's
status, logs, and preflight live in `charger_insertion_every_frame_pipeline_v1`.

Both score stride and calibration stride are 1: every pre-action observation
gets a newly computed score, and each checkpoint is calibrated on every
pre-action frame of its own training demonstrations. VIG retains 32 samples per
level; flow retains 512 loss draws. Every-frame calibration is recomputed in the
new directories; cached stride-8 references are not used in production.

The controller still replans every 8 steps. Intermediate flow samples are used
only for scoring, and their RNG is isolated. All 16 targets from the active
command chunk remain overlaid. Traces record replan steps and chunks, allowing
the plot generator to verify both one score per frame and exact correspondence
between executed actions and the chunk selected at the last replan. Real-policy
cadence tests exercise dense and MoE with 16 frames, 16 scores, and two replans;
these tests use temporary artifacts and prior references only to test cadence,
not to publish final calibrated evaluation results.

Success, raw-flow, flow-percentile, action-distribution, and (for MoE) raw-VIG and
VIG-percentile plots are regenerated from the same new episodes. The queue also
produces the dense-versus-MoE success comparison after both variants finish.

## VIG after averaging each 16-action chunk

The user approved averaging the 16 time steps of each sampled normalized action
chunk before computing VIG. `VIGScorer(32, temporal_average=True)` reduces the
sample tensor from (7, 32, 16, 8) to (7, 32, 8). It then computes 1D Wasserstein
distances over the 32 samples for each action dimension, averages over the eight
dimensions, and takes the maximum over six degraded resolutions. Shared noise
across levels is retained. The default scorer continues to support the original
metric.

`scripts/run_insertion_mean_chunk_vig.py` runs the new experiment detached across
GPUs 0–7. Results are in `charger_insertion_moe7_mean_chunk_vig_v1`. Each checkpoint
gets a new VIG calibration over all its training-demo frames. The unchanged
flow reference is reused after verifying the checkpoint hash and stride 1.
All 900 episodes are evaluated with flow and the new VIG on every video frame,
with action replanning every eight steps. Videos explicitly label VIG as
"chunk mean". All score, success, and action-distribution plots are regenerated
from these episodes, with `vig_raw.png` and `vig_percentiles.png` using the new
reference distributions. Earlier evaluations remain available for comparison.

Preflight checks temporal cancellation, retention of the sample distribution,
and averaging distances over action dimensions. A real-checkpoint smoke test
checks 16-frame scoring, two replans, RNG preservation, percentile mapping, and
MP4 validity. Its small temporary reference is test-only; production calibration
always uses every frame of the checkpoint's full training subset.

## Three scores in the same MP4

The user requested flow, original VIG, and chunk-mean VIG together. The detached
`scripts/run_insertion_three_scores.py` pipeline writes to
`charger_insertion_moe7_three_scores_v1`. It evaluates all 18 checkpoints and
900 episodes with every-frame scoring. Each 512-by-840 MP4 contains both cameras
and all 16 action targets, followed by flow, original VIG, and chunk-mean VIG
panels, each with raw values and independently calibrated percentiles.

Both VIG definitions are computed from the exact same action sample tensor in
one policy sampling call per frame. Original VIG compares the 16-by-8 cells;
chunk-mean VIG first averages time, then compares the eight action dimensions.
The existing every-frame reference for each score is reused only after matching
checkpoint SHA256, stride, VIG definition, and calibration frame indices. The
new rollout traces retain both VIG scores, per-resolution gaps, and percentiles.
Preflight validates shared sampling, frame alignment, all three percentile
mappings, and the MP4 dimensions before releasing the full eight-GPU queue.

The user additionally requested raw-value y-axes. Each episode now has two MP4s:
the percentile version in `videos/`, and a raw-axis version in `videos_raw/`.
The raw version uses the same recorded camera/action footage and score traces;
it does not rerun the policy. Its three linear y-axes start at zero and have
fixed, separately chosen limits for flow, original VIG, and chunk-mean VIG
across all 50 episodes of that checkpoint. Limits include every observed score
with a margin and are saved in `raw_axis_limits.json`. Raw rendering runs after
the checkpoint completes so limits are known. Gallery pages expose both versions.
The complete evaluation therefore produces 900 episodes and 1,800 MP4s.

The user then requested more flexible raw y-axis scales because checkpoint-wide
ranges made low-valued episodes hard to read. Raw ranges now fit each metric in
each individual rollout, starting at zero and covering the observed maximum
with a 5% margin plus readable tick rounding. They remain fixed within a video.
`scripts/rescale_insertion_raw_videos.py` regenerates the 900 raw videos from
the saved percentile footage and score traces on eight CPU workers, writing
`videos_raw_rollout/` and `raw_axis_limits_rollout.json`. Gallery links and the
video index switch to these versions as checkpoints complete. Original
checkpoint-wide videos remain on disk. Status is in `raw_rescale.json`.


## EMA video visualization (alpha = 0.1)

Apply causal EMA independently to flow, original VIG, and chunk-mean VIG, for both raw scores and their existing calibrated percentiles. Initialize with the first frame and reset per rollout: y[t] = 0.1*x[t] + 0.9*y[t-1]. This is display-only smoothing; retain original traces, calibration, videos, and aggregate statistics. Raw axes fit each episode's smoothed signal and remain fixed during playback. Percentile axes remain 0–100. Render all 900 MoE three-score rollouts in both formats (1,800 MP4s), using eight CPU workers. Gallery: charger_insertion_moe7_three_scores_v1/ema_01/index.html. Status: ema01_status.json. Preview verified six EMA recurrences, source SHA256 preservation, and both 144-frame videos at 512x840, 20 fps.

## Lowest-resolution-reference VIG (November 4)

Add a fourth video signal: compare factor 64 (the lowest-resolution level) with factors 1, 2, 4, 8, 16, and 32. For each comparison, average the per-cell 1D Wasserstein distances over the original 16 x 8 action cells, then take the maximum over the six comparisons. Each resolution uses 32 sampled chunks with shared initial noise. This variant does not average the chunk first.

Source: `charger_insertion_moe7_three_scores_v1`; output: `charger_insertion_moe7_lowest_reference_v1`. The original files contain starting simulator states and executed actions, but not per-frame observations or per-resolution samples. Recover observations by replaying only the stored actions, with no new policy decisions. Validate each frame against the recorded TCP trajectory, plug geometry, and both original VIG definitions regenerated with the original sampling seed. Require the same success outcome. Record numerical discrepancies in per-episode audit JSON; fail on drift beyond the documented tolerances. Preserve the original camera footage, actions, flow values, both existing VIG signals, and success labels in the derived videos/data.

Calibrate the new score on every pre-action observation in each checkpoint's own training-demo subset, checking the reproduced original VIG scores as well. Use the new empirical CDF for its percentile. Render all 900 episodes in raw and percentile versions (1,800 MP4s), with four panels, EMA alpha=0.1 independently applied to raw and percentile curves, and per-episode raw axes. Summary boxplots pool unsmoothed frame scores and show median, IQR, and 5th–95th percentile whiskers. Run the detached calibration/replay queue across the eight available GPUs after preflight verification. No retraining and no new policy rollouts.

Preflight finding: exact offline rescoring is not possible from the existing files alone. A full 144-frame saved-action replay of demos50_epoch300 / case22 preserved success and had maximum TCP drift 6.76 micrometres, plug-distance difference 4.43 micrometres, and plug-angle difference 0.000145 rad. Regenerated original and chunk-mean VIG differed by up to 0.000177 and 0.000172, respectively, failing the strict score-equivalence check. No batch replay/video rendering has been authorized by this preflight. The independent training-demo calibration queue is running; its first 16 frames reproduced both existing VIG definitions exactly. A methodological choice is pending: use saved-action replay and recompute all four signals together (report numerical differences), or collect new rollouts while retaining observations/samples. The original data and MP4s remain unchanged.

## Fresh four-score evaluation (user decision, supersedes replay)

The user chose fresh policy rollouts because of concern about replay divergence. Use `charger_insertion_moe7_four_scores_v1` as a separate output root, with the same 50 starting cases per checkpoint and 18 MoE checkpoints (900 fresh rollouts). The policy chooses actions normally, replanning every eight steps; every control-step observation receives flow, original full-reference VIG, chunk-mean VIG, and lowest-reference VIG. The three VIG variants share one 7 x 32 x 16 x 8 sample tensor. All 16 action targets remain projected on both cameras.

Continue the already-running lowest-reference training calibration in `charger_insertion_moe7_lowest_reference_v1`; adopt those child workers when handing scheduling to the fresh-evaluation coordinator. Reuse the existing flow/original-VIG/chunk-mean calibration only after checking checkpoint hashes and exact training-frame indices. Copy each completed new calibration into the fresh run. Render raw and percentile MP4s with four stacked panels and EMA alpha=0.1. Update success, flow, all VIG raw/percentile boxplots, and action distributions using only the fresh rollouts.

Retain per-frame pre-action qpos, lossless hand/side RGB, camera intrinsics/extrinsics, all seven-resolution normalized VIG sample sets, flow sampled chunks, sampling seeds, and executed actions in compressed HDF5 files. Verify all four percentile mappings and recompute each VIG definition from saved samples at the first, middle, and last frame of every episode. This supports future score variants without simulator replay. Prior replay experiments and original videos remain separate.
