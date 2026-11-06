# DAgger run history (archive)

This file preserves earlier run-specific requirements, investigations and operational changes. Later notes may supersede earlier notes. It is not the general analysis contract or an authoritative live status report.

Use [DAGGER_ANALYSIS.md](DAGGER_ANALYSIS.md) for the general specification, `configs/run.yaml` for the active experiment configuration, and the generated analysis overview for live results.

# Original run-specific analysis specification

Compare all 12 variants: MoE and per-resolution drivers; VIG and flow; takeover p99/p95/p90; return p50. Keep scientific settings and training selection unchanged. Report all attempts, successes, and failures separately. Annotate the first 100 successes selected for training; include overshoot successes in collection analysis.

## Existing stage definitions

Use `src/vig_dagger/stages.py` without changing its rules. Labels derive from simulator state, never from uncertainty scores. Three-step smoothing suppresses short intermediate runs; stages may regress during recovery.

| ID | Stage | Definition |
|---|---|---|
|0|Reach handle|Before opening, farther than 12 cm from handle|
|1|Handle approach|Within 12 cm; exits beyond 15 cm|
|2|Grasp handle|Within 5 cm; exits beyond 7 cm; not holding|
|3|Pull|Holding handle, including just after opening|
|4|Release|Opened, handle released, within 10 cm until first departure|
|5|Reach charger|Moving toward charger, farther than 12 cm|
|6|Charger approach|Within 12 cm; exits beyond 15 cm|
|7|Grasp charger|Within 4 cm; exits beyond 6 cm; not holding|
|8|Lift|Holding charger before picked milestone|
|9|Carry|Picked and held, outside drawer footprint|
|10|Lower, release|Over drawer, before placed milestone|
|11|Retreat|Placed, gripper within 10 cm of charger|
|12|Push approach|Placed, moving toward drawer front|
|13|Push|First closing motion after placement through closure|
|14|Hold|Drawer closed / success hold|

Four milestone latches: opened at 0.18 m; picked after 4 cm lift; placed inside drawer; closed. Geometry, grasp signals and hysteresis follow the source exactly. Successful final observations are labeled Hold. For action-weighted metrics use pre-action labels only; terminal observation is available for failure-location reporting.

## Metrics and figures

1. Successes / completed attempts, counts and 95% Wilson intervals; outcome bar plots. No interrupted attempts in denominator.
2. Pooled expert portion: sum expert steps / sum steps (length weighted).
3. Per-attempt expert portion: expert steps / attempt steps. Mean, median, quartiles, box plots. Equal attempt weighting.
4. Takeovers per attempt and per 100 policy-controlled steps. Mean, median, box plots. Zero policy exposure yields missing rate.
5. Returns per attempt and returns / takeovers. Mean, median, box plots; zero takeovers yields missing fraction.
6. Continuous expert segment duration in steps and seconds (60 Hz). Distribution within each attempt and pooled across segments. Mark returned versus episode termination; terminal segments are not assumed to represent completed recovery. Include equal-attempt mean duration summary to distinguish weighting.
7. Takeover starts per stage: per-attempt distributions and pooled counts; normalized by policy exposure in that stage. Stage with no policy exposure has undefined rate.
8. Expert steps / stage steps: per-attempt distributions and pooled proportions; absent stages are missing, not zero. Split interventions spanning stages by each action's actual stage.
9. Expert failure stage: final simulator stage, furthest stage reached, and recorded planner failure reason reported separately. Restrict expert-failure plots to explicit expert failures. Cross-tab final stage and reason. Do not interpret furthest stage as exact failure location.

Additional analysis: takeover outcomes (returned, success under expert, failure/timeout), repeated takeover within 60 steps after return (one simulated second; descriptive window), score distributions under policy/expert control and at takeover/return, numeric gate thresholds, per-attempt wall time, collection elapsed span including idle gaps, summed attempt time (parallel work, not elapsed), attempts to reach quota, training subset membership. Per-attempt and per-intervention CSVs retain identities for drill-down.

## Every-attempt video

One MP4 per completed attempt with full RGB recordings, including failures. Show every recorded RGB camera, synchronized with the pre-action observation. Bottom chart: actual VIG/flow decision score, takeover/return thresholds, event markers, expert shading, moving time cursor. Labels: round, variant, seed, success/failure reason, current stage, controller, simulated time and cumulative expert share. Playback is 60 Hz. Preserve missing-score gaps; do not interpolate unmeasured values. Include event clips around handoffs (one second before/after, clipped at boundaries).

Full recordings discarded before the retention change cannot be recovered. Keep their diagnostic metrics and list video as unavailable. Files still owned by live writers are pending, never treated as permanently missing. Render serially on CPU with one ffmpeg thread; no simulator or GPU use. Resume by skipping completed outputs. Use a separate analysis directory to avoid racing existing pipeline reports. Video index links all attempts, including unavailable/pending entries. Reports during collection are provisional snapshots.

## Outputs and execution

`python -m vig_dagger.audit --work-dir <run> [--watch]`

Writes `results/dagger_audit/REPORT.md`, JSON summaries, CSV tables, PNG plots, `videos/<round>/<variant>/<seed>.mp4`, and event clips. Watch mode refreshes as collection progresses and exits once all 12 variants reach quota and every available video is rendered. Detached execution logs to `results/dagger_audit/audit.log`. Existing collection and training continue unchanged.

## Review and organization update

The implemented sweep uses takeover p99/p95/p90 and return p50. Each of the two driver families has six collection variants. All eight initial driver trainings finish before collection; each completed collection can feed its own retraining. Full failed recordings are retained only after the retention change. Training still selects successful attempts only. Run configuration lives in `configs/run.yaml`; resolved inputs live in the run's `config.json` and collection `settings.json` files.

Start with `results/dagger_audit/README.md`. It links the compact comparison plots, collection report, individual-setting figures, all videos and failed-attempt index. It reports live counts separately from figure snapshot timestamps. Old multi-panel images are retained but omitted from the current navigation.

Review fixes:

- CLI collection progress now parses scorer aliases containing underscores correctly.
- Each detailed figure set records its generation time and attempt count in `snapshot.json`.
- After the full render pass completes, the analysis refreshes metrics and figures before exiting.
- Full MP4s and handoff clips resume independently. Clips are written to temporary files and renamed on success; a `.complete.json` marker indicates the rendering bundle completed.
- Compact comparison plots explicitly label both axes and include the fixed p50 return threshold.
- The latest analysis code is deployed by restarting only the analysis process; collection, training and evaluation continue.

Reproducible launch from the repository:

```bash
.venv/bin/python scripts/run_audit_detached.py --work-dir /cpfs02/user/caojianuo/vig-dagger-runs/opendrawer_takeover_sweep_v1
```

The launcher refuses a duplicate live renderer. Logs and PID are in `results/dagger_audit/audit.log` and `audit.pid`; the audit itself also holds an exclusive file lock.

Interpretation limits: simulated time uses the environment's 60 Hz control rate, not collection wall time. A terminal expert segment is not a completed recovery. The current expert has no grasp/lift retry loop, and its geometric holding detector can briefly report a grasp that is not secure. Failure labels therefore describe planner checks, not independently verified physical causes. Collection elapsed spans include scheduler idle gaps and exclude some startup/export overhead; use queue logs for end-to-end runtime. Wilson intervals summarize attempt proportions without accounting for any correlation between scenes or policies. Plots are descriptive comparisons, not causal evidence about thresholds.

## Recorded lift-failure investigation: MoE flow p95, seed 1000008

Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/investigate_lift_failure.py`. Outputs are `results/dagger_audit/lift_failure_1000008.json` and `.png`.

The expert controls steps 215–349; the lift action begins at step 334. The gripper close command remains -1 throughout the lift. The charger root rises approximately 11.97 mm from its pregrasp height before falling back. The finger gap shrinks from about 40 mm to 24.92 mm at observation 337, then almost zero. The hand continues upward by 116.17 mm during the lift while the charger ends about 21.09 mm sideways from its position at lift start. These traces support loss of a transient grasp during expert execution, rather than a policy handoff or commanded gripper opening.

The code calls `servo(..., name="lift")` without an explicit speed; `servo` defaults to `speed_transit`. This attempt sampled transit 0.24564 and careful 0.12193 (action-space limits, not metres/second), with three dwell steps and a lateral lift target offset (-15.08, -35.45) mm. Maximum measured hand displacement per step is 8.40 mm (~0.50 m/s at 60 Hz). Faster diagonal lifting is a plausible contributor; causality is unproven without controlled replays. Grasp stability is checked geometrically before the lift and again after the lift, so it does not stop when holding first becomes false at step 337.

Across the 41 recorded "charger dropped during the lift" failures, the object root rises 5.37–146.37 mm from the beginning of its final detected holding interval (median 37.33 mm). This is a descriptive diagnostic, not a contact-force measurement. All 41 lose the final holding predicate through the gap falling below 25 mm; that predicate uses this threshold, so this is not independent proof of the physical cause. Contact/friction forces and a controlled slower-lift comparison are needed to distinguish unstable grasp placement, friction/force limits, and lift dynamics. No expert behavior or experiment parameters were changed during this investigation.

## Expert parity clarification and outcome folders

The actual run uses `dagger.expert: demo`; worker headers confirm demo motion and in-place corrections. Compared with this repository's `expert=False` pure-planner path, the DAgger path shares sampled motion speeds, gains, dwell, via points, grasp/lift code and no-retry failure handling. In particular, the default transit-speed lift is shared, not an extra DAgger speed override. This code comparison does not establish the exact generator revision of externally downloaded demonstrations.

The expert-specific changes are broader than just a starting pose: conditional pregrasp skipping, nearest-current charger grasp orientation, a charger descent angular tolerance of 0.12 rad versus the pure planner's 0.05 rad, and conditional preservation of valid handle, placement and pushing points. If proximity/alignment checks fail it can still take the normal approach path. These differences are existing behavior; the investigation has not changed them. A stricter “same final grasp accuracy, only avoid unnecessary retreat” implementation would require an explicit experimental revision rather than silently mixing data into this sweep.

Videos now live at `videos/<driver>/<variant>/success/<seed>.mp4` and `videos/<driver>/<variant>/failed/<seed>.mp4`. Existing files were hard-linked into the outcome folders to retain working old links without duplicating video bytes. New videos and clips use the outcome folders. `SUCCESSFUL_ATTEMPTS.md` and `FAILED_ATTEMPTS.md` are separate indexes. Rendering alternates successes and failures within each setting, and rotates across settings; neither cohort must wait for the other to finish.

## Two-seed training priority revision

The user requested two active seeds and this order: MoE p99/p95 (both scores), per-resolution p99/p95 (both scores), then all p90 variants. Each group contains eight trainings; seed 0 is dispatched before seed 1 when jobs are ready, while available GPUs may run both seeds concurrently. A later group waits for the preceding group's trainings to finish. This avoids idle GPUs while preserving group order.

`configs/run.yaml` now selects seeds `[0, 1]`; seed 2 checkpoints are retained for later. Lower-priority baseline training and evaluation were gracefully paused, retaining their rolling checkpoints and completed evaluation episodes. They resume after the prioritized DAgger trainings. The existing base/seed0 driving checkpoint remains an upstream dependency and is exempt from deferral. No epochs, datasets, takeover settings, or model architectures changed. The pause manifest is `results/two_seed_priority_change.json`.

To include the third seed later, restore seed 2 in `retrain.seeds` and rebuild/restart the queue. Existing runs retain their IDs and resume checkpoints; seed 2 is deferred, not deleted.

## Baselines now; evaluate each completed policy on 100 scenes

This supersedes the earlier baseline/evaluation deferral. The user requested finishing both baseline settings now (base-only and base-plus-planner, seeds 0 and 1), and evaluating each policy as soon as its own training completes. Shared baseline jobs have priority 0. Three MoE seed-1 runs were paused to free slots for the three unfinished baseline runs; five MoE jobs continue. DAgger group priority is unchanged, and seed 2 remains deferred.

Every policy/resolution evaluation now uses 100 scenes, seeds 60000–60099. Existing raw records for later scenes are preserved but excluded from summaries and paired comparisons. Records from deferred training seed 2 are likewise excluded from the two-seed summary. Evaluation jobs depend only on their own policy's training; resources determine their precise launch time. The change manifest is `results/baseline_and_eval_revision.json`.

## Evaluate only baselines and retrained policies for now

The user deferred the collection-driver evaluations and requested only baseline/retraining evaluations now. `evaluation.evaluate_drivers` is false, leaving baseline/retrained checkpoints eligible immediately after their own training, on 100 common scenes. Active driver evaluators were gracefully stopped; training processes were preserved and adopted by the new queue. Existing raw driver evaluation records remain available but are excluded from the active summary. Restore `evaluate_drivers: true` later to resume unfinished driver evaluations. The pause manifest is `results/driver_evaluations_deferred.json`.
