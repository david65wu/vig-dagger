# DAgger planner revision: `dagger_recovery_v1`

The pure motion planner in `src/vig_dagger/expert/planner.py` is unchanged.
The DAgger collector can use `expert/dagger_planner.py`, a subclass with stricter
charger approach control and bounded alignment recovery. Other task stages and
the success criterion retain their previous behavior.

## Problem addressed

In recorded attempt `moe/vig_p95/1000096`, wrist joint 7 reached 2.8973 radians
while the expert moved from the drawer toward the charger. The old grasp selector
considered only the nearest yaw in each of two grasp directions. When neither
passed its wrist-limit estimate, it silently used the nearest one anyway.
The shared servo could also freeze an orientation with up to 30 degrees of error
and proceed; the following descent would request the original orientation again.

## Changed behavior

- Search both grasp directions across their allowed ±25-degree yaw intervals,
  including endpoints and intermediate angles. Use a vertical grasp orientation.
- Check the approximate wrist roll after in-place alignment, motion to the
  charger, and subsequent motion toward the drawer. Require at least 0.3 radians
  of predicted reserve and prefer additional reserve when turns are comparable.
  An infeasible candidate is not accepted as a fallback.
- Require the actual position and orientation tolerances for charger approach.
  Do not freeze the orientation or ignore an incomplete servo result before
  closing the gripper. Finish alignment before descent, including when the
  positional in-place shortcut applies.
- A stalled or exhausted approach can try at most two alternatives (three total
  approaches). Recovery provides up to 6 cm of local vertical clearance as
  needed, then selects a different yaw at least 10 degrees from tried targets.
  Failure to obtain clearance ends the plan. There is no unconditional retreat
  to a distant starting pose.
- Missed grasps, dropped objects, and failures in other stages are not retried
  by this revision. The ordinary episode deadline still bounds all actions.
  The DAgger gate can return control to the policy during any recovery action.

This remains a local Cartesian-IK controller with a wrist feasibility heuristic.
It is **not** a full-arm, collision-aware joint-space trajectory planner, and
predicted wrist reserve does not guarantee execution success.

## Provenance and existing experiments

New collection settings default to `planner_revision: dagger_recovery_v1`.
Explicit `dagger.planner_revision: legacy` selects the old DAgger expert.
Saved settings without a revision are interpreted as legacy by the worker.
Preparation refuses to mix revisions in an existing collection directory; use
a new work directory for new data. Existing collected data and retraining inputs
are not rewritten by this change.

Each new attempt records the planner revision and target/recovery events, with
takeover-local and episode step indices. Events are stored as `planner_events`
in trajectory attributes and `planner_events_json` in attempt diagnostics and
the episode summary. An approach retry is internal expert behavior, not a new
policy-to-expert takeover.

## Validation

`tests/test_dagger_planner.py` checks the recorded wrist-limit geometry,
infeasible-candidate rejection, alternative selection, strict alignment,
unchanged legacy servo behavior, bounded recovery, and collection revision
isolation. It uses the real Isaac Lab quaternion utilities without starting Kit.

`scripts/validate_dagger_planner.py` restores a recorded initial state, replays
saved actions up to the last takeover, measures agreement with recorded state,
then compares legacy and revised **continuous expert** execution. The source
data is read-only; outputs must be separate. Its results are not live-gate
DAgger success rates. See the validation output's JSON for replay agreement,
task outcome, failure reason, and recovery events.

Validated on 2026-10-30 with `moe/vig_p95` scenes 1000096, 1000048, 1000068,
and 1000116. All eight replay prefixes matched the recorded takeover states
exactly. Legacy reproduced 1000096's abort at step 386; the revision completed
the task at step 780. Both versions completed the other three continuations.
No retry was needed in these four revised executions; the retry limit was
verified in unit tests. Raw evidence and the comparison table are in
`/cpfs02/user/caojianuo/vig-dagger-runs/dagger_planner_validation_20261030_072156/REPORT.md`.
