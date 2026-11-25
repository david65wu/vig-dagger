# Updates to a running experiment

For an agent on a machine that started the experiment before a `git pull`: read the entries below that are newer
than the commit this machine started from, and do what each says **for this machine**. The humans who own the
experiment approve every step written here, deletions included; do nothing beyond them, and stop and ask if
something doesn't match what an entry describes. Then start the queue again (AGENTS.md, step 4) and report what you
did. Entries are newest first.

## 2026-11-24 (4): failed attempts now count in the adaptive threshold; collect round 1 again with it (a test)

Round 1 of a 50-demo run climbed from p80 to about p98, where 4 % of the attempts succeeded: the MoE fails in states
that look like its training frames, so its failed attempts pushed the threshold up too. The new option
`iterate.failed_last_stretch` (README, Iteration mode, "Failed attempts"; on in `configs/plugcharger.yaml`) changes
one thing in the update: in a **failed** attempt, the policy steps after its last take-over (the policy drove into the
failure without the gate firing; all its policy steps if the expert never took over) count as if their VIG were at
the threshold the attempt ran with. Successful attempts count as before. `threshold_log.jsonl` and the episode lines
record how many steps were counted that way (`n_last_stretch`, `adapt_last_stretch`).

Also: `results/figures/distributions.png` now shows the expert steps alone (dash-dot, with their mean percentile;
`iterate_summary.json` has `expert_mean_pct` per round).

**The machine running the DAgger rounds** (round 1 collected with the earlier rule, not finished):
1. Stop the queue and its jobs: `vig-dagger stop -c configs/run.yaml`.
2. If any of these exist, stop here and ask (round 1 went further than expected):
   `<work_dir>/policies/retrain/round1_sd0_vig_adaptive_random_move/`, `<work_dir>/data/datasets/round2_train/`,
   `<work_dir>/policies/moe/round2/`.
3. In `configs/run.yaml`, under `iterate:`, add `failed_last_stretch: true` (as in `configs/plugcharger.yaml`).
4. Move round 1's collection aside; **never delete it**. With `C=round1_sd0_vig_adaptive_random_move`:
   `mv <work_dir>/dagger/$C <work_dir>/dagger/$C.before_failed_rule`, and the same suffix for
   `<work_dir>/videos/$C` and every `<work_dir>/results/collections/$C*` file, where they exist. Keep everything
   else (the round-1 MoE, its reference, any fixed-threshold collections) as it is.
5. Start the queue again (AGENTS.md step 4). Round 1's collection starts again from p80 on the same scenes, with the
   new rule (`dagger_prepare` prints "failed attempts' last policy stretch at the threshold").
6. Report after about 50 and 100 attempts: `vig-dagger report -c configs/run.yaml`, then send
   `results/figures/thresholds.png` and `distributions.png`, successes / attempts so far, and the threshold now.
   Do not stop after round 1: the queue continues into round 1's retraining and the later rounds as usual.

**A machine running the planner-only comparison** (`configs/planner_reference.yaml`): nothing to do (it collects no
DAgger data).

## 2026-11-24 (3): the planner-only comparison runs on another machine; keep it off here

The comparison added in the entry below (dense policies on as many planner demos) runs on another machine
(`configs/planner_reference.yaml`). `configs/plugcharger.yaml` now has `iterate.planner_baselines: false`.

**A machine running the DAgger rounds:** make sure `configs/run.yaml` has `iterate.planner_baselines: false` (or no
such line). If an earlier `update` set it to `true`: stop the queue (`vig-dagger stop -c configs/run.yaml
--queue-only`), set it to `false`, and start the queue again (AGENTS.md step 4; running jobs continue), so it drops
the `collect_baseline_*`, `planner_dataset_*` and `train/planner/*` jobs. Leave whatever they already wrote
(`data/raw/planner_baselines/`, `data/datasets/planner_*`, `policies/planner/`) where it is and say so in your report;
never delete it. If the queue is stopped, leave it stopped.

## 2026-11-24 (2): videos of DAgger attempts, the VIG distribution figure (and a planner-only comparison, off here)

- **Videos:** after each round's collection the queue renders 20 successful and 20 failed attempts, spread over the
  collection, into `<work_dir>/videos/<c>/` (job `videos_<c>`). `vig-dagger videos -c configs/run.yaml` renders them
  at any time, also during a collection.
- **`results/figures/distributions.png`:** per round, the VIG histograms of the round MoE's training frames and of
  its collection's policy steps (and policy + expert steps), with their overlap. Every `vig-dagger report` refreshes it.
- **Planner-only comparison** (`iterate.planner_baselines`): dense policies on as many planner demos, evaluated on the
  same scenes. It runs on another machine (entry (3) above): keep it off here.

Nothing that exists changes: the base demos, the round datasets, the MoEs, the collections and their thresholds stay
as they are; the new jobs only add outputs. The smoke test now ends with `[queue] finished: 45 of 45 jobs done`.

**A machine running this experiment:** after `git pull`, copy the `videos` section from `configs/plugcharger.yaml`
into `configs/run.yaml` if run.yaml is an older copy (the defaults are the same, so this is optional). If the queue is
running, restart it so it adds the video jobs (`vig-dagger stop -c configs/run.yaml --queue-only`, then the launch
command of AGENTS.md step 4; the running jobs continue). If the queue is stopped, leave it stopped: only start it when
the humans say so. Either way run `vig-dagger videos -c configs/run.yaml` and `vig-dagger report -c
configs/run.yaml` once, and send `results/figures/distributions.png` and how many videos each collection has.

## 2026-11-24: branch plugcharger-iterate-adaptive: DAgger iteration with an adaptive threshold (a new experiment)

This branch runs a different experiment (README, "Iteration mode"): 50 base demos collected here with the fast
expert's motion, then 4 rounds of DAgger (100 / 200 / 400 / 800 successful episodes), each driven by a MoE trained
on the data so far, with one threshold for take-over and hand-back that adapts after every attempt (round 1 from
p80), the 'fast_maniskill' expert, every attempt saved, and a dense policy retrained and evaluated after each round.

- **A machine that runs another branch's experiment:** leave it and its work directory as they are. Set this
  experiment up as a new one in a **new** work directory (default `~/vig-dagger-runs/plugcharger_iterate_adaptive`),
  following AGENTS.md from step 1 (`git checkout plugcharger-iterate-adaptive`).
- **Task boards are not used:** the rounds follow each other on one machine.

## 2026-11-23: the sweep now uses the 'fast_slowed' expert (a new experiment)

The hand-back sweep (take over above p90, hand back below p70, p75, p80, p85) is repeated with another expert:
`dagger.expert: fast_slowed`, the earlier maniskill DAgger expert at fixed speeds 0.7 / 0.625 (README, Part 3). The
experiment's name and default work directory are now `plugcharger_return_sweep_fast_slowed`, and it has no
reference (`reference.dirs: []`).

- **A machine that ran or is running the earlier sweep (`expert: demo`):** that sweep is finished or abandoned; do
  not continue it with this code. Stop its queue if it still runs (`vig-dagger stop -c configs/run.yaml`), and leave
  its work directory as it is (never delete it). Then set this machine up for the new experiment as if new: copy
  `configs/plugcharger.yaml` to a new `configs/run_fast_slowed.yaml`, set a **new** `work_dir` (default
  `~/vig-dagger-runs/plugcharger_return_sweep_fast_slowed`), and run steps 3 and 4 of AGENTS.md with that config.
  Reusing the old work directory is refused ("was collected with different settings").
- **With a task board:** use a new board directory; the old one holds the earlier experiment.
- **A machine that has not started:** nothing special; follow AGENTS.md.
