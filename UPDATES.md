# Updates to a running experiment

For an agent on a machine that started the experiment before a `git pull`: read the entries below that are newer
than the commit this machine started from, and do what each says **for this machine**. The humans who own the
experiment approve every step written here, deletions included; do nothing beyond them, and stop and ask if
something doesn't match what an entry describes. Then start the queue again (AGENTS.md, step 4) and report what you
did. Entries are newest first.

## 2026-11-26 (3): the 150-demo MoE's thresholds come from all its 150 training demos

The reference frames, of which the thresholds p65 ... p95 are percentiles (and the report's reference), are now every
4th frame of ALL the MoE's 150 training demos (`thresholds.num_demos: 150`, scored by 8 jobs), not only of the first
50: the same rule for every row of the sweep and as in iteration mode. The dry run of entry (2) now lists 145 jobs
(8 scoring jobs instead of 4). Nothing else changes.

**The vast.ai machine (one RTX 4090) only:**
1. `git pull`. If this machine is still on branch `plugcharger-iterate-adaptive`, do entry 2026-11-26 (2) first (it
   switches to `plugcharger-threshold-sweep`), with 145 jobs in its dry run; its queue then starts with this config,
   and you are done here.
2. Otherwise look at `NEW/policies/moe/sd0/` (`NEW` = `~/vig-dagger-runs/plugcharger_threshold_planner150`):
   - no `thresholds.json` and no `demo_scores_parts/` (the MoE still trains): stop only the queue,
     `vig-dagger stop -c configs/threshold_planner150.yaml --queue-only`; the training keeps running.
   - `demo_scores_parts/` but no `thresholds.json` (the scoring runs, with chunks cut for 50 demos):
     `vig-dagger stop -c configs/threshold_planner150.yaml` (the training is finished by then), then rename the
     folder: `mv NEW/policies/moe/sd0/demo_scores_parts NEW/policies/moe/sd0/demo_scores_parts.first50`.
   - `thresholds.json` exists: if `NEW/dagger/` exists (a collection started with thresholds from the 50 base demos),
     `vig-dagger stop -c configs/threshold_planner150.yaml` and ask. Otherwise stop the queue the same way, then rename
     `thresholds.json` to `thresholds.first50.json` and `demo_scores.npz` to `demo_scores.first50.npz` (and
     `demo_scores_parts` to `demo_scores_parts.first50` if it is there). Never delete them.
3. Start the queue as in entry (2) step 5. When the scoring is done, `NEW/policies/moe/sd0/thresholds.json` must say
   `"demos": 150` and about 5,000 frames; report its p65 ... p95.

## 2026-11-26 (2): run the whole 150-demo threshold sweep, starting as soon as its MoE is trained (branch plugcharger-threshold-sweep)

The 150-demo row of the threshold sweep no longer stops after the MoE: one queue runs everything in
`configs/threshold_planner150.yaml`, now on the branch **`plugcharger-threshold-sweep`** (this branch + a report made
for the sweep, `vig-dagger sweep-report`; nothing the MoE training uses changed). After the MoE and its thresholds:
seven DAgger collections at p65, p70, ..., p95 of that MoE's VIG (200 successful episodes each, scenes from 1,000,000,
failed attempts keep only their per-step record), a dense policy per collection on the 150 demos + its 200 episodes,
each evaluated on the 200 scenes from 60000 next to the MoE and next to the planner-only dense policy on 350 demos
(downloaded from the public repo `chengenwu/planner-only-dense-fast-plug-charger`, evaluated here).

**The vast.ai machine (one RTX 4090) only:**
1. Where this machine is with entry 2026-11-26:
   - not started yet: do its steps 1-4 (work directory, the three planner parts, their check), then continue here at
     step 2 (do not start its `--only` queue);
   - its `--only` queue is running (the MoE training or its scoring): stop only the queue, the jobs keep running:
     `vig-dagger stop -c configs/threshold_planner150.yaml --queue-only`;
   - its `--only` queue has finished: nothing to stop.
2. Switch the code: `git fetch && git switch plugcharger-threshold-sweep` (later "update"s pull this branch). If the
   iteration experiment's queue (`configs/run.yaml`) runs on this machine, stop here and ask.
3. Disk: the work directory `NEW` = `~/vig-dagger-runs/plugcharger_threshold_planner150` needs about 40 GB free on its
   disk (1,400 successful episodes of ~20 MB, the datasets and 7 dense policies). Less: stop and ask.
4. Dry run: `vig-dagger run -c configs/threshold_planner150.yaml --dry-run`. Expected: the MoE jobs done or running,
   then `download_extra_planner_350` and eight `eval/extra/planner_350/<scene>`, for each of the seven collections
   `sd0_vig_p65_random_move` ... `sd0_vig_p95_random_move` its `dagger_prepare_`, four `dagger_..._<worker>` and
   `analyze_`, then its `dagger_dataset_`, `train/retrain/<c>/seed0` and eight `eval/retrain/<c>/seed0/<scene>`, the
   MoE's eight evaluations, `summarize` and `final_report` (145 jobs since entry (3): 8 scoring jobs). Anything
   else: stop and ask.
5. Start the whole queue (it takes over a MoE training that is still running; the collections start as soon as the
   MoE's thresholds exist):
   ```bash
   setsid nohup vig-dagger run -c configs/threshold_planner150.yaml >> $NEW/run.log 2>&1 < /dev/null &
   ```
6. On "status": `vig-dagger status -c configs/threshold_planner150.yaml`, `tail $NEW/run.log`, `df -h $NEW`. Each
   collection's success rate per attempt falls with the threshold (on another MoE: about 80 % at p75, 30 % at p90);
   stop and ask if one has fewer than 15 successes after 300 attempts. On "report", and at the end:
   `vig-dagger sweep-report -c configs/threshold_planner150.yaml`, then send `$NEW/results/SWEEP_REPORT.md` and the
   figures in `$NEW/results/figures/from_report/`. Keep `NEW` as it is.

## 2026-11-26: train a MoE on the first 150 planner demos (a new experiment, in its own work directory)

A new experiment: fixed-threshold sweeps (p65 ... p95) with MoEs trained on planner demos only, 50 / 150 / 350 / 750 of
them, to find how the threshold should follow the MoE. The 350- and 750-demo MoEs are trained on another machine;
**this machine trains the 150-demo MoE** with `configs/threshold_planner150.yaml` (sweep mode). For now only the
MoE jobs run; the DAgger collections follow in a later entry. Nothing changes in the code, and nothing in any other
work directory is touched.

**The vast.ai machine (one RTX 4090) only:**
1. If a vig-dagger queue is still running on this machine (`vig-dagger status -c configs/run.yaml`), stop here and
   ask the humans: this needs the GPU.
2. Make the work directory `NEW` = `~/vig-dagger-runs/plugcharger_threshold_planner150` (needs about 3 GB free).
3. Download the first 150 planner demos (parts 0-2) from the dataset repo and put the three parts side by side in
   `NEW/data/raw/planner/`:
   ```bash
   NEW=~/vig-dagger-runs/plugcharger_threshold_planner150
   python -c "from huggingface_hub import snapshot_download as s; s('chengenwu/fast-exp-plug-charger', repo_type='dataset', local_dir='$NEW/hub/dataset', allow_patterns=['raw/planner/part_000.*', 'raw/planner_baselines/part_001.*', 'raw/planner_baselines/part_002.*'])"
   mkdir -p $NEW/data/raw/planner
   cp -al $NEW/hub/dataset/raw/planner/part_000.* $NEW/hub/dataset/raw/planner_baselines/part_001.* $NEW/hub/dataset/raw/planner_baselines/part_002.* $NEW/data/raw/planner/
   ```
   If some of these parts are already on this machine (the base demos `part_000.*` are in the iteration run's
   `data/raw/planner/`; `part_001` and `part_002` were collected on another machine, so probably only on the hub),
   hard-link those with `cp -al` instead and download only the rest.
   Check: `NEW/data/raw/planner/` has `part_000`, `part_001`, `part_002`, each `.h5`, `.json` and `.done.json`; the
   `.done.json` files say `first_seed` 2000000 / 2001000 / 2002000, `strategy_seed` 1 / 2 / 3, `motion` fast_maniskill,
   50 demos each. Anything else: stop and ask.
4. Dry run: `vig-dagger run -c configs/threshold_planner150.yaml --only 'moe|thresholds|score' --dry-run`. Expected:
   `collect_planner_000` ... `_002` done; to run: `convert_planner`, `train/moe/sd0`, `score_sd0_0` ... `_3`,
   `thresholds_sd0` and eight `eval/moe/sd0/<scene>` jobs. Anything else listed: stop and ask.
5. Start it like step 4 of AGENTS.md, with the same `--only` and its own log:
   ```bash
   setsid nohup vig-dagger run -c configs/threshold_planner150.yaml --only 'moe|thresholds|score' > $NEW/run.log 2>&1 < /dev/null &
   ```
   The MoE trains on 146 demos (3 of the 150 go to validation), 18,177 samples, 17,100 updates.
6. When `vig-dagger status -c configs/threshold_planner150.yaml` shows every job done, report: how long the training
   took, the training numbers (from `NEW/policies/moe/sd0/`), all of `NEW/policies/moe/sd0/thresholds.json`, and the
   MoE's success on the 200 scenes (`vig-dagger report -c configs/threshold_planner150.yaml`, then
   `NEW/results/REPORT.md`). Keep `NEW` as it is: this row's DAgger collections will run in it.

## 2026-11-25 (3): a figure of success against the number of training demos

`vig-dagger report` now also writes `results/figures/success_by_demos.png`: success on the evaluation scenes against
the number of training demos (log scale, Wilson 95 % bands), the dense policy retrained after each round next to the
dense policies on as many planner demos only, both from round 0 (the base demos alone). The planner-only line appears
once their evaluations are in `<work_dir>/eval/planner/` (the planner-only machine has finished: the humans send that
folder). On the planner-only machine the same figure shows that line alone. Nothing else changes: no experiment
parameter, no job, no output.

**A machine running this experiment:** after `git pull`, nothing to stop or restart. When you next report, run
`vig-dagger report -c configs/run.yaml` and send `success_by_demos.png` with the usual figures.

## 2026-11-25 (2): every attempt counts the same in the threshold update; run the rounds again in a new work directory (a test)

With the update by policy steps (n / N), round 1 here jumped from p80 to p100 on its first attempt, a few 400-step
failures at p100 (where the expert cannot take over) pulled it to p35, and there, with only ~20 policy steps per
attempt against ~2,300 so far, it crept to p41 over 100 attempts: 88 % of round 1's data was the expert's. The new
option `iterate.update_weight: attempts` (README, Iteration mode, "Every attempt counts the same"; on in
`configs/plugcharger.yaml`) makes every attempt move the threshold the same and counts the start as one attempt:
`p <- p + (q(M) - q(m)) / (u + 2)`, u = the round's earlier updates. Everything else stays as it is, failed attempts
counted as in entry (4).

**The machine running the DAgger rounds** (round 2 retraining when this was written). The test starts from the same
base demos and the same round-1 MoE, in a new work directory, so only the update rule differs:
1. Stop the queue and its jobs: `vig-dagger stop -c configs/run.yaml`.
2. Leave the current work directory (`OLD` below) exactly as it is: never delete or move anything in it. Its results
   stay for the comparison.
3. Make the new work directory `NEW` = `~/vig-dagger-runs/plugcharger_iterate_adaptive_equal` (needs about 100 GB free
   on its disk; if there is not, ask) and copy into it with `cp -a`, keeping the same relative paths:
   `OLD/data/raw/planner/`, `OLD/data/datasets/base/`, `OLD/data/datasets/planner/`, `OLD/policies/moe/round1/` (the
   round-1 MoE with its `thresholds.json` and `demo_scores.npz`) and, if it has 200 episodes, `OLD/eval/moe/round1/`.
   Copy nothing else.
4. In `configs/run.yaml`: set `work_dir` to `NEW`, and under `iterate:` add `update_weight: attempts` (as in
   `configs/plugcharger.yaml`). Change nothing else.
5. `vig-dagger run -c configs/run.yaml --dry-run`: the base demos, `convert_planner`, `train/moe/round1`, its
   `score_round1_*` and `thresholds_round1` (and its evaluation, if copied) must show as done. If not, stop and ask.
6. Start the queue (AGENTS.md step 4). Round 1's collection starts again from p80 on the same scenes;
   `logs/dagger_prepare_round1_sd0_vig_adaptive_random_move.log` says "every attempt weighted the same (start counts as
   one)". Do not stop after round 1: the queue continues through all rounds.
7. Report after about 50 and 100 attempts of round 1: `vig-dagger report -c configs/run.yaml`, then send
   `results/figures/thresholds.png`, `distributions.png` and `collections.png`, successes / attempts so far and the
   threshold now; then report as usual after each round.

## 2026-11-25: more figures (collections.png, retraining.png, the random moves per round)

`vig-dagger report` now also writes `results/figures/collections.png` (per round's collection, as bars: success rate
per attempt; expert and random-move share of the DAgger data, of all attempts and of the failed attempts; takeovers
per kept episode) and `results/figures/retraining.png` (per round: the dense policy retrained after the round, the
dense policy on as many planner demos when its evaluation is in `eval/planner/`, and the round's MoE, with Wilson
intervals and paired tests) and `results/figures/moves/spheres_{held,progress}_rounds.png` (every random move of
each round's DAgger data on the third-person camera, one panel per round). Nothing else changes: no experiment parameter, no job, no output.

Round 0 (the dense policy on the 50 base demos alone, `planner/round0`) is trained on the planner-only machine; when
its evaluation is copied into `eval/planner/round0/`, the figures and tables start from it.

**A machine running this experiment:** after `git pull`, nothing to stop or restart (the next report job uses the new
code). Run `vig-dagger report -c configs/run.yaml` once and send the new figures with the usual ones.

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
