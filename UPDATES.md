# Updates to a running experiment

For an agent on a machine that started the experiment before a `git pull`: read the entries below that are newer
than the commit this machine started from, and do what each says **for this machine**. The humans who own the
experiment approve every step written here, deletions included; do nothing beyond them, and stop and ask if
something doesn't match what an entry describes. Then start the queue again (AGENTS.md, step 4) and report what you
did. Entries are newest first.

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
