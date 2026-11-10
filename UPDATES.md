# Updates to a running experiment

For an agent on a machine that started the experiment before a `git pull`: read the entries below that are newer
than the commit this machine started from, and do what each says **for this machine**. The humans who own the
experiment approve every step written here, deletions included; do nothing beyond them, and stop and ask if
something doesn't match what an entry describes. Then start the queue again (AGENTS.md, step 4) and report what you
did. Entries are newest first.

## 2026-11-10: the retrainings' expected number of training demos was too narrow

**What changed.** Only the expected range in the entry below and in AGENTS.md (milestones); no code. A retraining
holds out ~2 % of the 500 DAgger episodes for validation, each one picked independently by a hash of its name: 10
on average, but anywhere from about 0 to 20 by chance, different for every collection. So `_info.training_demos` is
975 + 480–500 = **1,455–1,475**, not "about 1,465–1,475" (which allowed at most 10 held out and would have stopped
about 4 retrainings in 10). The 975 base demos stay exactly 975.

**Does it apply to this machine?** Only if its queue was stopped because a retraining's `_info.training_demos` was
below 1,465.

**Steps:** if that value is 1,455–1,475 and `training.max_updates` is about 280,000–330,000, the training is right:
start the queue again with the same command (AGENTS.md, step 4). Keep the retraining's folder; it resumes, or
starts from zero if it saved nothing yet. Outside 1,455–1,475, stop and ask.

## 2026-11-09 (the commit after 6df7dc7): retraining and evaluation, new tasks 41–80

**What changed.**

1. New tasks after 40; tasks 1–40 keep their numbers (TASKS.md lists all 80):
   - 41–76: task N + 37 retrains collection task N: a dense full-resolution policy on the 1000 base demos + that
     collection's 500 DAgger episodes (60 epochs, training seed 0), then evaluates it on 200 scenes. It needs the
     collection's episodes, so it runs on the machine that ran task N;
   - 77–79: MoE seed 0, 1, 2 evaluated at full resolution on the same scenes, where task 1, 2, 3 ran;
   - 80: the final report, on one machine at the end.
2. The demo names of the entry below (2026-11-07) are now the default in the code, so a `configs/run.yaml` copied
   before that entry gets them too.

**Does it apply to this machine?** Yes, every machine: it adds the retrainings of the collections it ran.

**Steps:**

1. Check this machine's dataset as in the entry below. If `data/datasets/planner` exists and its first demo name is
   `part_000__traj_0` (also if it was rebuilt after that entry, from a `configs/run.yaml` without `demo_names`), or if
   a `policies/moe/sd1` or `sd2` here has `"max_updates": 189180` in its `run_config.json`, do that entry's steps 1–3
   first (stop and ask instead where it says so). The deletions there are approved here too.
2. Find the tasks this machine was given: the `--tasks` of its queue (`ps -o args= -p $(cat <work_dir>/logs/queue.pid)`
   while it runs; otherwise the command it was launched with).
3. The new list: those tasks, plus N + 37 for every collection task N (4–39) among them, plus 77, 78 or 79 if task 1,
   2 or 3 is among them. For example, tasks 1, 4–15 become 1, 4–15, 41–52, 77; tasks 2, 16–27 become 2, 16–27, 53–64,
   78; tasks 3, 28–39 become 3, 28–39, 65–76, 79.
4. If its queue is running: `vig-dagger stop -c configs/run.yaml --queue-only` (the DAgger workers keep running).
   Then launch it with the new list (AGENTS.md, step 4), e.g.
   `setsid nohup vig-dagger run -c configs/run.yaml --tasks 1 4-15 41-52 77 > run_tasks_1_4-15_41-52_77.log 2>&1 < /dev/null &`.
   It takes over the running workers. Each retraining starts when its collection is finished (after downloading and
   converting the 1000 base demos, if this machine hasn't); one training per GPU at a time.
5. Once a retraining has started, check `<work_dir>/policies/retrain/<collection>/seed0/run_config.json`:
   `training.max_updates` about 280,000–330,000 and `_info.training_demos` 1,455–1,475 (corrected by the entry
   above). If not, stop the queue and ask.
6. Report the new task list and what is running.

**The final report (task 80)** comes last: when every machine is done, the humans ask each machine but one for
"pack tasks *all its tasks* for task 80" (under a MB), bring the archives to the last machine, and ask there for
"unpack *file*" for each, then "run task 80", then "report".

## 2026-11-07, commit 11b6b67: seeds 1 and 2 hold out the same validation demos as seed 0

**What changed.** The demos converted for the MoE trainings are now named as in the dataset the seed-0 MoE was
trained on. The names decide which demos are held out for validation, so with the old names (`part_000__traj_0`) a
training used 980 training demos and 189,180 updates; with the new ones it uses seed 0's 975 demos and exactly 188,580.

**Does it apply to this machine?** Only if this machine converted the demos before this commit, that is, it trains
(or trained) MoE seed 1 or 2. Check:

```bash
grep -o '"demo_key": "[^"]*' <work_dir>/data/datasets/planner/plug_charger/demo_manifest.json | head -1
```

- No such file, or the name starts with `sd0_plug_charger_hole4mm_sidecam_randsocket_vargrasp_part0__traj_`:
  nothing to do here. Restart the queue if it was stopped.
- The name is `part_000__traj_0`: this machine has the old dataset. Do the steps below.

**Steps (only with the old dataset):**

1. Stop the queue: `vig-dagger stop -c configs/run.yaml`.
2. If a collection of seed 1 or 2 has started on this machine (a folder `<work_dir>/dagger/sd1_*` or `dagger/sd2_*`
   exists), stop here and ask the humans: that collection was driven by a MoE trained on the old dataset.
3. Delete these, and nothing else:
   - `<work_dir>/data/datasets/planner` and `<work_dir>/data/datasets/base` (rebuilt from the raw demos in minutes);
   - `<work_dir>/policies/moe/sd1` and `<work_dir>/policies/moe/sd2`, whichever exists (a training on the old
     dataset; it starts again from zero).

   Keep `data/raw/`, `hub/` and everything else.
4. Start the queue again (AGENTS.md, step 4).
5. Once the training has started, check `<work_dir>/policies/moe/sd<s>/run_config.json`: `training.max_updates` must
   be 188580 and `_info.training_demos` 975. If not, stop the queue and ask.

Without these steps the pipeline refuses to go on: the conversion says the dataset "names its demos like
'part_000__traj_0'", and a training says its `run_config.json` "was written for another dataset". Both name the same
folders as above.
