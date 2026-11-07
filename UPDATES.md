# Updates to a running experiment

For an agent on a machine that started the experiment before a `git pull`: read the entries below that are newer
than the commit this machine started from, and do what each says **for this machine**. The humans who own the
experiment approve every step written here, deletions included; do nothing beyond them, and stop and ask if
something doesn't match what an entry describes. Then start the queue again (AGENTS.md, step 4) and report what you
did. Entries are newest first.

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
