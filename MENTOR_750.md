# The 750-demo row of the threshold sweep: how to run it

This file is for whoever runs `configs/threshold_planner750.yaml` (branch `plugcharger-threshold-sweep`), a person or
an agent. For this run it replaces AGENTS.md, which describes another experiment. The README explains the pipeline
(sections "The job queue", "Part 2" to "Part 4", "The threshold-sweep report" and "Several machines").

**Only one file comes back:** `results/SWEEP_REPORT.md`, about 100 kB. It holds every number the threshold analysis
needs, and the figures can be redrawn from it. Nothing is uploaded to Hugging Face. Everything the run needs is
downloaded from public repos, so no login is needed.

## What the run does

The work comes in 19 tasks; `vig-dagger tasks -c configs/threshold_planner750.yaml` lists them.

1. **The MoE and its reference** (task 1):
   - Download the first 750 fast-motion planner demos (`chengenwu/fast-exp-plug-charger`, 15 files of 50, about
     10 GB).
   - Download the MoE trained on them (`chengenwu/moe-planner750-fast-plug-charger`, weights only, 190 MB). On the 200
     evaluation scenes this MoE alone solves 29.0 %.
   - Score the MoE's VIG on every 4th frame of all 750 demos (about 26,600 frames, 16 jobs). Its percentiles are the
     thresholds.
2. **Seven DAgger collections** (tasks 2–8), one fixed threshold each: p65, p70, p75, p80, p85, p90, p95.
   - Each collection gathers 800 successful episodes on the scenes from 1,000,000, with 4 simulator workers.
   - The expert takes over after 2 VIG scores above the threshold and hands back after 16 below it.
   - At every take-over the gripper makes a random 5 cm / 5° move.
   - Failed attempts are not saved.
3. **Dense policies** (tasks 10–16). Per collection, one full-resolution policy is trained on the 750 demos plus its
   800 episodes, with the random-move steps cut, then evaluated on the 200 scenes from 60000.
4. **Comparisons** (tasks 17–18). The MoE and the planner-only dense policy on 1550 demos (downloaded) are evaluated on
   the same scenes. Tasks 9 and 19 write summaries.

## Requirements

- **System:** Linux with NVIDIA GPUs. The simulator renders with Vulkan on the GPU. In a container this needs
  `NVIDIA_DRIVER_CAPABILITIES=all`.
- **Programs:** `git`, `uv`, `ffmpeg`, and `rsync` for several machines.
- **Disk per machine:**
  - 10 GB for the demos;
  - about 20 GB for each collection the machine runs, since a successful episode is about 23 MB;
  - a few GB for checkpoints.
- **Time:** about 1–3 days when the collections run side by side.
  - Collections at low thresholds finish first, because the expert drives more and most attempts succeed. p95 takes
    longest.
  - Then each dense training takes about 8 h on one GPU (86,000+ updates).

## Setup (on every machine)

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout plugcharger-threshold-sweep
uv sync
source .venv/bin/activate
```

Settings you may change in `configs/threshold_planner750.yaml`, only under `work_dir`, `resources` and `sharing`:

| setting | default | what it does |
|---|---|---|
| `work_dir` | `~/vig-dagger-runs/plugcharger_threshold_planner750` | where everything is written. Set it before the first start. It can differ between machines |
| `resources.gpus` | `auto` (every GPU) | e.g. `[0, 1, 2, 3]` |
| `resources.sim_slots` | `auto` (half the CPU cores) | how many simulator jobs run at once. A collection takes 4, and about 4 workers keep one GPU busy, so a good value is 4 × the machine's GPUs |
| `sharing.board` | `null` (one machine) | the task board shared by several machines, see below |

Changing `resources.*` while the run is going is fine: `vig-dagger stop -c … --queue-only`, edit, then start again.
**Never change anything else**, above all `demos`, `moe`, `thresholds`, `dagger`, `retrain` and `evaluation`.

## Several machines (recommended)

The machines divide the tasks among themselves through a **task board**: a directory on one machine that the others
reach with `ssh` and `rsync`.

1. **Pick a board** on one machine, e.g. `gpu1:/data/vig-board`. Any empty or new directory works. Every other
   machine needs passwordless `ssh gpu1` (an SSH key). A directory that every machine mounts also works.
2. **On every machine**, set `sharing.board: gpu1:/data/vig-board` in the config. Use the same value on every machine;
   the host itself may give the plain path `/data/vig-board`. Apart from `work_dir`, `resources` and `sharing`, keep
   the config identical: the board refuses a machine with other experiment settings.
3. **On every machine**, check and launch:

   ```bash
   vig-dagger check -c configs/threshold_planner750.yaml     # every line ok (WARN is fine), including the board
   setsid nohup vig-dagger run -c configs/threshold_planner750.yaml > run750.log 2>&1 < /dev/null &
   ```

   Machines can join later.

**How the work is divided:**
- **The MoE task.** The first machine with room claims task 1, the MoE and its reference, and publishes the MoE and
  the thresholds to the board.
- **Collections.** Each collection is then claimed by one machine, which collects it, then trains and evaluates its
  dense policy. Its episodes stay on that machine.
- **Spreading.** A machine claims one new task per minute while it has free simulator slots and at least 25 GB free.
  To spread the collections over several machines, set `resources.sim_slots` to about 4 × the machine's GPUs.
- **What moves through the board:** the MoE, the reference, and each collection's metrics, attempt records and
  episode lists (tens of MB each), plus the evaluation results.

**Watching and recovery:**
- `vig-dagger status` on any machine lists every task (done, running and where, failed, waiting) and every machine's
  last heartbeat.
- A machine that went down keeps its tasks. Start it again with the same work directory and it continues them. If it
  won't come back, `vig-dagger release <task>` on any machine frees each of its tasks for another machine; a released
  collection starts again from its first episode.

## One machine

Leave `sharing.board: null` and run the same two commands (`check`, then the `setsid nohup … run` line) on that
machine. It needs about 170 GB free.

## Checking on the run

```bash
vig-dagger status -c configs/threshold_planner750.yaml
tail -n 30 run750.log
df -h <work_dir>
```

If a machine's queue stopped (reboot, closed session) and tasks remain, run its `setsid nohup vig-dagger run …` line
again. It resumes: trainings from their last checkpoint, collections from their saved episodes.

## What to expect

The paths below are inside the work directory of the machine that ran the task.

| milestone | where | expected |
|---|---|---|
| demos | `data/raw/planner/part_000 … part_014` | 15 parts, 750 demos |
| MoE | `policies/moe/sd0/latest.pt` | 190 MB |
| reference | `policies/moe/sd0/thresholds.json` | `"demos": 750`, `"frames"` ≈ 26,579. VIG p65 ≈ 0.0202, p70 0.0227, p75 0.0260, p80 0.0303, p85 0.0363, p90 0.0452, p95 0.0604 (computed on the machine that trained the MoE; small differences are fine) |
| MoE alone | `eval/moe/sd0/level1/*.jsonl` | about 58 of 200 (29 %) |
| planner-only 1550 | `eval/extra/planner_1550/level1/*.jsonl` | about 101 of 200 (50.5 %) |
| collections | `vig-dagger status` | 800 successes each; success per attempt falls from p65 to p95 |
| dense policies | `policies/retrain/sd0_vig_p*_random_move/seed0/` | validation loss falling; 200 evaluation episodes each |

**Stop and tell us** (do not try to work around it):
- `vig-dagger check` fails on rendering or on the board;
- a task fails again after one restart;
- a machine's disk drops below 20 GB;
- a collection has fewer than 40 successes after 1,000 attempts.

**Never:**
- delete files in a work directory or on the board;
- run two queues on one work directory.

## When it finishes

`vig-dagger status` shows every task done. Then, on **any one machine**:

```bash
vig-dagger sweep-report -c configs/threshold_planner750.yaml
```

With a board, it first copies what it needs from the board: every collection's attempts and episode lists, the
reference scores and every evaluation. It then writes `<work_dir>/results/SWEEP_REPORT.md`. **Send us that file
only.**

A report sent before the end is welcome too, e.g. once the collections are done. `sweep-report` works at any point
and covers whatever has finished.

**Please keep the work directories and the board** until we confirm the analysis is done. If the report needs
another number, we push a change and you run `git pull` and the same `sweep-report` command again; nothing is
recomputed.
