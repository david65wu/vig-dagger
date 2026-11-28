# The 750-demo row of the threshold sweep: how to run it

This file is for whoever runs `configs/threshold_planner750.yaml` (branch `plugcharger-threshold-sweep`), a person or
an agent. For this run it replaces AGENTS.md, which describes another experiment. The README explains the pipeline
(sections "The job queue", "Part 2" to "Part 4", "The threshold-sweep report" and "Several machines").

**Only one file comes back:** `results/SWEEP_REPORT.md`, about 100 kB (one per machine when the tasks are assigned
by hand). It holds every number the threshold analysis needs, and the figures can be redrawn from it. Nothing is
uploaded to Hugging Face. Everything the run needs is downloaded from public repos, so no login is needed.

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
| `sharing.board` | `null` | the task board shared by several machines (only for "Several machines (with a board)") |

Changing `resources.*` while the run is going is fine: `vig-dagger stop -c … --queue-only`, edit, then start again.
**Never change anything else**, above all `demos`, `moe`, `thresholds`, `dagger`, `retrain` and `evaluation`.

## First: how to spread the run

Three ways; pick one before you start:

1. **One machine** (the simplest): one queue runs everything. Best on a machine with 7 or more GPUs; with fewer it
   takes longer. See "One machine".
2. **Several machines, tasks assigned by hand** (no shared directory needed): you decide which machine runs which
   thresholds, each machine runs its own, and each sends back its own report. See "Several machines, tasks assigned by
   hand".
3. **Several machines with a task board**: only if the machines share a directory or reach each other with
   passwordless ssh; the machines then take the tasks themselves. See "Several machines (with a board)".

**Machines under a job scheduler** (e.g. SLURM, where you submit jobs instead of logging in to a machine): the queue
must run for days. Prefer long jobs, one per machine, each with its own work directory. When a job ends at its time
limit, submit the same command again: it resumes where it stopped.

## Several machines, tasks assigned by hand

Nothing moves between the machines: every machine downloads what it needs from Hugging Face, runs the tasks you give
it, and writes its own report.

**Assign by threshold.** A collection and its dense policy run on the same machine (the dense policy trains on the
collection's episodes, about 18 GB). Task 1 (download the MoE and score the demos, under an hour) runs on every
machine. The two comparison evaluations (17 and 18, under an hour each) run on any one machine.

| threshold | tasks to run |
|---|---|
| p65 | `1 2 10` |
| p70 | `1 3 11` |
| p75 | `1 4 12` |
| p80 | `1 5 13` |
| p85 | `1 6 14` |
| p90 | `1 7 15` |
| p95 | `1 8 16` |
| the comparisons (the MoE and the planner-only policy) | `1 17 18` |

**How many per machine.** One collection keeps about one GPU busy (4 simulator workers), so a machine with G GPUs runs
G thresholds at once; set `resources.sim_slots` to 4 × G (it needs about as many CPU cores). Start the slow thresholds
first: p95 takes longest, then p90; p65 is the quickest.

**Launch** on each machine, its tasks in one command. Example: a machine with 2 GPUs that runs p95, p65 and the
comparisons (with your local disk in place of `/local/disk`):

```bash
W=/local/disk/vig750
vig-dagger check -c configs/threshold_planner750.yaml --set work_dir=$W --set resources.sim_slots=8
setsid nohup vig-dagger run -c configs/threshold_planner750.yaml --set work_dir=$W --set resources.sim_slots=8 \
    --tasks 1 8 16 2 10 17 18 > run750.log 2>&1 < /dev/null &
```

Every later command on that machine (`status`, `stop`, `sweep-report`) takes the same `--set` options. The disk line
of `check` estimates the whole experiment; a machine needs only about 20 GB per threshold it runs plus 10 GB of demos.

**Giving a machine more thresholds later** (e.g. when one finished): stop its queue with `vig-dagger stop -c … --set
work_dir=$W --queue-only` (its jobs keep running), then start the same `run` line with the longer list, e.g.
`--tasks 1 8 16 2 10 17 18 4 12`. Finished tasks are skipped and running jobs are taken over. Never start a second
queue on the same work directory, and never give the same threshold to two machines.

**Reports.** On each machine, `vig-dagger sweep-report -c configs/threshold_planner750.yaml --set work_dir=$W` writes
that machine's `<work_dir>/results/SWEEP_REPORT.md`. Send us every machine's file, renamed after the machine (e.g.
`SWEEP_REPORT_<hostname>.md`); we merge them (`python -m vig_dagger.sweep_report merge`).

In this mode, the queue's last line suggests `vig-dagger pack` for the tasks that use this machine's outputs: ignore it, nothing is copied between the machines.

## Several machines (with a board)

First check that the machines can share a board directory, with one of these two checks (a minute each; `A` and `B`
stand for two of your machines):

- **A shared filesystem.** Many clusters mount the same disk on every machine, often the home directory or something
  like `/data`, `/shared` or `/scratch`. Run `echo hello > <dir>/vig-board-test` on A and `cat <dir>/vig-board-test` on
  B: if B prints "hello", `<dir>` is shared and the board is a plain path there, e.g. `sharing.board: <dir>/vig-board`.
  (`df -hT <dir>` shows the filesystem type: `nfs`, `lustre`, `gpfs`, `cephfs`, `beegfs` and the like are shared.)
  Delete `vig-board-test` afterwards.
- **Passwordless ssh.** On B, `ssh A hostname` prints A's name without asking for a password: the board can be a
  directory on A, written `A:/path/on/A/vig-board`. Every other machine must reach A this way, and every machine needs
  `rsync`.

If neither works, assign the tasks by hand (above).

1. **On every machine**, set `sharing.board` in the config to the value from the check above, the same on every
   machine (on A itself, `A:/path` may also be written as the plain path `/path`). Apart from `work_dir`,
   `resources` and `sharing`, keep the config identical: the board refuses a machine with other experiment settings.
2. **Every machine needs its own work directory**, best on a disk of its own (not the shared one). If the home
   directory is shared, the default `work_dir` would be the same directory on every machine, which breaks the run.
   Give each machine its own, e.g. with `--set work_dir=…` and the host name, as below.
3. **On every machine**, check and launch (with your local disk in place of `/local/disk`):

   ```bash
   W=/local/disk/vig750_$(hostname)
   vig-dagger check -c configs/threshold_planner750.yaml --set work_dir=$W     # every line ok (WARN is fine), including the two board lines
   setsid nohup vig-dagger run -c configs/threshold_planner750.yaml --set work_dir=$W > run750_$(hostname).log 2>&1 < /dev/null &
   ```

   Every later command on that machine (`status`, `stop`, `sweep-report`) takes the same `--set work_dir=$W`.

   The disk line of `check` estimates the whole run on every machine; on a machine with less than that it shows
   FAIL or WARN, which is fine with a board as long as the machine has room for the collections it will run (about
   20 GB each, plus 10 GB of demos). Machines can join later.

**How the work is divided:**
- **The MoE task.** The first machine with room claims task 1, the MoE and its reference, and publishes the MoE and
  the thresholds to the board.
- **Collections.** Each collection is then claimed by one machine, which collects it, then trains and evaluates its
  dense policy. Its episodes stay on that machine. Before its first collection, a machine downloads the 750 demos
  (10 GB, a few minutes): the collection checks the expert against them and the dense policy trains on them.
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

Leave `sharing.board: null` and run, on that machine:

```bash
vig-dagger check -c configs/threshold_planner750.yaml
setsid nohup vig-dagger run -c configs/threshold_planner750.yaml > run750.log 2>&1 < /dev/null &
```

It needs about 170 GB free. Set `resources.sim_slots` to about 4 × its GPUs (at least 28 for all seven
collections at once, if the machine has the CPU cores: about one core per slot).

## Checking on the run

```bash
vig-dagger status -c configs/threshold_planner750.yaml      # plus the same --set options as the run line
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
- `vig-dagger check` fails on rendering (or on the board, with a board);
- a task fails again after one restart;
- a machine's disk drops below 20 GB;
- a collection has fewer than 40 successes after 1,000 attempts.

**Never:**
- delete files in a work directory or on the board;
- run two queues on one work directory.

## When it finishes

`vig-dagger status` shows every task done. Then:

```bash
vig-dagger sweep-report -c configs/threshold_planner750.yaml            # plus the same --set options as the run line
```

- **One machine, or a board:** run it on any one machine. With a board, it first copies what it needs from the board
  (every collection's attempts and episode lists, the reference scores and every evaluation). **Send us
  `<work_dir>/results/SWEEP_REPORT.md` only.**
- **Tasks assigned by hand:** run it on every machine, and send us every machine's `SWEEP_REPORT.md`, renamed after the
  machine.

A report sent before the end is welcome too, e.g. once the collections are done. `sweep-report` works at any point
and covers whatever has finished.

**Please keep the work directories and the board** until we confirm the analysis is done. If the report needs
another number, we push a change and you run `git pull` and the same `sweep-report` command again; nothing is
recomputed.
