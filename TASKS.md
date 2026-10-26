# OpenDrawer tasks

The experiment of `configs/opendrawer.yaml` as 63 numbered tasks, for running it on several machines with the tasks
assigned by hand. Tell a machine's agent "run task 45" (or "run tasks 3, 17, 32–40"); it runs
`vig-dagger run -c configs/run.yaml --tasks 45`. A task can start once the outputs of the tasks under "needs tasks"
are on that machine: on the machine that made them, `vig-dagger pack -c configs/run.yaml --tasks 3 --for 15`
writes one archive of what task 15 needs from task 3; on the other machine, `vig-dagger unpack -c configs/run.yaml
<file>` puts it in place. A task's inputs always come from lower numbers, and every machine downloads the demos
itself. Every machine must use the same config, so the numbers mean the same everywhere (`vig-dagger tasks` prints
this list).

What moves between machines: a checkpoint (weights only: ~130 MB, the MoE ~180 MB) to the tasks that use that policy; a DAgger training
set (a few hundred MB) to its three retrainings; thresholds and evaluation episodes (a few KB) to the next step and
the summary. Times are for one task on a free machine with an RTX PRO 6000 (trainings: 4.5–13 updates per second).

The longest chains: task 1 → 14 → 16/17 → 20–25 → 51–56, and tasks 2–8 → 15 → 18/19 → 26–31 → 57–62 (training,
thresholds, DAgger, retraining, evaluation). Everything else can run alongside.

### Stage 1: trainings that need only the demos

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 1 | the 7-level MoE (round `moe`'s driver) <br>`train/drivers/moe` | 1 GPU (training) | 3–10 h | – | 14, 16–17, 32–38 |
| 2 | dense policy on images degraded by 2 (round `per_resolution`) <br>`train/drivers/per_resolution/res2` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 39 |
| 3 | dense policy on images degraded by 4 (round `per_resolution`) <br>`train/drivers/per_resolution/res4` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 40 |
| 4 | dense policy on images degraded by 8 (round `per_resolution`) <br>`train/drivers/per_resolution/res8` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 41 |
| 5 | dense policy on images degraded by 16 (round `per_resolution`) <br>`train/drivers/per_resolution/res16` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 42 |
| 6 | dense policy on images degraded by 32 (round `per_resolution`) <br>`train/drivers/per_resolution/res32` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 43 |
| 7 | dense policy on images degraded by 64 (round `per_resolution`) <br>`train/drivers/per_resolution/res64` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 44 |
| 8 | retraining `base` seed0 (100 demos) <br>`train/retrain/base/seed0` | 1 GPU (training) | 3–10 h | – | 15, 18–19, 45 |
| 9 | retraining `base` seed1 (100 demos) <br>`train/retrain/base/seed1` | 1 GPU (training) | 3–10 h | – | 46 |
| 10 | retraining `base` seed2 (100 demos) <br>`train/retrain/base/seed2` | 1 GPU (training) | 3–10 h | – | 47 |
| 11 | retraining `base_plus_planner` seed0 (200 demos) <br>`train/retrain/base_plus_planner/seed0` | 1 GPU (training) | 7–20 h | – | 48 |
| 12 | retraining `base_plus_planner` seed1 (200 demos) <br>`train/retrain/base_plus_planner/seed1` | 1 GPU (training) | 7–20 h | – | 49 |
| 13 | retraining `base_plus_planner` seed2 (200 demos) <br>`train/retrain/base_plus_planner/seed2` | 1 GPU (training) | 7–20 h | – | 50 |

### Stage 2: thresholds

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 14 | round `moe`: score 50 demos, VIG and flow-loss percentiles <br>`thresholds/moe` | 1 GPU (inference) | 0.5–1 h | 1 | 16–17 |
| 15 | round `per_resolution`: score 50 demos, VIG and flow-loss percentiles <br>`thresholds/per_resolution` | 1 GPU (inference) | 0.5–1 h | 2–8 | 18–19 |

### Stage 3: DAgger collections

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 16 | round `moe`, VIG DAgger: 100 successful episodes, their training set and metrics <br>`dagger/moe/vig` | RTX GPU + many CPU cores (Isaac Sim) | 3–10 h | 1, 14 | 20–22, 63 |
| 17 | round `moe`, flow-loss DAgger: 100 successful episodes, their training set and metrics <br>`dagger/moe/flow` | RTX GPU + many CPU cores (Isaac Sim) | 3–10 h | 1, 14 | 23–25, 63 |
| 18 | round `per_resolution`, VIG DAgger: 100 successful episodes, their training set and metrics <br>`dagger/per_resolution/vig` | RTX GPU + many CPU cores (Isaac Sim) | 3–10 h | 2–8, 15 | 26–28, 63 |
| 19 | round `per_resolution`, flow-loss DAgger: 100 successful episodes, their training set and metrics <br>`dagger/per_resolution/flow` | RTX GPU + many CPU cores (Isaac Sim) | 3–10 h | 2–8, 15 | 29–31, 63 |

### Stage 4: retrainings on DAgger data

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 20 | retraining `base_plus_moe_vig` seed0 (200 demos) <br>`train/retrain/base_plus_moe_vig/seed0` | 1 GPU (training) | 7–20 h | 16 | 51 |
| 21 | retraining `base_plus_moe_vig` seed1 (200 demos) <br>`train/retrain/base_plus_moe_vig/seed1` | 1 GPU (training) | 7–20 h | 16 | 52 |
| 22 | retraining `base_plus_moe_vig` seed2 (200 demos) <br>`train/retrain/base_plus_moe_vig/seed2` | 1 GPU (training) | 7–20 h | 16 | 53 |
| 23 | retraining `base_plus_moe_flow` seed0 (200 demos) <br>`train/retrain/base_plus_moe_flow/seed0` | 1 GPU (training) | 7–20 h | 17 | 54 |
| 24 | retraining `base_plus_moe_flow` seed1 (200 demos) <br>`train/retrain/base_plus_moe_flow/seed1` | 1 GPU (training) | 7–20 h | 17 | 55 |
| 25 | retraining `base_plus_moe_flow` seed2 (200 demos) <br>`train/retrain/base_plus_moe_flow/seed2` | 1 GPU (training) | 7–20 h | 17 | 56 |
| 26 | retraining `base_plus_per_resolution_vig` seed0 (200 demos) <br>`train/retrain/base_plus_per_resolution_vig/seed0` | 1 GPU (training) | 7–20 h | 18 | 57 |
| 27 | retraining `base_plus_per_resolution_vig` seed1 (200 demos) <br>`train/retrain/base_plus_per_resolution_vig/seed1` | 1 GPU (training) | 7–20 h | 18 | 58 |
| 28 | retraining `base_plus_per_resolution_vig` seed2 (200 demos) <br>`train/retrain/base_plus_per_resolution_vig/seed2` | 1 GPU (training) | 7–20 h | 18 | 59 |
| 29 | retraining `base_plus_per_resolution_flow` seed0 (200 demos) <br>`train/retrain/base_plus_per_resolution_flow/seed0` | 1 GPU (training) | 7–20 h | 19 | 60 |
| 30 | retraining `base_plus_per_resolution_flow` seed1 (200 demos) <br>`train/retrain/base_plus_per_resolution_flow/seed1` | 1 GPU (training) | 7–20 h | 19 | 61 |
| 31 | retraining `base_plus_per_resolution_flow` seed2 (200 demos) <br>`train/retrain/base_plus_per_resolution_flow/seed2` | 1 GPU (training) | 7–20 h | 19 | 62 |

### Stage 5: evaluations

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 32 | evaluate `driver/moe` at full resolution, 200 episodes <br>`eval/driver/moe/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 33 | evaluate `driver/moe` at 1/2, 200 episodes <br>`eval/driver/moe/level2` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 34 | evaluate `driver/moe` at 1/4, 200 episodes <br>`eval/driver/moe/level4` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 35 | evaluate `driver/moe` at 1/8, 200 episodes <br>`eval/driver/moe/level8` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 36 | evaluate `driver/moe` at 1/16, 200 episodes <br>`eval/driver/moe/level16` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 37 | evaluate `driver/moe` at 1/32, 200 episodes <br>`eval/driver/moe/level32` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 38 | evaluate `driver/moe` at 1/64, 200 episodes <br>`eval/driver/moe/level64` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 1 | 63 |
| 39 | evaluate `driver/per_resolution/res2` at 1/2, 200 episodes <br>`eval/driver/per_resolution/res2/level2` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 2 | 63 |
| 40 | evaluate `driver/per_resolution/res4` at 1/4, 200 episodes <br>`eval/driver/per_resolution/res4/level4` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 3 | 63 |
| 41 | evaluate `driver/per_resolution/res8` at 1/8, 200 episodes <br>`eval/driver/per_resolution/res8/level8` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 4 | 63 |
| 42 | evaluate `driver/per_resolution/res16` at 1/16, 200 episodes <br>`eval/driver/per_resolution/res16/level16` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 5 | 63 |
| 43 | evaluate `driver/per_resolution/res32` at 1/32, 200 episodes <br>`eval/driver/per_resolution/res32/level32` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 6 | 63 |
| 44 | evaluate `driver/per_resolution/res64` at 1/64, 200 episodes <br>`eval/driver/per_resolution/res64/level64` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 7 | 63 |
| 45 | evaluate `retrain/base/seed0` at full resolution, 200 episodes <br>`eval/retrain/base/seed0/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 8 | 63 |
| 46 | evaluate `retrain/base/seed1` at full resolution, 200 episodes <br>`eval/retrain/base/seed1/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 9 | 63 |
| 47 | evaluate `retrain/base/seed2` at full resolution, 200 episodes <br>`eval/retrain/base/seed2/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 10 | 63 |
| 48 | evaluate `retrain/base_plus_planner/seed0` at full resolution, 200 episodes <br>`eval/retrain/base_plus_planner/seed0/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 11 | 63 |
| 49 | evaluate `retrain/base_plus_planner/seed1` at full resolution, 200 episodes <br>`eval/retrain/base_plus_planner/seed1/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 12 | 63 |
| 50 | evaluate `retrain/base_plus_planner/seed2` at full resolution, 200 episodes <br>`eval/retrain/base_plus_planner/seed2/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 13 | 63 |
| 51 | evaluate `retrain/base_plus_moe_vig/seed0` at full resolution, 200 episodes <br>`eval/retrain/base_plus_moe_vig/seed0/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 20 | 63 |
| 52 | evaluate `retrain/base_plus_moe_vig/seed1` at full resolution, 200 episodes <br>`eval/retrain/base_plus_moe_vig/seed1/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 21 | 63 |
| 53 | evaluate `retrain/base_plus_moe_vig/seed2` at full resolution, 200 episodes <br>`eval/retrain/base_plus_moe_vig/seed2/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 22 | 63 |
| 54 | evaluate `retrain/base_plus_moe_flow/seed0` at full resolution, 200 episodes <br>`eval/retrain/base_plus_moe_flow/seed0/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 23 | 63 |
| 55 | evaluate `retrain/base_plus_moe_flow/seed1` at full resolution, 200 episodes <br>`eval/retrain/base_plus_moe_flow/seed1/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 24 | 63 |
| 56 | evaluate `retrain/base_plus_moe_flow/seed2` at full resolution, 200 episodes <br>`eval/retrain/base_plus_moe_flow/seed2/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 25 | 63 |
| 57 | evaluate `retrain/base_plus_per_resolution_vig/seed0` at full resolution, 200 episodes <br>`eval/retrain/base_plus_per_resolution_vig/seed0/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 26 | 63 |
| 58 | evaluate `retrain/base_plus_per_resolution_vig/seed1` at full resolution, 200 episodes <br>`eval/retrain/base_plus_per_resolution_vig/seed1/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 27 | 63 |
| 59 | evaluate `retrain/base_plus_per_resolution_vig/seed2` at full resolution, 200 episodes <br>`eval/retrain/base_plus_per_resolution_vig/seed2/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 28 | 63 |
| 60 | evaluate `retrain/base_plus_per_resolution_flow/seed0` at full resolution, 200 episodes <br>`eval/retrain/base_plus_per_resolution_flow/seed0/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 29 | 63 |
| 61 | evaluate `retrain/base_plus_per_resolution_flow/seed1` at full resolution, 200 episodes <br>`eval/retrain/base_plus_per_resolution_flow/seed1/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 30 | 63 |
| 62 | evaluate `retrain/base_plus_per_resolution_flow/seed2` at full resolution, 200 episodes <br>`eval/retrain/base_plus_per_resolution_flow/seed2/level1` | RTX GPU + ~8 CPU cores (Isaac Sim) | 1–2 h | 31 | 63 |

### Stage 6: summary

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 63 | evaluation summary and REPORT.md <br>`summary` | CPU | minutes | 16–19, 32–62 | – |

