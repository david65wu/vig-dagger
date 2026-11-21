# Tasks of configs/plugcharger.yaml

The experiment's work as 12 numbered tasks. A task's inputs always come from lower numbers. With a task board
(`sharing.board`, see [AGENTS.md](AGENTS.md#several-machines)) the machines claim these tasks themselves, most urgent
first; without one, the humans assign them by number: `vig-dagger run -c configs/run.yaml --tasks N ...` runs tasks,
`vig-dagger pack --tasks N --for M` / `vig-dagger unpack FILE` move outputs between machines. `vig-dagger tasks
-c configs/run.yaml` prints this list. Times are rough, for a shared 64-core machine with 2 GPUs. On one machine,
`vig-dagger run -c configs/run.yaml` runs all of them.

### Stage 1: the MoEs and their thresholds

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 1 | MoE seed 0: download it (Hugging Face), score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd0` | 1 GPU (inference) | 0.5–1 h | – | 2–5, 11 |

### Stage 2: DAgger collections

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 2 | MoE seed 0, VIG: take over above p90, hand back below p70 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_r70_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 6–7, 12 |
| 3 | MoE seed 0, VIG: take over above p90, hand back below p75 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_r75_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 6, 8, 12 |
| 4 | MoE seed 0, VIG: take over above p90, hand back below p80 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_r80_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 6, 9, 12 |
| 5 | MoE seed 0, VIG: take over above p90, hand back below p85 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_r85_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 6, 10, 12 |

### Stage 3: summary

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 6 | every collection side by side: results/summary.*, the main figure, REPORT.md <br>`summary` | CPU | minutes | 2–5 | – |

### Stage 4: retrainings on each collection's DAgger data, and their evaluations

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 7 | retrain on MoE seed 0, VIG: take over above p90, hand back below p70 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes with the random-move steps cut out (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p90_r70_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–24 h | 2 | 12 |
| 8 | retrain on MoE seed 0, VIG: take over above p90, hand back below p75 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes with the random-move steps cut out (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p90_r75_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–24 h | 3 | 12 |
| 9 | retrain on MoE seed 0, VIG: take over above p90, hand back below p80 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes with the random-move steps cut out (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p90_r80_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–24 h | 4 | 12 |
| 10 | retrain on MoE seed 0, VIG: take over above p90, hand back below p85 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes with the random-move steps cut out (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p90_r85_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–24 h | 5 | 12 |

### Stage 5: the MoEs evaluated at full resolution, for comparison

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 11 | evaluate MoE seed 0 at full resolution on 200 scenes (the comparison for the retrained policies) <br>`eval/moe/sd0` | CPU cores + a GPU | under 1 h | 1 | 12 |

### Stage 6: final report

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 12 | final report: the evaluations next to the collections and the reference (REPORT.md, retraining.png) <br>`final` | CPU | minutes | 2–5, 7–11 | – |

