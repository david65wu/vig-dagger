# Tasks of configs/plugcharger.yaml

The experiment's work as 80 numbered tasks (41–80, the retrainings and evaluations, were added later; 1–40 kept
their numbers). A task's inputs always come from lower numbers. With a task board
(`sharing.board`, see [AGENTS.md](AGENTS.md#several-machines)) the machines claim these tasks themselves, most urgent
first; without one, the humans assign them by number: `vig-dagger run -c configs/run.yaml --tasks N ...` runs tasks,
`vig-dagger pack --tasks N --for M` / `vig-dagger unpack FILE` move outputs between machines. `vig-dagger tasks
-c configs/run.yaml` prints this list. Times are rough, for a shared 64-core machine with 2 GPUs.

### Stage 1: the MoEs and their thresholds

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 1 | MoE seed 0: download it (Hugging Face), score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd0` | 1 GPU (inference) | 0.5–1 h | – | 4–15, 77 |
| 2 | MoE seed 1: train it (60 epochs on 1000 demos), then score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd1` | 1 GPU (training, then inference) | 8–14 h | – | 16–27, 78 |
| 3 | MoE seed 2: train it (60 epochs on 1000 demos), then score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd2` | 1 GPU (training, then inference) | 8–14 h | – | 28–39, 79 |

### Stage 2: DAgger collections

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 4 | MoE seed 0, VIG at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40–41, 80 |
| 5 | MoE seed 0, VIG at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 42, 80 |
| 6 | MoE seed 0, VIG at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 43, 80 |
| 7 | MoE seed 0, flow loss at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 44, 80 |
| 8 | MoE seed 0, flow loss at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 45, 80 |
| 9 | MoE seed 0, flow loss at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 46, 80 |
| 10 | MoE seed 0, VIG at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 47, 80 |
| 11 | MoE seed 0, VIG at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 48, 80 |
| 12 | MoE seed 0, VIG at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 49, 80 |
| 13 | MoE seed 0, flow loss at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 50, 80 |
| 14 | MoE seed 0, flow loss at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 51, 80 |
| 15 | MoE seed 0, flow loss at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40, 52, 80 |
| 16 | MoE seed 1, VIG at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 53, 80 |
| 17 | MoE seed 1, VIG at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 54, 80 |
| 18 | MoE seed 1, VIG at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 55, 80 |
| 19 | MoE seed 1, flow loss at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 56, 80 |
| 20 | MoE seed 1, flow loss at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 57, 80 |
| 21 | MoE seed 1, flow loss at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 58, 80 |
| 22 | MoE seed 1, VIG at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 59, 80 |
| 23 | MoE seed 1, VIG at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 60, 80 |
| 24 | MoE seed 1, VIG at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 61, 80 |
| 25 | MoE seed 1, flow loss at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 62, 80 |
| 26 | MoE seed 1, flow loss at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 63, 80 |
| 27 | MoE seed 1, flow loss at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40, 64, 80 |
| 28 | MoE seed 2, VIG at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 65, 80 |
| 29 | MoE seed 2, VIG at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 66, 80 |
| 30 | MoE seed 2, VIG at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 67, 80 |
| 31 | MoE seed 2, flow loss at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 68, 80 |
| 32 | MoE seed 2, flow loss at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 69, 80 |
| 33 | MoE seed 2, flow loss at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 70, 80 |
| 34 | MoE seed 2, VIG at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 71, 80 |
| 35 | MoE seed 2, VIG at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 72, 80 |
| 36 | MoE seed 2, VIG at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 73, 80 |
| 37 | MoE seed 2, flow loss at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 74, 80 |
| 38 | MoE seed 2, flow loss at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 75, 80 |
| 39 | MoE seed 2, flow loss at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40, 76, 80 |

### Stage 3: summary

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 40 | every collection side by side: results/summary.*, the main figure, REPORT.md <br>`summary` | CPU | minutes | 4–39 | – |

### Stage 4: retrainings on each collection's DAgger data, and their evaluations

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 41 | retrain on MoE seed 0, VIG at p90: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p90_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 4 | 80 |
| 42 | retrain on MoE seed 0, VIG at p95: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p95_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 5 | 80 |
| 43 | retrain on MoE seed 0, VIG at p99: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p99_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 6 | 80 |
| 44 | retrain on MoE seed 0, flow loss at p90: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_flow_p90_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 7 | 80 |
| 45 | retrain on MoE seed 0, flow loss at p95: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_flow_p95_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 8 | 80 |
| 46 | retrain on MoE seed 0, flow loss at p99: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_flow_p99_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 9 | 80 |
| 47 | retrain on MoE seed 0, VIG at p90 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p90_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 10 | 80 |
| 48 | retrain on MoE seed 0, VIG at p95 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p95_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 11 | 80 |
| 49 | retrain on MoE seed 0, VIG at p99 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_vig_p99_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 12 | 80 |
| 50 | retrain on MoE seed 0, flow loss at p90 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_flow_p90_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 13 | 80 |
| 51 | retrain on MoE seed 0, flow loss at p95 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_flow_p95_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 14 | 80 |
| 52 | retrain on MoE seed 0, flow loss at p99 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd0_flow_p99_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 15 | 80 |
| 53 | retrain on MoE seed 1, VIG at p90: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_vig_p90_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 16 | 80 |
| 54 | retrain on MoE seed 1, VIG at p95: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_vig_p95_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 17 | 80 |
| 55 | retrain on MoE seed 1, VIG at p99: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_vig_p99_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 18 | 80 |
| 56 | retrain on MoE seed 1, flow loss at p90: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_flow_p90_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 19 | 80 |
| 57 | retrain on MoE seed 1, flow loss at p95: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_flow_p95_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 20 | 80 |
| 58 | retrain on MoE seed 1, flow loss at p99: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_flow_p99_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 21 | 80 |
| 59 | retrain on MoE seed 1, VIG at p90 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_vig_p90_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 22 | 80 |
| 60 | retrain on MoE seed 1, VIG at p95 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_vig_p95_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 23 | 80 |
| 61 | retrain on MoE seed 1, VIG at p99 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_vig_p99_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 24 | 80 |
| 62 | retrain on MoE seed 1, flow loss at p90 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_flow_p90_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 25 | 80 |
| 63 | retrain on MoE seed 1, flow loss at p95 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_flow_p95_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 26 | 80 |
| 64 | retrain on MoE seed 1, flow loss at p99 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd1_flow_p99_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 27 | 80 |
| 65 | retrain on MoE seed 2, VIG at p90: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_vig_p90_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 28 | 80 |
| 66 | retrain on MoE seed 2, VIG at p95: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_vig_p95_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 29 | 80 |
| 67 | retrain on MoE seed 2, VIG at p99: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_vig_p99_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 30 | 80 |
| 68 | retrain on MoE seed 2, flow loss at p90: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_flow_p90_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 31 | 80 |
| 69 | retrain on MoE seed 2, flow loss at p95: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_flow_p95_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 32 | 80 |
| 70 | retrain on MoE seed 2, flow loss at p99: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_flow_p99_plain` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 33 | 80 |
| 71 | retrain on MoE seed 2, VIG at p90 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_vig_p90_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 34 | 80 |
| 72 | retrain on MoE seed 2, VIG at p95 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_vig_p95_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 35 | 80 |
| 73 | retrain on MoE seed 2, VIG at p99 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_vig_p99_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 36 | 80 |
| 74 | retrain on MoE seed 2, flow loss at p90 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_flow_p90_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 37 | 80 |
| 75 | retrain on MoE seed 2, flow loss at p95 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_flow_p95_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 38 | 80 |
| 76 | retrain on MoE seed 2, flow loss at p99 + random move: a dense full-resolution policy on the 1000 base demos + its 500 DAgger episodes (60 epochs, seed 0), then evaluate it on 200 scenes <br>`retrain/sd2_flow_p99_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 8–15 h | 39 | 80 |

### Stage 5: the MoEs evaluated at full resolution, for comparison

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 77 | evaluate MoE seed 0 at full resolution on 200 scenes (the comparison for the retrained policies) <br>`eval/moe/sd0` | CPU cores + a GPU | under 1 h | 1 | 80 |
| 78 | evaluate MoE seed 1 at full resolution on 200 scenes (the comparison for the retrained policies) <br>`eval/moe/sd1` | CPU cores + a GPU | under 1 h | 2 | 80 |
| 79 | evaluate MoE seed 2 at full resolution on 200 scenes (the comparison for the retrained policies) <br>`eval/moe/sd2` | CPU cores + a GPU | under 1 h | 3 | 80 |

### Stage 6: final report

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 80 | final report: the evaluations next to the collections (REPORT.md, eval_success.png) <br>`final` | CPU | minutes | 4–39, 41–79 | – |

