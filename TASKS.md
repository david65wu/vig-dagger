# Tasks of configs/plugcharger.yaml

The experiment's work as 40 numbered tasks, for several machines with the tasks assigned by hand (see
[AGENTS.md](AGENTS.md#several-machines)). A task's inputs always come from lower numbers. `vig-dagger tasks -c configs/run.yaml`
prints this list; `vig-dagger run -c configs/run.yaml --tasks N ...` runs tasks; `vig-dagger pack --tasks N --for M` /
`vig-dagger unpack FILE` move outputs between machines. Times are rough, for a shared 64-core machine with 2 GPUs.

### Stage 1: the MoEs and their thresholds

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 1 | MoE seed 0: download it (Hugging Face), score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd0` | 1 GPU (inference) | 0.5–1 h | – | 4–15 |
| 2 | MoE seed 1: train it (60 epochs on 1000 demos), then score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd1` | 1 GPU (training, then inference) | 8–14 h | – | 16–27 |
| 3 | MoE seed 2: train it (60 epochs on 1000 demos), then score 50 demos for its VIG and flow-loss percentiles <br>`moe/sd2` | 1 GPU (training, then inference) | 8–14 h | – | 28–39 |

### Stage 2: DAgger collections

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 4 | MoE seed 0, VIG at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 5 | MoE seed 0, VIG at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 6 | MoE seed 0, VIG at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 7 | MoE seed 0, flow loss at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 8 | MoE seed 0, flow loss at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 9 | MoE seed 0, flow loss at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 10 | MoE seed 0, VIG at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 11 | MoE seed 0, VIG at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 12 | MoE seed 0, VIG at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_vig_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 13 | MoE seed 0, flow loss at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 14 | MoE seed 0, flow loss at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 15 | MoE seed 0, flow loss at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd0_flow_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 40 |
| 16 | MoE seed 1, VIG at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 17 | MoE seed 1, VIG at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 18 | MoE seed 1, VIG at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 19 | MoE seed 1, flow loss at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 20 | MoE seed 1, flow loss at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 21 | MoE seed 1, flow loss at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 22 | MoE seed 1, VIG at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 23 | MoE seed 1, VIG at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 24 | MoE seed 1, VIG at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_vig_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 25 | MoE seed 1, flow loss at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 26 | MoE seed 1, flow loss at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 27 | MoE seed 1, flow loss at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd1_flow_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2 | 40 |
| 28 | MoE seed 2, VIG at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 29 | MoE seed 2, VIG at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 30 | MoE seed 2, VIG at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 31 | MoE seed 2, flow loss at p90: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p90_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 32 | MoE seed 2, flow loss at p95: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p95_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 33 | MoE seed 2, flow loss at p99: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p99_plain` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 34 | MoE seed 2, VIG at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 35 | MoE seed 2, VIG at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 36 | MoE seed 2, VIG at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_vig_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 37 | MoE seed 2, flow loss at p90 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p90_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 38 | MoE seed 2, flow loss at p95 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p95_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |
| 39 | MoE seed 2, flow loss at p99 + random move: 500 successful DAgger episodes and their metrics <br>`dagger/sd2_flow_p99_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 3 | 40 |

### Stage 3: summary

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 40 | every collection side by side: results/summary.*, the main figure, REPORT.md <br>`summary` | CPU | minutes | 4–39 | – |

