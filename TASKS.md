# Tasks of configs/plugcharger.yaml

The experiment's work as 18 numbered tasks, round by round: round r's MoE and its reference, its collection, the dense
policy retrained after it, and the MoE's evaluation; then the summary and the final report. A task's inputs always
come from lower numbers. The experiment runs on one machine: `vig-dagger run -c configs/run.yaml` runs all of them, in
this order as far as their inputs allow (round r's retraining next to round r+1's MoE and collection). `vig-dagger
tasks -c configs/run.yaml` prints this list. Times are rough.

### Stage 1: the MoEs and their thresholds

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 1 | round 1 MoE: train it (60 epochs on the base demos), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round1` | 1 GPU (training, then inference) | 2–14 h | – | 2, 4 |
| 5 | round 2 MoE: train it (60 epochs on the base demos + the DAgger data of round 1), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round2` | 1 GPU (training, then inference) | 2–14 h | 2 | 6, 8 |
| 9 | round 3 MoE: train it (60 epochs on the base demos + the DAgger data of rounds 1–2), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round3` | 1 GPU (training, then inference) | 2–14 h | 2, 6 | 10, 12 |
| 13 | round 4 MoE: train it (60 epochs on the base demos + the DAgger data of rounds 1–3), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round4` | 1 GPU (training, then inference) | 2–14 h | 2, 6, 10 | 14, 16 |

### Stage 2: DAgger collections

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 2 | round 1: round-1 MoE, VIG at an adaptive threshold + random move: 100 successful DAgger episodes and their metrics <br>`dagger/round1_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 3, 5–7, 9, 11, 13, 15, 17–18 |
| 6 | round 2: round-2 MoE, VIG at an adaptive threshold + random move: 200 successful DAgger episodes and their metrics <br>`dagger/round2_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2, 5 | 7, 9–11, 13, 15, 17–18 |
| 10 | round 3: round-3 MoE, VIG at an adaptive threshold + random move: 400 successful DAgger episodes and their metrics <br>`dagger/round3_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 6, 9 | 11, 13–15, 17–18 |
| 14 | round 4: round-4 MoE, VIG at an adaptive threshold + random move: 800 successful DAgger episodes and their metrics <br>`dagger/round4_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 10, 13 | 15, 17–18 |

### Stage 3: summary

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 17 | every collection side by side: results/summary.*, the main figure, REPORT.md <br>`summary` | CPU | minutes | 2, 6, 10, 14 | – |

### Stage 4: retrainings on each collection's DAgger data, and their evaluations

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 3 | after round 1: a dense full-resolution policy on the 50 base demos + the 100 DAgger episodes of round 1 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round1_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2 | 18 |
| 7 | after round 2: a dense full-resolution policy on the 50 base demos + the 300 DAgger episodes of rounds 1–2 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round2_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2, 6 | 18 |
| 11 | after round 3: a dense full-resolution policy on the 50 base demos + the 700 DAgger episodes of rounds 1–3 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round3_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2, 6, 10 | 18 |
| 15 | after round 4: a dense full-resolution policy on the 50 base demos + the 1500 DAgger episodes of rounds 1–4 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round4_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2, 6, 10, 14 | 18 |

### Stage 5: the MoEs evaluated at full resolution, for comparison

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 4 | evaluate the round1 MoE at full resolution on 200 scenes <br>`eval/moe/round1` | CPU cores + a GPU | under 1 h | 1 | 18 |
| 8 | evaluate the round2 MoE at full resolution on 200 scenes <br>`eval/moe/round2` | CPU cores + a GPU | under 1 h | 5 | 18 |
| 12 | evaluate the round3 MoE at full resolution on 200 scenes <br>`eval/moe/round3` | CPU cores + a GPU | under 1 h | 9 | 18 |
| 16 | evaluate the round4 MoE at full resolution on 200 scenes <br>`eval/moe/round4` | CPU cores + a GPU | under 1 h | 13 | 18 |

### Stage 6: final report

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 18 | final report: the evaluations next to the collections (REPORT.md, retraining.png) <br>`final` | CPU | minutes | 2–4, 6–8, 10–12, 14–16 | – |
