# Tasks of configs/plugcharger.yaml

The experiment's work as 22 numbered tasks, round by round: round r's MoE and its reference, its collection, the dense
policy retrained after it, the dense policy on as many planner demos (the comparison), and the MoE's evaluation; then
the summary and the final report. A task's inputs always come from lower numbers. The experiment runs on one machine:
`vig-dagger run -c configs/run.yaml` runs all of them, in this order as far as their inputs allow (round r's
retraining next to round r+1's MoE and collection). `vig-dagger tasks -c configs/run.yaml` prints this list. Times are
rough.

### Stage 1: the MoEs and their thresholds

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 1 | round 1 MoE: train it (60 epochs on the base demos), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round1` | 1 GPU (training, then inference) | 2–14 h | – | 2, 5 |
| 6 | round 2 MoE: train it (60 epochs on the base demos + the DAgger data of round 1), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round2` | 1 GPU (training, then inference) | 2–14 h | 2 | 7, 10 |
| 11 | round 3 MoE: train it (60 epochs on the base demos + the DAgger data of rounds 1–2), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round3` | 1 GPU (training, then inference) | 2–14 h | 2, 7 | 12, 15 |
| 16 | round 4 MoE: train it (60 epochs on the base demos + the DAgger data of rounds 1–3), then score every 4th frame of its training demos (the adaptive threshold's reference) <br>`moe/round4` | 1 GPU (training, then inference) | 2–14 h | 2, 7, 12 | 17, 20 |

### Stage 2: DAgger collections

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 2 | round 1: round-1 MoE, VIG at an adaptive threshold + random move: 100 successful DAgger episodes and their metrics <br>`dagger/round1_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 1 | 3, 6–8, 11, 13, 16, 18, 21–22 |
| 7 | round 2: round-2 MoE, VIG at an adaptive threshold + random move: 200 successful DAgger episodes and their metrics <br>`dagger/round2_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 2, 6 | 8, 11–13, 16, 18, 21–22 |
| 12 | round 3: round-3 MoE, VIG at an adaptive threshold + random move: 400 successful DAgger episodes and their metrics <br>`dagger/round3_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 7, 11 | 13, 16–18, 21–22 |
| 17 | round 4: round-4 MoE, VIG at an adaptive threshold + random move: 800 successful DAgger episodes and their metrics <br>`dagger/round4_sd0_vig_adaptive_random_move` | GPU + as many CPU cores as possible (simulator workers) | 20–60 worker-hours | 12, 16 | 18, 21–22 |

### Stage 3: summary

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 21 | every collection side by side: results/summary.*, the main figure, REPORT.md <br>`summary` | CPU | minutes | 2, 7, 12, 17 | – |

### Stage 4: retrainings on each collection's DAgger data, and their evaluations

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 3 | after round 1: a dense full-resolution policy on the 50 base demos + the 100 DAgger episodes of round 1 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round1_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2 | 4, 22 |
| 8 | after round 2: a dense full-resolution policy on the 50 base demos + the 300 DAgger episodes of rounds 1–2 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round2_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2, 7 | 9, 22 |
| 13 | after round 3: a dense full-resolution policy on the 50 base demos + the 700 DAgger episodes of rounds 1–3 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round3_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2, 7, 12 | 14, 22 |
| 18 | after round 4: a dense full-resolution policy on the 50 base demos + the 1500 DAgger episodes of rounds 1–4 with the random-move steps cut out (60 epochs), then evaluate it on 200 scenes <br>`retrain/round4_sd0_vig_adaptive_random_move` | 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 2, 7, 12, 17 | 19, 22 |

### Stage 5: dense policies on planner demos only, as many as each round's dense policy trains on

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 4 | planner comparison for round 1: a dense full-resolution policy on 150 planner demos (the 50 base demos + 100 more collected like them), as many as the dense policy after round 1 trains on (60 epochs; after that policy), then evaluate it on 200 scenes <br>`planner/round1` | CPU cores (collecting the demos), 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 3 | 22 |
| 9 | planner comparison for round 2: a dense full-resolution policy on 350 planner demos (the 50 base demos + 300 more collected like them), as many as the dense policy after round 2 trains on (60 epochs; after that policy), then evaluate it on 200 scenes <br>`planner/round2` | CPU cores (collecting the demos), 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 8 | 22 |
| 14 | planner comparison for round 3: a dense full-resolution policy on 750 planner demos (the 50 base demos + 700 more collected like them), as many as the dense policy after round 3 trains on (60 epochs; after that policy), then evaluate it on 200 scenes <br>`planner/round3` | CPU cores (collecting the demos), 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 13 | 22 |
| 19 | planner comparison for round 4: a dense full-resolution policy on 1550 planner demos (the 50 base demos + 1500 more collected like them), as many as the dense policy after round 4 trains on (60 epochs; after that policy), then evaluate it on 200 scenes <br>`planner/round4` | CPU cores (collecting the demos), 1 GPU (training), then CPU cores + a GPU (evaluation) | 2–24 h | 18 | 22 |

### Stage 6: the MoEs evaluated at full resolution, for comparison

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 5 | evaluate the round1 MoE at full resolution on 200 scenes <br>`eval/moe/round1` | CPU cores + a GPU | under 1 h | 1 | 22 |
| 10 | evaluate the round2 MoE at full resolution on 200 scenes <br>`eval/moe/round2` | CPU cores + a GPU | under 1 h | 6 | 22 |
| 15 | evaluate the round3 MoE at full resolution on 200 scenes <br>`eval/moe/round3` | CPU cores + a GPU | under 1 h | 11 | 22 |
| 20 | evaluate the round4 MoE at full resolution on 200 scenes <br>`eval/moe/round4` | CPU cores + a GPU | under 1 h | 16 | 22 |

### Stage 7: final report

| # | task | runs on | time | needs tasks | used by tasks |
|---|---|---|---|---|---|
| 22 | final report: the evaluations next to the collections (REPORT.md, retraining.png) <br>`final` | CPU | minutes | 2–5, 7–10, 12–15, 17–20 | – |

