# Reference: hand-back at p90 (`sd0_vig_p90_random_move`)

A collection made before this branch with the same MoE, expert and DAgger settings as this experiment's collections,
except the hand-back threshold: the policy takes back control below **p90**, the take-over threshold itself (one
threshold both ways). It is the p90 end of the sweep (p70, p75, p80, p85 here) and is shown next to it as
*[reference]*. `reference.py` uses it only while it matches this experiment (settings, take-over threshold within
0.5 %, and the collection code below unchanged) and lists it as excluded, with the reasons, otherwise.

**Collection** (2026-11-17, `vig-dagger run --only '^analyze_'` on branch `plugcharger-moe-seeds`, commit 8139bf9, 4
workers, so the scene sequence 1,000,000 + w + 4k that this experiment's collections run too):

- MoE: `chengenwu/sd0_plug_charger_hole4mm_sidecam_randsocket_vargrasp_1000mp_full_resmoe64_sd0` (the 7-level MoE on the
  1000 planner demos, downloaded from Hugging Face).
- VIG at p90 = 0.0260371 (the MoE's p90 on every 4th frame of the first 50 planner demos) for take-over (2 scores above)
  and hand-back (16 below); expert `demo` (the planner demos' strategy sampler and speeds); a random 5 cm / 5° move at
  every take-over; 400-step episodes; failed attempts saved (does not change the episodes).
- 502 successes in 622 attempts (80.7 %); the first 500 to finish are the DAgger data.
- `settings.json`: its settings; `collection.json`: its metrics from this branch's analysis (also the expert share of
  all attempts: 37.2 % + 16.7 % random move); `code.json`: sha256 of the collection code at 8139bf9.

**Retrained policy** (`retrain.json`): the 1000 base demos + those 500 episodes with the random-move steps cut out, a
dense full-resolution policy, 60 epochs, seed 0, made with maniskill's pipeline, whose converter output, random-move
cut and training config are this branch's. `eval/`: its 200 evaluation episodes from this repo's `workers/evaluate.py`
on this experiment's scenes (60000-60199): 77 / 200 = 38.5 %. (maniskill's evaluator gave 83 / 200 on the same scenes;
the two evaluators differ by a few points, so only this repo's numbers sit next to this experiment's.)
