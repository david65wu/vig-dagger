# Running the experiment: instructions for an agent

These instructions are for an agent (or a person) who runs `configs/plugcharger.yaml` (branch
`plugcharger-iterate-adaptive`) from start to finish without supervision, on one machine. The README explains what
the pipeline does; this file says how to operate it: what to run, what a healthy run looks like, when to stop and
ask, and what never to change.

**Running a row of the threshold sweep** (`configs/threshold_planner750.yaml`) instead? Follow
[MENTOR_750.md](MENTOR_750.md); this file does not apply to it.

## Short commands

The humans start and check the run with short messages. The wording varies ("start the experiment", "how is it
going"); go by the intent, and ask only if you can't tell which of these is meant:

| the human says | do |
|---|---|
| "run plugcharger" | the whole experiment on this machine: set up, check and launch (steps 0–4), then report |
| "status" | `vig-dagger status -c configs/run.yaml`, `tail run.log`, `df -h <work_dir>`; summarize: queue alive, failed jobs, the round in progress, each collection's successes and where its threshold is |
| "report" | `vig-dagger report -c configs/run.yaml`, then send `results/REPORT.md`, `results/figures/collections.png`, `retraining.png`, `success_by_demos.png`, `thresholds.png`, `rounds.png`, `distributions.png` and `moves/spheres_held_rounds.png` and summarize them (step 7) |
| "videos" | `vig-dagger videos -c configs/run.yaml` (renders the missing videos of every collection with attempts, also one still running; videos already there are kept), then say how many there are per collection and where (`<work_dir>/videos/<c>/index.json`) |
| "stop" | `vig-dagger stop -c configs/run.yaml` (every job pauses; the same "run" again resumes) |
| "update" (or "git pull and read UPDATES.md") | `git pull`, then read [UPDATES.md](UPDATES.md) and do what its entries newer than this machine's previous commit say for this machine (the humans approved those steps); then start the queue again (step 4) and report |

If an earlier session already set up and launched this machine (there is a `configs/run.yaml` and a work directory),
"run plugcharger" means: start the queue again if it is not running (step 4), then report the status. Never
reinstall or reconfigure a machine that has started.

## The goal and what "done" means

- **The run:** PlugCharger in ManiSkill, 4 rounds of DAgger starting from **50 planner demos**, collected here by the
  planner moving like the DAgger expert (`demos.motion: fast_maniskill`). Round r trains a 7-level resolution MoE on
  the 50 base demos + the DAgger data of rounds 1 … r-1 (random-move steps cut), scores every 4th frame of all its
  training demos with VIG (its **reference**), and collects **100 / 200 / 400 / 800** successful DAgger episodes with
  the **'fast_maniskill' expert** and a random 5 cm / 5° gripper move at every takeover. One threshold serves both
  ways (take over after 2 VIG scores above it, hand back after 16 below) and **adapts after every attempt** (README,
  "Iteration mode"): round 1 starts at p80 of its reference, every later round where the previous one ended. After
  each round a dense full-resolution policy is retrained on the base demos + every round's DAgger data so far (60
  epochs, training seed 0) and evaluated on 200 scenes next to the round's MoE. The comparison (the same dense policy
  on as many planner demos) runs on another machine (`configs/planner_reference.yaml`); keep
  `iterate.planner_baselines: false` here.
- **The results:** per round the threshold's path and where it ended, the collection's success rate and expert share,
  the dense policy's success (and paired tests against the previous round's and the round's MoE).
- **Done when:**
  - `vig-dagger status` shows every job `done`;
  - `<work_dir>/results/REPORT.md` has both tables filled in for all 4 rounds, and `results/figures/collections.png`,
    `retraining.png`, `success_by_demos.png`, `thresholds.png`, `rounds.png`, `distributions.png` and
    `moves/spheres_{held,progress}_rounds.png` exist;
  - `<work_dir>/videos/<c>/` has 20 successful and 20 failed attempts for each round.

Every result is written to files in `<work_dir>/results`; W&B is not used (`wandb.enabled: false`). Expect about 2–3
days on one machine with one free GPU (most of it round 4), less with two GPUs.

## Rules

**Never, without explicit approval from the humans who own the experiment:**

- **Change an experiment parameter**, in the YAML file or with `--set`: anything under `demos`, `training`, `moe`,
  `scorers`, `thresholds`, `iterate`, `dagger`, `retrain`, `evaluation`. These define the experiment. A changed value
  either silently mixes two experiments, or the pipeline refuses to continue.
- **Delete or move anything in `work_dir`**, above all `dagger/` (with each collection's `threshold_state.json`),
  `policies/`, `data/`. Collections take hours; partial outputs are what lets a run resume, and a later round starts
  where the previous round's threshold ended.
- **Upload** anything to Hugging Face, **push** to git, or turn on W&B.
- **Run two queues on the same `work_dir`**, or start workers by hand outside the queue.
- **Edit the code**, including the planner, the random move, the environment and the threshold rule.

**You may:**

- set `work_dir` before the first start;
- change `resources.*` settings, e.g. fewer `sim_slots` when the machine is shared, or a higher `max_load` when other
  users' jobs keep the load average above the core count (the queue starts nothing new above it). To apply them
  without interrupting jobs: `vig-dagger stop --queue-only`, edit the config, then `vig-dagger run` again;
- stop and restart the queue (`vig-dagger stop`, `vig-dagger run`);
- rerun `vig-dagger report` and `vig-dagger videos` at any time;
- read any file or log.

**Stop and ask the humans when:**

- `vig-dagger check` reports a failure you cannot fix by installing something: no Vulkan (or only a software driver),
  not enough disk, no internet;
- a job is `FAILED` again after you fixed what its log showed and restarted once, or you can't tell why it failed;
- a collection is heading for its attempt limit (20 × its target): after 300 attempts it has fewer than 15 successes;
- a collection's threshold sits at p0 or p100 for more than 50 attempts (`vig-dagger status` shows where it is);
- free disk in `work_dir` drops below 30 GB;
- a training's loss is NaN, or its validation loss rises steadily over the second half of training;
- a milestone value is far outside the expected range below.

## 0. Before you start

**Everything else has a default.** Use it unless the human's message says otherwise, and say in your first report
which values you used:

| setting | default |
|---|---|
| work directory | `~/vig-dagger-runs/plugcharger_iterate_adaptive` if that disk has room (about 100 GB: every attempt is saved, failed ones included); otherwise a directory you can write to on a disk that has; if there is none, ask |
| GPUs | every GPU (`resources.gpus: auto`) |
| CPU budget | the default (`resources.sim_slots: auto`, half the cores' worth of simulator jobs). If other users' jobs already load the machine (`uptime` load average above half of `nproc`), halve it and say so |
| reporting | a message after the launch, after each round's collection, and the final report (step 7) |

Say in your first report that the run takes days and keeps going after your session ends; the humans check on it
with "status".

**On a rented cloud machine** (e.g. a container on vast.ai): the simulator renders with Vulkan on the NVIDIA GPU, and
containers often expose only CUDA. The container needs `NVIDIA_DRIVER_CAPABILITIES=all` (set when the instance is
created). `vig-dagger check` tells you (step 3); if it fails on rendering, ask.

## 1. Setup

```bash
git clone https://github.com/david65wu/vig-dagger.git && cd vig-dagger
git checkout plugcharger-iterate-adaptive
uv sync
source .venv/bin/activate
```

Nothing is downloaded from Hugging Face except the DINOv2 weights (public, by the first training): the base demos
are collected and every MoE is trained here. `uv` and `ffmpeg` are the only programs to install. If either is missing
and installing it needs administrator rights, ask.

## 2. Configure

Copy `configs/plugcharger.yaml` to `configs/run.yaml`. Set `work_dir` (step 0). If the human named GPUs or a CPU
budget, set `resources.gpus` / `resources.sim_slots`. Leave everything else as it is.

## 3. Preflight

```bash
vig-dagger check -c configs/run.yaml
```

Every line must say `ok`; `WARN` lines are acceptable. It checks free disk, `ffmpeg`, the GPUs, that torch sees CUDA,
that the simulator renders (Vulkan) on a GPU, that the DINOv2 weights are reachable, and the W&B setting (it should
say `W&B disabled`).

On a new machine, also run the smoke test (20–40 minutes, writes only to the given directory):

```bash
vig-dagger run -c configs/smoke.yaml --set work_dir=/tmp/vig-dagger-smoke
```

It runs 2 tiny rounds with every kind of job and must end with `[queue] finished: 45 of 45 jobs done`. The smoke
MoEs are barely trained, so their episodes are mostly the expert's; that is expected. Delete `/tmp/vig-dagger-smoke`
afterwards.

## 4. Launch

```bash
setsid nohup vig-dagger run -c configs/run.yaml > run.log 2>&1 < /dev/null &
```

`run.log` is the queue's log: every job it starts, finishes, retries or gives up on. Each job writes its own log to
`<work_dir>/logs/<job name>.log`.

## 5. Monitor

About every hour:

```bash
vig-dagger status -c configs/run.yaml    # queue alive? failed and running jobs, progress, milestones
tail -n 30 run.log
df -h <work_dir>
```

After each round's collection, run `vig-dagger report -c configs/run.yaml`: it refreshes `results/REPORT.md` and the
figures with everything finished so far.

If the queue is NOT RUNNING and jobs remain, start it again with the command from step 4. It takes over jobs that
are still running.

### Milestones and expected values

| milestone | when | what to check | expected |
|---|---|---|---|
| base demos | first ~30 min | `<work_dir>/data/raw/planner/part_000.done.json` | 50 demos, `"motion": "fast_maniskill"`; demos ~130–160 steps (the last 30 a hold with the plug in the socket) |
| base dataset | right after | `num_demos` in `data/datasets/base/dataset_manifest.json` | 50 |
| round r MoE | ~minutes (round 1) to hours (round 4) | `policies/moe/round<r>/run_config.json` → `_info`, `training.max_updates`; `metrics.jsonl` | training demos grow round by round (round 1: 49 of 50, one held out for validation); round 4 about 127,000 updates; validation loss falling |
| round r reference | after its MoE | `policies/moe/round<r>/thresholds.json` | `frames` = every 4th frame of all its training demos (round 1 ~1,700; round 4 tens of thousands), `vig_mean_percentile` (the target) somewhere around p55–p65 |
| collections | after each reference | `dagger/<c>/settings.json` → `adaptive`; `status` progress and "threshold now pNN" | `start_percentile` 80 in round 1, the previous round's end later; `seed_start` 1,000,000 + 100,000 × (r − 1); the threshold moves most in the first attempts and then settles (on a 1000-demo MoE it settled near p88–90 within 10–15 attempts); successes rising steadily |
| collection metrics | as each collection finishes | `results/collections/<c>.json`; `results/REPORT.md`; `results/figures/distributions.png` | the experiment's new numbers; a new panel per round |
| videos | minutes after each collection's metrics | `videos/<c>/index.json`, `videos/<c>/{success,failed}/*.mp4` | 20 + 20 videos (fewer failed ones only if the collection had fewer failed attempts) |
| round datasets | after each collection | `logs/round_dataset_<r+1>.log` | "dagger_<c>: <target> episodes", "random moves cut: …", "round<r+1>_train: <n> episodes" |
| dense policies | after each round | `policies/retrain/<c>/seed0/run_config.json`; no `moe` in `policy.head` | validation loss falling; after round 4 about 270,000 updates |
| evaluations | after each training | `eval/<policy>/level1/*.jsonl` | 200 episodes per policy (8 files of 25): 4 dense policies and 4 MoEs |
| final report | at the end | `results/REPORT.md`, `results/iterate_summary.json` | 4 rounds in both tables |

## 6. When something goes wrong

| symptom | likely cause | what to do |
|---|---|---|
| `status` lists a FAILED job | see its log (path in `status`) | fix the cause if it is in the "you may" list, then `vig-dagger run` again; otherwise ask |
| a job retries again and again | the same error each time | read the log; don't wait for the retries to run out |
| nothing starts although jobs are ready | the load average is above `resources.max_load` (other users' jobs) | raise `resources.max_load` (allowed, see "You may") and restart the queue |
| `CUDA out of memory` in a training or worker log | other jobs filled the GPU | nothing; the retry runs when memory frees up. If it keeps happening, ask |
| `vk::`, "no device", rendering errors; `check` names llvmpipe or a CPU | no NVIDIA Vulkan driver in the machine or container | ask (needs `NVIDIA_DRIVER_CAPABILITIES=all` or a system fix) |
| `No space left on device` | disk full | stop the queue (`vig-dagger stop`) and ask; never delete experiment outputs to make room |
| queue NOT RUNNING, jobs remain | the queue process died (SSH session, reboot) | start it again (step 4) |
| machine rebooted | | start it again; trainings resume from their last checkpoint, collections from their saved episodes and threshold state |
| "was collected with different settings" in a `dagger_prepare_*` log | the config, the round's reference or the previous round's end changed after the collection started | restore the original config; ask |
| "has no threshold state yet" in a `dagger_prepare_*` log | the previous round's collection directory lacks `threshold_state.json` | ask (never recreate it by hand) |
| "the expert's strategy sampler does not draw strategies like the planner demos did" | the expert code or the demos differ from what the experiment assumes | stop and ask; never change the code |
| a collection's threshold stuck at p0 or p100 | the rule is not working for this MoE | ask |

**Reading a DAgger worker's log** (`logs/dagger_<c>_<w>.log`): one line per attempt, e.g.
`[seed 1000123] SUCCESS (success_expert) steps=231 expert=150 move=14 takeovers=1 returns=0 96.2s threshold p84.1->p84.6
(74 policy steps)`.

## 7. When it finishes

Run `vig-dagger report -c configs/run.yaml` once more and send the humans `<work_dir>/results/REPORT.md`,
`results/figures/collections.png`, `retraining.png`, `success_by_demos.png`, `thresholds.png`, `rounds.png`,
`distributions.png`, `moves/spheres_{held,progress}_rounds.png` and `results/iterate_summary.json` (or the whole `results/` folder); the videos are in `<work_dir>/videos/`. In your message, summarize:

1. **Per round:** where the threshold started and ended (percentile and VIG value), the target q(M), the collection's
   attempts and success rate, the expert and random-move shares of its DAgger data.
2. **The dense policies:** per round the success rate, and the paired comparisons with the previous round's policy
   and the round's MoE (only this / only the other, p).
3. **The distributions** (`distributions.png`, numbers in `iterate_summary.json` under `distribution`): per round
   the overlap of the policy steps' VIG with the training frames' VIG, the policy steps' mean percentile against
   the target q(M), and the expert steps' mean percentile.
4. **Anything unusual:** failures you fixed, restarts, jobs that needed several attempts, a threshold that did not
   settle.

## Reference

**Job names** (what `status`, `run.log` and `<work_dir>/logs/` use; `<c>` is `round<r>_sd0_vig_adaptive_random_move`):

| job | what it does |
|---|---|
| `collect_planner_<k>`, `convert_planner` | collect the 50 base demos, convert them to the base dataset |
| `round_dataset_<r>` | round r's training set: round r-1's + round r-1's DAgger data (random moves cut) |
| `train/moe/round<r>` | round r's MoE |
| `score_round<r>_<chunk>`, `thresholds_round<r>` | its VIG on every 4th training frame: the reference and the target |
| `dagger_prepare_<c>` | check the expert against the base demos, write `settings.json` (reference, target, start) |
| `dagger_<c>_<worker>` | DAgger workers, sharing one collection and its threshold |
| `analyze_<c>` | the collection's metrics |
| `videos_<c>` | videos of 20 successful and 20 failed attempts (`videos/<c>/`); nothing waits for it |
| `train/retrain/<c>/seed0` | the dense policy after round r (trains on `round<r+1>_train`) |
| `eval/<policy>/<first scene>` | 25 evaluation episodes of a dense policy or a round's MoE |
| `summarize`, `final_report` | REPORT.md, iterate_summary.json, the figures |

**Useful commands:**

- `vig-dagger run -c … --dry-run`: every job with its priority and dependencies (`done` marks finished ones).
- `vig-dagger stop -c …`: stop everything. With `--queue-only`, only the queue stops.
- `vig-dagger show-config -c …`: the resolved configuration.
- `vig-dagger tasks -c …`: the numbered tasks ([TASKS.md](TASKS.md)).

**Where things are:** see "Output directory" in the README. A collection's threshold: `dagger/<c>/threshold_state.json`
(now) and `threshold_log.jsonl` (every update).
