# DAgger analysis specification

## Purpose and scope

Describe how DAgger collection behaves: how often it succeeds, when the expert intervenes, how much control it uses, how interventions end, and where failures occur. Produce a readable Markdown report, reproducible metric tables, figures, and videos linked to their source attempts.

This specification applies to **any nonempty set of variants**, including one, two, or three. A variant is one collection configuration, identified by a stable ID. It may differ in driver policy, scoring method, gate thresholds, or another declared collection parameter. Neither the number of variants nor their names, thresholds, quotas, camera count, or control rate is fixed by this specification.

Collection success is success of the combined policy–expert system. It must not be presented as the standalone policy evaluation success rate. Training schedules and standalone evaluations are outside this document's scope. Historical decisions for the current experiment are preserved in [DAGGER_RUN_NOTES.md](DAGGER_RUN_NOTES.md).

## 1. Variant selection and provenance

Use an explicit selected-variant list when supplied. Otherwise use the run configuration; if unavailable, discover collections with valid settings and label the scope as discovered. Do not silently infer missing members of a grid from naming conventions.

For every selected variant, record:

- Stable ID and readable display name; driver checkpoint identity and collection seed scheme.
- Scoring method and version, units, and direction of greater uncertainty.
- Takeover and return rules: percentile labels if applicable, resolved numeric thresholds, comparison operators, persistence counts, and any minimum intervention duration.
- Expert implementation/version, task/environment version, episode termination rules and maximum length.
- Collection quota or explicit completion signal, control period/timestamps, recorded cameras, and stage-definition version.
- Input snapshot time, completed attempt IDs included, exclusions, and analysis code/configuration version.

Do not merge attempts collected under incompatible settings under a single variant ID. Scene seeds can repeat across variants; use `(variant ID, attempt ID)` as the identity. If retries reuse a scene seed, distinguish them with a separate attempt ID. Deduplicate only duplicate records of the same attempt. Report conflicting duplicate records instead of silently choosing one.

A selected variant with no data must appear as **pending/no completed attempts**, not 0% success. Historical directories outside the selected set must not enter comparisons.

## 2. Units, alignment, and cohorts

An **attempt** is one rollout from reset to recorded termination. An **action step** is one executed action. A **takeover event** switches policy control to expert control; a **return event** switches expert control back to the policy. An **intervention** is one contiguous segment of expert-controlled actions.

For an attempt with `T` actions, observations may have length `T` or `T+1`. Action `t` is paired with its pre-action observation, score, stage, and controller. Use the final observation separately for terminal-stage reporting. Never shift all camera frames by one to compensate for a missing terminal frame.

Event records and executed segments are related but not interchangeable. A takeover can fail before producing its first action, and a return can occur at a terminal observation. Retain those events; identify zero-action interventions explicitly and exclude them only from executed-segment duration distributions. If an attempt begins under expert control, identify it as initial expert control rather than inventing a takeover event. Reconcile counts and flag unexplained discrepancies.

Use three primary cohorts **within each variant**:

1. **All completed attempts:** successes plus task failures, including overshoot beyond the collection quota.
2. **Successful completed attempts.**
3. **Failed completed attempts:** explicit task failure or episode timeout.

Track interrupted, incomplete, corrupt, and unknown-outcome recordings separately. Do not count an infrastructure interruption as task failure unless the experiment explicitly defines it that way. Preserve those artifacts and diagnostics when available.

Annotate the actual training subset using its selection manifest. If reconstructed from collection rules, label it inferred and record the exact ordering/tie-breaking rule. Do not assume every successful attempt was used for training or that failure retention authorizes failure use for training.

For every statistic, show the number of valid observations and excluded/missing observations. A zero denominator yields **N/A**, never zero or infinity. Missing data for one metric need not exclude the attempt from unrelated metrics.

## 3. Required metrics

Notation for attempt `i`: `T_i` = executed steps; `E_i` = expert-controlled steps; `P_i` = policy-controlled steps; `K_i` = takeover events; `R_i` = return events. With complete controller labels, `T_i = E_i + P_i`. Unknown-controller steps are reported separately; do not silently assign them to the policy.

Distributions must include valid count, mean, median, quartiles, minimum and maximum. Box plots show median and mean explicitly. Each table and plot states its cohort and observation unit.

| # | Metric | Definition and required presentation |
|---|---|---|
| 1 | Collection success rate | Successful completed attempts / completed attempts with known outcomes. Show numerator, denominator, percentage, and 95% Wilson interval. Compare variants using the same outcome definition; disclose differing termination limits. |
| 2 | Overall expert portion | `sum(E_i) / sum(T_i)` over attempts with complete controller records. Show pooled numerator/denominator and coverage. This weights attempts by length. |
| 3 | Expert portion per attempt | `E_i / T_i`. Mean, median, quartiles and box plot across attempts. Each attempt has equal weight. |
| 4 | Takeover frequency | `K_i` per attempt, plus `100 * K_i / P_i` per 100 policy-controlled steps. Show distributions and pooled `100 * sum(K_i) / sum(P_i)`. These are event/exposure rates, not probabilities, and need not be below 100. |
| 5 | Return frequency | `R_i` per attempt, plus `R_i / K_i` where defined and counts reconcile. Show distributions and the pooled fraction. Separately report the fraction of executed interventions that actually returned; disclose terminal/zero-action event differences. |
| 6 | Intervention duration | Steps and simulated seconds per executed intervention. Show a distribution **within each attempt**, and one pooled over interventions within each variant. Also show the distribution of each attempt's mean duration to distinguish equal-attempt weighting. Split by intervention outcome. Attempts with no interventions have N/A duration. |
| 7 | Takeover starts by stage | Per-attempt counts `K_i,s`, pooled stage counts, fraction of attempts visiting that stage with at least one takeover there, and event rates per 100 policy-controlled steps in that stage. Show per-attempt distributions and pooled summaries. |
| 8 | Expert portion by stage | `E_i,s / T_i,s` per attempt and `sum(E_i,s) / sum(T_i,s)` pooled. An unvisited stage has N/A portion. Divide the duration of a stage-spanning intervention across the actual stages of its actions. |
| 9 | Expert failures by stage | Counts of explicit expert failures by recorded failure stage and reason, per-attempt listings, and pooled stage/reason plots. Also show failed interventions in stage `s` / interventions reaching stage `s`, with numerator and denominator. Keep final stage, furthest stage and planner reason distinct; if exact failure stage is unavailable, label final stage as a proxy. |

For stage takeover counts, provide both unconditional counts over all valid attempts (unvisited stages contribute zero) and counts among attempts visiting the stage. Label them explicitly; use visited-stage cohorts for stage-specific box plots by default. Stage proportions and exposure rates always exclude absent/zero-exposure stages and display their valid sample counts.

“Pooled” means pooling within **one variant and one cohort** unless explicitly stated otherwise. A comparison figure places variants side by side; it does not silently merge their samples.

For success-rate intervals, disclose that Wilson intervals do not account for correlation between repeated scenes or policies. Partial collections stopped at a success quota are descriptive collection results, not unbiased policy evaluation trials.

## 4. Additional required diagnostics

### Intervention outcomes and repeated handoffs

Classify each intervention as returned to policy, success under expert control, explicit expert failure, timeout under expert control, or other/unknown termination. Distinguish the intervention outcome from the whole-attempt outcome: a returned intervention can belong to an attempt that eventually fails.

For each return, measure time until the next takeover, if any. Count re-takeovers within a declared window (default: one simulated second). Report the count, fraction of eligible returns, and observation coverage. Returns too close to termination to observe the whole window are censored; do not automatically count them as stable handoffs.

Terminal intervention durations are observed durations up to termination. They must not be described as completed recovery time or used to infer how quickly the expert would have recovered.

### Scores and gate decisions

Show actual decision-score distributions during policy and expert control, at takeover, and at return. Show both numeric thresholds and percentile names where relevant. Preserve missing/nonfinite score gaps and report coverage. A skipped score evaluation must not be represented as a fresh measured score.

Record the gate persistence rule so a threshold crossing is not mistaken for an immediate switch. Compare raw score magnitudes only across compatible score definitions/scales; otherwise show separate axes or panels.

### Collection cost and training selection

Report per-attempt wall time, simulated duration, summed attempt wall time, collection elapsed wall time, and attempts/time needed to reach the requested success quota. Separate startup, idle/scheduling, export and rendering time where records permit. Summed worker time is not parallel elapsed time. If quota was not reached, say “not reached” rather than extrapolating a completed cost.

Report successful attempts selected for training, unused successes, and retained failures. Provide links from metric rows to the original attempt and video. Explain known selection bias: training on successful attempts excludes failed recovery states from that training set.

## 5. Stage definitions

Reuse the task's existing, versioned stage labeling rules. Document their inputs, smoothing/hysteresis, ordering, and whether regressions are allowed. Do not derive stages from uncertainty scores. Assign takeover/return events to their event observation's stage, and each executed action to its pre-action stage.

Unknown stages remain an explicit category with coverage counts. “Furthest stage” is meaningful only if the task defines a progress ordering; it is not necessarily the final or failure stage. Planner failure messages are diagnoses, not independently verified physical causes.

For the current OpenDrawer adapter, the authoritative rules are [`src/vig_dagger/stages.py`](../src/vig_dagger/stages.py). Its 15 labels are:

| IDs | Stages |
|---|---|
| 0–4 | Reach handle; Handle approach; Grasp handle; Pull; Release |
| 5–8 | Reach charger; Charger approach; Grasp charger; Lift |
| 9–11 | Carry; Lower, release; Retreat |
| 12–14 | Push approach; Push; Hold |

The four task latches are drawer opened, charger picked, charger placed, and drawer closed. The implementation's geometry and smoothing rules remain authoritative. These OpenDrawer-specific labels are an adapter example, not a requirement for other tasks.

## 6. Figure requirements

- Include all selected variants that have valid data; show missing variants/cohorts explicitly in the accompanying table. Never require a particular driver family, score name or threshold set.
- Use compact horizontal comparison plots, one metric per figure. Adjust height to row count; two or three variants should produce a compact figure, not a mostly empty twelve-row layout. Paginate large collections consistently.
- Put explicit x/y labels **on every plot**, including units, normalization and cohort. A comparison y-axis identifies the variant; the x-axis names the measured quantity. Write out the control rate used to convert steps to seconds.
- Use the same order and colors for a variant across figures. Identify the actual cohort and valid `n` for every row. Thresholds may differ by variant; never use a global p50 label unless all variants actually use p50.
- Box: 25th–75th percentiles; line: median; diamond: mean; whiskers: most extreme observations within 1.5 IQR; dots: values beyond the whiskers. Include this legend. Mark singleton cohorts and absent distributions clearly.
- For cumulative distributions, x = duration or score; y = fraction of observations less than or equal to x. State whether observations are attempts, interventions or steps. Separate outcomes using readable colors and valid counts.
- Stage plots use readable stage names. Paginate per-attempt duration plots (default: at most 20 attempts/page), keeping IDs and outcome labels visible. Annotate failure heatmap counts.
- Export PNG and vector PDF for figures intended for sharing. Include underlying data tables so results can be reproduced without reading values off an image.

## 7. Every-attempt videos

Create one full MP4 for each completed attempt with usable image recordings, including successes and failures. Keep outcomes in separate folders and indexes. When both cohorts are pending, alternate them and rotate across variants so neither waits for another entire cohort to finish.

Each video must show:

1. Every recorded RGB camera, synchronized to the pre-action observation. Label each camera. Identify missing camera frames explicitly; never replace them with unlabeled stale images.
2. A bottom chart of the **actual decision score**, numeric takeover/return thresholds, takeover and return markers, expert-control shading, and a moving time cursor. Preserve unmeasured gaps.
3. Variant and attempt IDs, current controller, task stage, elapsed simulated time, cumulative expert portion, and final success/failure with the recorded reason. Do not imply that a detector message proves a physical cause.
4. Stage-transition markers where readable, plus short handoff clips (default: one second before and after each event, clipped to available frames).

Use recorded simulation timestamps or the declared control rate for playback. Do not hard-code 60 Hz for an unknown task. If timing metadata is absent, show steps and explicitly state any playback assumption; do not invent simulated durations. Different camera rates must be aligned using timestamps or a documented mapping.

Video status must distinguish ready, pending/open writer, unavailable/no RGB, corrupt/unreadable, and rendering error. Never reconstruct discarded images or count missing images as rendering failure. Rendering must not require rerunning the simulator or changing collection/training data.

Resume full videos and event clips independently. Write each output atomically, verify encoding success, and mark completion only when all required outputs are valid. Serialize writers to the same output. Preserve a source/configuration identity so stale videos are not silently reused after changing inputs or rendering semantics.

## 8. Report and output structure

Use one stable entry point with links to:

- Variant metadata, live completion counts, source snapshot times and missing-data coverage.
- All-variant comparisons and each variant's detailed figures.
- All attempts, successful attempts, and failed attempts with reasons and videos.
- Machine-readable attempt, event/intervention, stage, selection and aggregate tables.
- Data-quality findings, exclusions, and limitations.

Recommended layout (IDs are illustrative placeholders):

```text
results/dagger_audit/
  README.md                  # entry point and current coverage
  REPORT.md                  # metrics, interpretation, definitions
  COMPARISONS.md              # all selected variants
  VIDEOS.md
  SUCCESSFUL_ATTEMPTS.md
  FAILED_ATTEMPTS.md
  <group>/<variant>/         # figures, tables, metadata and snapshot
  videos/<group>/<variant>/success/<attempt>.mp4
  videos/<group>/<variant>/failed/<attempt>.mp4
```

Use collision-free, filesystem-safe IDs. Never identify attempts globally by scene seed alone. Include run/variant provenance in exported tables. Report timestamps for metric snapshots separately from live video inventory timestamps. A partial report must be clearly labeled provisional.

## 9. Completion, validation and edge cases

Analysis completion is independent of training/evaluation completion. For the selected scope, analysis is complete when collection has a recorded terminal state, every completed attempt has been accounted for, final tables/figures reflect that snapshot, and every usable recording has a validated full video and required clips. Missing historical images remain a disclosed limitation; they must not cause an endless rendering loop. A quota that is still being collected remains pending. An explicitly finalized collection that failed to reach quota can be analyzed completely and must be labeled quota not reached.

Validate before reporting completion:

- The selected set contains one or more unique variants; missing requested variants remain visible and block a claim of complete coverage.
- Attempt identity, action/controller lengths, observation alignment, finite-score coverage, event indices and stage labels reconcile. Quarantine inconsistent records with reasons rather than silently truncating arrays.
- Successes + failures equals completed attempts with known outcomes. Interrupted/unknown attempts are counted separately. Controller labels and stage counts reconcile to executed actions, with unknown counts explicit.
- Aggregate counts match exported rows. Training membership matches the manifest or its declared reconstruction. Per-attempt and pooled denominators are independently checked.
- One variant, two variants, three variants, an empty cohort, a zero-step failure, no interventions, zero policy exposure, absent stages, missing RGB, partial collection, terminal handoffs, and resumed rendering all have defined behavior.
- Figures have readable labels, visible valid sample counts and no invented data for empty groups. Reports and video indexes contain only valid links or explicit pending/unavailable states.

## 10. Repository adapter and execution

The current entry point is `python -m vig_dagger.audit --work-dir <run> [--watch]`. For detached execution use `.venv/bin/python scripts/run_audit_detached.py --work-dir <run>`; log and PID files are in the analysis directory.

The specification above is the analysis contract, not a claim that every advanced diagnostic is already implemented. The repository currently uses OpenDrawer stage/state conventions and its 60 Hz control rate. Adapting it to another task requires supplying the corresponding stage, timing, camera and identity adapters. Existing logs identify attempts by scene seed within a variant; data with repeated seeds need an explicit attempt-ID adapter.

For variant scope, the adapter accepts `analysis.variants` as a list of `<group>/<variant>` IDs when specified. Otherwise it uses configured `rounds × dagger.scorers`, or discovers `dagger/*/*/settings.json` when a run configuration is absent. Two or three variants use the same workflow. Render/collection completion must be checked against that selected set, never a fixed count such as twelve.
