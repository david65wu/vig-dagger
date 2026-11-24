"""The pipeline's short steps, each a command the job queue runs: ``python -m vig_dagger.steps <step> --config ...``.

  download-demos              the raw planner demos from hub.dataset_repo
  convert                     raw planner demos -> datasets/planner and datasets/base (the first demos.num_demos,
                              what the MoEs train on)
  download-moe --seed S       the MoE of seed S from its repo in moe.download (the checkpoint only; its thresholds
                              are recomputed here like every seed's)
  thresholds --seed S | --moe round<r>   merge the MoE's demo-frame scores into thresholds.json (percentiles) and
                              demo_scores.npz (in iteration mode the round's reference for the adaptive threshold)
  dagger-prepare --collection C   resolve the collection's threshold and write its settings.json
  round-dataset --round R     iteration mode: datasets/round<R>_train = round R-1's training set + the DAgger data of
                              round R-1's collection (random-move steps cut)
  analyze --collection C      the collection's metrics (results/collections/, REPORT.md, W&B if enabled)
  dagger-dataset --collection C   the collection's DAgger data (its first dagger.num_demos successful episodes) ->
                              datasets/dagger_C, and datasets/base_plus_C = the base demos + those episodes
  summarize                   every collection side by side and every evaluation (results/summary.*,
                              eval_summary.json, the figures, REPORT.md)
  publish --task T / fetch --task T   several machines: copy a task's outputs to the board / from it (board.py)
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vig_dagger import adaptive
from vig_dagger import config as config_module
from vig_dagger import dataset
from vig_dagger.config import Collection, Paths, collections, moe_key, target_of


def raw_parts(raw: Path) -> list[Path]:
    return sorted(raw.glob("part_*.h5"))


def _convert_one(args: tuple) -> int:
    path, indices, out, name, cut, min_length = args
    out_path = Path(out)
    if dataset.ready(out_path):
        return dataset.num_demos(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    n = dataset.convert([(Path(path), indices)], tmp, names=[name] if name else None, cut_random_moves=cut,
                        min_length=min_length)
    dataset.finalize(tmp, out_path)
    return n


def convert_sources(sources: list[tuple[Path, list[int] | None]], target: Path, workers: int,
                    names: list[str] | None = None, cut_random_moves: bool = False, min_length: int = 16) -> int:
    """Convert trajectory files in parallel (one piece per source) and combine them. ``names``: per source, what its
    demos are called (default: the file's stem); ``cut_random_moves``: cut DAgger episodes at their random moves
    (``dataset.convert``)."""
    pieces = target.with_name(target.name + ".pieces")
    pieces.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), i, str(pieces / f"piece_{k:04d}"), names[k] if names else None, bool(cut_random_moves), int(min_length))
            for k, (p, i) in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_convert_one, jobs))
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    n = dataset.combine([(Path(job[2]), None) for job in jobs], tmp)
    dataset.finalize(tmp, target)
    shutil.rmtree(pieces)
    return n


def demo_names(cfg: dict, parts: list[Path]) -> list[str] | None:
    """What the demos of each raw part are called in the training dataset: ``demos.demo_names`` with the part number
    (``{k}``), or the file name (None). The names decide which demos are held out for validation and the data order."""
    pattern = cfg["demos"]["demo_names"]
    return [pattern.format(k=int(p.stem.rsplit("_", 1)[-1])) for p in parts] if pattern else None


def names_current(cfg: dict, target: Path) -> bool:
    """The dataset's demos carry the names ``demos.demo_names`` gives them (always true without a pattern)."""
    pattern = cfg["demos"]["demo_names"]
    if not pattern:
        return True
    try:
        manifest = json.loads((target / "plug_charger" / "demo_manifest.json").read_text())
        return manifest["shards"][0]["demos"][0]["demo_key"].startswith(pattern.format(k=0) + "__traj_")
    except (OSError, KeyError, IndexError, ValueError):
        return False


def check_demo_names(cfg: dict, target: Path) -> None:
    """Refuse a dataset converted with other demo names (e.g. by an earlier version): its validation demos differ."""
    if not names_current(cfg, target):
        manifest = json.loads((target / "plug_charger" / "demo_manifest.json").read_text())
        first = manifest["shards"][0]["demos"][0]["demo_key"]
        want = cfg["demos"]["demo_names"].format(k=0) + "__traj_"
        raise SystemExit(f"{target} names its demos like {first!r}, not {want}<n>: it was converted with other demo names, "
                         f"so its validation demos differ from the seed-0 MoE's. Delete {target} and "
                         f"{target.with_name('base')} (and any MoE training started on them, policies/moe/sd<s>) and run again.")


def convert(cfg: dict) -> None:
    paths = Paths.of(cfg)
    target = paths.dataset("planner")
    if not dataset.ready(target):
        parts = raw_parts(paths.raw_demos())
        if not parts:
            raise SystemExit(f"no raw demos in {paths.raw_demos()}")
        n = convert_sources([(p, None) for p in parts], target, int(cfg["demos"]["convert_workers"]), demo_names(cfg, parts))
        print(f"[convert] planner: {n} demos -> {target}")
    check_demo_names(cfg, target)
    base = paths.dataset("base")
    if not dataset.ready(base):
        need = int(cfg["demos"]["num_demos"])
        if dataset.num_demos(target) < need:
            raise SystemExit(f"{target} has {dataset.num_demos(target)} demos; demos.num_demos is {need}")
        tmp = base.with_name("base.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        dataset.combine([(target, need)], tmp)
        dataset.finalize(tmp, base)
        print(f"[convert] base: the first {need} demos of {target}")


def thresholds(cfg: dict, seed: int | str) -> None:
    from vig_dagger.workers.score_demos import part_path

    paths, t = Paths.of(cfg), cfg["thresholds"]
    scorers = [x for x in ("vig", "flow") if x in t["scorers"]]
    parts = [np.load(part_path(cfg, seed, c)) for c in range(int(t["chunks"]))]
    parts = [p for p in parts if len(p["vig"])]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in ("vig", "vig_gap", *scorers[1:], "demo", "frame")}
    np.savez(paths.demo_scores(seed), **merged, factors=parts[0]["factors"])
    table = {s: {f"p{q}": float(np.percentile(merged[s], q)) for q in t["percentiles"]} for s in scorers}
    demos = int(len(np.unique(merged["demo"])))
    table.update(frames=int(len(merged["vig"])), demos=demos, frame_stride=int(t["frame_stride"]),
                 **({"seed": int(seed)} if isinstance(seed, int) else {"moe": seed}))
    if not isinstance(seed, int):  # iteration mode: the adaptive threshold's target
        ref = np.sort(merged["vig"].astype(float))
        table["vig_mean"] = float(ref.mean())
        table["vig_mean_percentile"] = adaptive.percentile_of(ref, float(ref.mean()))
    paths.thresholds(seed).write_text(json.dumps(table, indent=1))
    shutil.rmtree(part_path(cfg, seed, 0).parent)
    name = f"MoE seed {seed}" if isinstance(seed, int) else f"{seed} MoE"
    for scorer in scorers:
        print(f"[thresholds] {name}, {scorer} on {table['frames']} frames of {demos} training demos: "
              + ", ".join(f"{q} {v:.6f}" for q, v in table[scorer].items()), flush=True)
    if "vig_mean" in table:
        print(f"[thresholds] {name}: mean VIG {table['vig_mean']:.6f} = p{table['vig_mean_percentile']:.2f} (the adaptive "
              "threshold's target)", flush=True)


def resolve_threshold(cfg: dict, seed: int, scorer: str, spec: str) -> float:
    """"p<q>": the q-th percentile of the scorer on the MoE's demo frames (its thresholds.json)."""
    path = Paths.of(cfg).thresholds(seed)
    table = json.loads(path.read_text())
    if spec not in table[scorer]:
        raise SystemExit(f"{spec} of {scorer} is not in {path} (has {sorted(table[scorer])}); add it to "
                         "thresholds.percentiles and recompute the thresholds")
    return float(table[scorer][spec])


def start_percentile(cfg: dict, collection: Collection) -> float:
    """Iteration mode: round 1 starts at ``iterate.start_percentile``, round r where round r-1's collection ended."""
    if collection.round == 1:
        return float(cfg["iterate"]["start_percentile"])
    previous = collections(cfg)[collection.round - 2]
    p = adaptive.final_percentile(Paths.of(cfg).dagger(previous))
    if p is None:
        raise SystemExit(f"{collection.id} starts where {previous.id} ended, but {previous.id} has no threshold state yet")
    return p


def adaptive_settings(cfg: dict, collection: Collection) -> dict:
    """The adaptive threshold's settings (``adaptive.py``): the round MoE's reference, the target q(M), the start."""
    path = Paths.of(cfg).demo_scores(collection.moe)
    if not path.is_file():
        raise SystemExit(f"{path} is missing: the {collection.moe} MoE's reference is computed by its thresholds job")
    ref = adaptive.load_reference(path)
    mean = float(ref.mean())
    return dict(rule="p <- clip(p + n / N * (q(M) - q(m)), 0, 100) after every attempt", reference=str(path),
                reference_frames=int(len(ref)), reference_mean=mean, target_pct=adaptive.percentile_of(ref, mean),
                start_percentile=start_percentile(cfg, collection),
                exclude_trigger=bool(cfg["iterate"]["exclude_trigger"]))


def dagger_settings(cfg: dict, collection: Collection) -> dict:
    d = cfg["dagger"]
    target = target_of(cfg, collection)
    if collection.threshold == "adaptive":
        auto = adaptive_settings(cfg, collection)
        threshold = back = adaptive.threshold_at(adaptive.load_reference(Path(auto["reference"])), auto["start_percentile"])
    else:
        auto = None
        threshold = resolve_threshold(cfg, collection.seed, collection.scorer, collection.threshold)
        back = resolve_threshold(cfg, collection.seed, collection.scorer, collection.return_spec)
    if back > threshold:
        raise SystemExit(f"{collection.id}: hand-back threshold {collection.return_spec} ({back:.6g}) is above the "
                         f"take-over threshold {collection.threshold} ({threshold:.6g})")
    return dict(
        collection=collection.id, moe_seed=collection.seed, scorer=collection.scorer, variant=collection.variant,
        policies=str(Paths.of(cfg).moe_checkpoint(collection.moe)),
        take_threshold=threshold, return_threshold=back, take_spec=collection.threshold, return_spec=collection.return_spec,
        take_count=int(d["take_count"]), return_count=int(d["return_count"]), expert=d["expert"],
        random_move=dict(d["random_move"]) if collection.variant == "random_move" else None,
        atomic_grasp=bool(d["atomic_grasp"]), target=target, max_attempts=int(d["max_attempts"] or 20 * target),
        seed_start=int(d["seed_start"]) + ((collection.round - 1) * int(cfg["iterate"]["round_seed_stride"])
                                           if collection.round is not None else 0),
        max_episode_steps=int(d["max_episode_steps"]),
        chunk_steps=int(d["policy_chunk_steps"]), strategy_seed=int(d["strategy_seed"]),
        scorer_settings=cfg["scorers"][collection.scorer], save_failed=bool(d["save_failed_trajectories"]),
        **({"round": collection.round, "adaptive": auto} if auto else {}))


def check_expert_strategies(cfg: dict) -> int:
    """Every planner demo's recorded strategy (grasp, standoff, lift, approach, planner and both speeds) must be what
    the expert's strategy sampler draws from that strategy's seed, i.e. the expert samples its strategies exactly as
    the base demos did (every expert mode draws one per takeover; only the 'demo' expert also moves at its sampled
    speeds). Returns the number of demos checked; exits on a mismatch."""
    from vig_dagger.expert.planner import StrategyParams

    checked, bad = 0, []
    for part in sorted(Paths.of(cfg).raw_demos().glob("part_*.json")):
        for e in json.loads(part.read_text()).get("episodes", []):
            rec = e.get("strategy")
            if not rec:
                continue
            got = StrategyParams.sample(int(rec["seed"])).to_dict()
            diff = sorted(k for k in set(rec) | set(got) if k not in rec or k not in got
                          or (abs(rec[k] - got[k]) > 1e-9 if isinstance(rec[k], float) else rec[k] != got[k]))
            checked += 1
            if diff:
                bad.append(f"{part.name} episode {e.get('episode_id')}: {diff}")
    if bad or not checked:
        raise SystemExit("the expert's strategy sampler does not draw strategies like the planner demos did: "
                         + ("no demo with a recorded strategy found in " + str(Paths.of(cfg).raw_demos()) if not checked
                            else f"{len(bad)} of {checked} differ, e.g. " + "; ".join(bad[:3])))
    return checked


def dagger_prepare(cfg: dict, collection: Collection) -> None:
    out = Paths.of(cfg).dagger(collection)
    expert = cfg["dagger"]["expert"]
    n = check_expert_strategies(cfg)
    print(f"[dagger] expert '{expert}': its strategy sampler reproduces all {n} planner demos' strategies "
          + ("and speeds from their seeds" if expert == "demo" else "from their seeds (it moves at its own fixed speeds)"),
          flush=True)
    out.mkdir(parents=True, exist_ok=True)
    settings = dagger_settings(cfg, collection)
    path = out / "settings.json"
    if path.exists():
        old = json.loads(path.read_text())
        changed = sorted(k for k in settings if k not in ("target", "max_attempts", "policies") and old.get(k) != settings[k])
        if changed:
            raise SystemExit(f"{out} was collected with different settings ({changed}); use a new work_dir, or delete "
                             "the collection to start it again")
    path.write_text(json.dumps(settings, indent=1))
    move = settings["random_move"]
    if settings.get("adaptive"):
        a = settings["adaptive"]
        print(f"[dagger] {collection.id}: adaptive threshold from p{a['start_percentile']:.2f} = {settings['take_threshold']:.6g} "
              f"of the {collection.moe} MoE's {a['reference_frames']} reference frames, target q(M) = p{a['target_pct']:.2f} "
              f"(M = {a['reference_mean']:.6f}), trigger scores {'left out' if a['exclude_trigger'] else 'kept'}", flush=True)
    print(f"[dagger] {collection.id}: take > {settings['take_threshold']:.6g} ({settings['take_spec']}) "
          f"x{settings['take_count']}, return < {settings['return_threshold']:.6g} ({settings['return_spec']}) "
          f"x{settings['return_count']}, expert {settings['expert']}, "
          + (f"random move {move['radius'] * 100:g} cm / {move['rot_deg']:g} deg" if move else "no random move")
          + f", target {settings['target']}")


def dagger_sources(out: Path, target: int) -> list[tuple[Path, list[int]]]:
    """The first ``target`` successful episodes to finish (the collection's DAgger data, as the analysis counts it), as
    (trajectory file, indices) in finishing order."""
    rows = []
    for path in (out / "episodes").glob("worker_*.jsonl"):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    used = sorted((r for r in rows if r["success"]), key=lambda r: r["finished_at"])[:target]
    if len(used) < target:
        raise SystemExit(f"{out} has {len(used)} successful episodes; dagger.num_demos is {target}")
    sources: list[tuple[Path, list[int]]] = []
    for r in used:
        path = out / r["file"]
        if sources and sources[-1][0] == path:
            sources[-1][1].append(int(r["traj"]))
        else:
            sources.append((path, [int(r["traj"])]))
    return sources


def dagger_data(cfg: dict, collection: Collection) -> Path:
    """datasets/dagger_<c>: the collection's DAgger data (its first target successful episodes: policy and expert
    steps, with the executed actions as labels; with ``retrain.cut_random_moves`` every episode is cut at its random
    moves, which leave training with the 15 steps before each)."""
    paths, n = Paths.of(cfg), target_of(cfg, collection)
    cut, min_length = bool(cfg["retrain"]["cut_random_moves"]), int(cfg["retrain"]["min_piece_length"])
    root = paths.dataset(f"dagger_{collection.id}")
    if dataset.ready(root) and (dataset.cut_stats(root) is not None) != cut:
        raise SystemExit(f"{root} was built {'without' if cut else 'with'} the random-move cut, but "
                         f"retrain.cut_random_moves is {cut}; delete it to rebuild")
    if not dataset.ready(root):
        count = convert_sources(dagger_sources(paths.dagger(collection), n), root, int(cfg["demos"]["convert_workers"]),
                                cut_random_moves=cut, min_length=min_length)
        print(f"[dagger-dataset] dagger_{collection.id}: {count} episodes", flush=True)
        st = dataset.cut_stats(root)
        if st:
            print(f"[dagger-dataset] random moves cut: {st['with_moves']} of {st['episodes']} episodes had them, "
                  f"{st['moves']} moves, {st['move_steps']} of their {st['steps']} steps -> {st['pieces']} demos "
                  f"({st['short_pieces']} pieces under {st['min_length']} steps dropped, {st['short_steps']} steps)", flush=True)
    return root


def dagger_dataset(cfg: dict, collection: Collection) -> None:
    """datasets/dagger_<c> (``dagger_data``) and datasets/base_plus_<c> (the base demos first, keeping their identities
    and so their train/validation assignment, then the DAgger episodes)."""
    paths, n = Paths.of(cfg), target_of(cfg, collection)
    cut = bool(cfg["retrain"]["cut_random_moves"])
    existing = paths.dataset(f"base_plus_{collection.id}")
    if dataset.ready(existing) and (dataset.cut_stats(existing) is not None) != cut:
        raise SystemExit(f"{existing} was built {'without' if cut else 'with'} the random-move cut, but "
                         f"retrain.cut_random_moves is {cut}; delete it to rebuild")
    root = dagger_data(cfg, collection)
    target = paths.dataset(f"base_plus_{collection.id}")
    if not dataset.ready(target):
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        count = dataset.combine([(paths.dataset("base"), None), (root, n)], tmp)
        dataset.finalize(tmp, target)
        print(f"[dagger-dataset] base_plus_{collection.id}: {count} demos", flush=True)


def round_dataset(cfg: dict, r: int) -> None:
    """Iteration mode: datasets/round<r>_train = round r-1's training set (round 1's: the base demos) + the DAgger data
    of round r-1's collection. The earlier inputs come first and keep their shard ids, so every demo keeps its
    train/validation assignment from round to round."""
    paths = Paths.of(cfg)
    if r < 2:
        raise SystemExit("round 1 trains on the base demos (convert)")
    target = paths.round_dataset(r)
    if dataset.ready(target):
        return
    previous = paths.round_dataset(r - 1)
    if not dataset.ready(previous):
        raise SystemExit(f"{previous} is not built yet")
    collection = collections(cfg)[r - 2]
    root = dagger_data(cfg, collection)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    count = dataset.combine([(previous, None), (root, None)], tmp)
    dataset.finalize(tmp, target)
    print(f"[round-dataset] {target.name}: {count} episodes ({dataset.num_demos(previous)} of {previous.name} + "
          f"{dataset.num_demos(root)} of {root.name})", flush=True)


def analyze(cfg: dict, collection: Collection) -> None:
    from vig_dagger import analysis, report

    paths = Paths.of(cfg)
    analysis.analyze(cfg, collection, paths.dagger(collection), paths.collection_results(collection))
    report.write(cfg)


def summarize(cfg: dict) -> None:
    from vig_dagger import report
    from vig_dagger.config import board_mode

    if board_mode(cfg):  # every machine's collection metrics, MoE logs and thresholds, from the board
        from vig_dagger import board

        board.pull_results(cfg)

    report.write(cfg)  # summary.summarize first (results/summary.*, figures/collections.png), then REPORT.md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--moe", help="thresholds: the MoE (a seed, or round<r> in iteration mode); default --seed")
    p.add_argument("--round", type=int)
    p.add_argument("--collection")
    p.add_argument("--task", help="publish / fetch: the task (see board.py)")
    args = p.parse_args()
    from vig_dagger.jobs import exit_on_sigterm

    exit_on_sigterm()
    cfg = config_module.load(args.config)
    from vig_dagger import hub

    step = args.step
    if step == "download-demos":
        hub.download_demos(cfg)
    elif step == "convert":
        convert(cfg)
    elif step == "download-moe":
        hub.download_moe(cfg, args.seed)
    elif step == "thresholds":
        thresholds(cfg, moe_key(args.moe) if args.moe is not None else args.seed)
    elif step == "round-dataset":
        round_dataset(cfg, args.round)
    elif step == "dagger-prepare":
        dagger_prepare(cfg, Collection.parse(args.collection))
    elif step == "analyze":
        analyze(cfg, Collection.parse(args.collection))
    elif step == "dagger-dataset":
        dagger_dataset(cfg, Collection.parse(args.collection))
    elif step == "summarize":
        summarize(cfg)
    elif step in ("publish", "fetch"):
        from vig_dagger import board

        (board.publish if step == "publish" else board.fetch)(cfg, args.task)
    else:
        raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main()
