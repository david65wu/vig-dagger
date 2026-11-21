"""Reference collections: a collection made earlier with the same MoE, expert and DAgger settings (and its retrained
policy), shown next to this experiment's collections and policies in the summary, the report and the figures.

``reference.dirs`` names directories under the repository's ``reference/`` (or absolute paths). Each holds:

  settings.json    the collection's settings.json
  collection.json  its metrics: ``results/collections/<id>.json`` as this branch's analysis writes it
  code.json        sha256 of the files that decide how episodes are collected (expert, DAgger worker, gate, scorer,
                   environment, policy wrapper, recorder) in the commit that collected it
  retrain.json     how the dense policy on its DAgger data was trained (demos, samples, updates, the random-move cut)
  eval/*.jsonl     that policy's evaluation episodes (workers/evaluate.py on this experiment's scenes)
  README.md        where it comes from

A reference is used only if it matches: every setting equal to this experiment's collections' except the hand-back
threshold and what does not change the episodes (the collection's id, target, attempt limit, checkpoint path, whether
failed trajectories were saved), the take-over threshold within 0.5 % (the same MoE scored on other hardware), and the
collection code unchanged. Otherwise it is reported as excluded, with the reasons.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "reference"
PACKAGE = Path(__file__).resolve().parent
CODE_FILES = ("expert/planner.py", "expert/interruptible_planner.py", "expert/random_move.py", "workers/dagger.py",
              "gate.py", "scoring.py", "envs.py", "policy.py", "recording.py")
IGNORED = ("collection", "return_threshold", "return_spec", "target", "max_attempts", "policies", "save_failed")
THRESHOLD_TOLERANCE = 0.005


def code_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((PACKAGE / f).read_bytes()).hexdigest() for f in CODE_FILES}


def _dirs(cfg: dict) -> list[Path]:
    out = []
    for d in (cfg.get("reference") or {}).get("dirs") or []:
        p = Path(d)
        out.append(p if p.is_absolute() else ROOT / p)
    return out


def _expected_settings(cfg: dict, ref_id: str) -> dict | None:
    """The settings of this experiment's collection with the reference's MoE seed, scorer, take-over threshold and
    variant: its settings.json if it exists, else what dagger-prepare would write (needs the MoE's thresholds). None
    when there is no such collection or its thresholds are not known yet."""
    from vig_dagger.config import Collection, Paths, collections

    paths, ref = Paths.of(cfg), Collection.parse(ref_id)
    same = [c for c in collections(cfg) if (c.seed, c.scorer, c.threshold, c.variant) == (ref.seed, ref.scorer, ref.threshold, ref.variant)]
    for c in same:
        path = paths.dagger(c) / "settings.json"
        if path.is_file():
            return json.loads(path.read_text())
    if not same:
        return None
    try:
        from vig_dagger.steps import dagger_settings

        return dagger_settings(cfg, same[0])
    except (OSError, KeyError, SystemExit):
        return None


def check(cfg: dict, ref: dict) -> list[str]:
    """Why a reference does not match this experiment (empty: it matches)."""
    from vig_dagger.config import Collection, collections

    problems = []
    want, have = _expected_settings(cfg, ref["id"]), ref["settings"]
    r = Collection.parse(ref["id"])
    if not any((c.seed, c.scorer, c.threshold, c.variant) == (r.seed, r.scorer, r.threshold, r.variant) for c in collections(cfg)):
        problems.append(f"this experiment has no collection with MoE seed {r.seed}, {r.scorer}, take-over threshold "
                        f"{r.threshold} and variant {r.variant}")
    elif want is None:
        problems.append("this experiment's settings are not known yet (no thresholds)")
    else:
        for key in sorted(set(want) | set(have)):
            if key in IGNORED:
                continue
            a, b = want.get(key), have.get(key)
            if key == "take_threshold" and a is not None and b is not None:
                if abs(a - b) > THRESHOLD_TOLERANCE * abs(a):
                    problems.append(f"take-over threshold {b:.6g} vs this experiment's {a:.6g}")
            elif a != b:
                problems.append(f"{key}: {b!r} vs this experiment's {a!r}")
    code = json.loads((ref["dir"] / "code.json").read_text())
    now = code_hashes()
    changed = [f for f in CODE_FILES if code["sha256"].get(f) != now[f]]
    if changed:
        problems.append("collection code changed since the reference: " + ", ".join(changed))
    return problems


def load(cfg: dict) -> list[dict]:
    """Every configured reference: dir, id, settings, result (collection.json), code, retrain info, evaluation rows,
    readme, and ``problems`` (empty when it matches this experiment)."""
    out = []
    for d in _dirs(cfg):
        if not (d / "collection.json").is_file():
            out.append(dict(dir=d, id=d.name, problems=[f"{d} has no collection.json"]))
            continue
        result = json.loads((d / "collection.json").read_text())
        rows = [json.loads(line) for p in sorted((d / "eval").glob("*.jsonl")) for line in p.read_text().splitlines()
                if line.strip()]
        ref = dict(dir=d, id=result["collection"], settings=json.loads((d / "settings.json").read_text()), result=result,
                   retrain=json.loads((d / "retrain.json").read_text()) if (d / "retrain.json").is_file() else None,
                   eval_rows=sorted({r["seed"]: r for r in rows}.values(), key=lambda r: r["seed"]),
                   readme=(d / "README.md").read_text() if (d / "README.md").is_file() else "")
        ref["problems"] = check(cfg, ref)
        out.append(ref)
    return out


def usable(cfg: dict) -> list[dict]:
    return [r for r in load(cfg) if not r["problems"]]
