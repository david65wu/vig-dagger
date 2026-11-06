"""Per-step statistics of a DAgger collection: takeovers, and per task stage and failure flag the numbers behind the
stage tables of a collection's page data (``page_data.py``).

Input: every attempt's stage, driver, score and flags per step, as ``analysis.load_episodes`` reads them from the
collection's ``attempts/``. Attempts are grouped into all, successful and failed ones; a step belongs to the stage
of the observation it was taken from. A takeover is a run of steps the policy did not drive: the random move (variant
``random_move``) and the expert's steps after it.
"""

from __future__ import annotations

import numpy as np

from vig_dagger import stages
from vig_dagger.recording import EXPERT, MOVE, POLICY

GROUPS = ("all", "success", "failed")


def _runs(mask) -> list[tuple[int, int]]:
    """(start, length) of every run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def _group(rows: list[dict], group: str) -> list[dict]:
    return rows if group == "all" else [a for a in rows if a["success"] == (group == "success")]


def load_steps(path) -> list[dict]:
    """Every attempt's per-step record (``results/collections/<collection>_steps.npz``), in seed order."""
    if not path.exists():
        return []
    z = np.load(path)
    o = z["offsets"]
    rows = [dict(seed=int(z["seed"][i]), success=bool(z["success"][i]), reason=str(z["reason"][i]),
                 furthest=int(z["furthest"][i]), **{k: z[k][o[i]:o[i + 1]] for k in ("stage", "controller", "score", "flags")})
            for i in range(len(z["seed"]))]
    return sorted(rows, key=lambda a: a["seed"])


def takeovers(rows: list[dict], settings: dict) -> list[dict]:
    """Every takeover: its attempt, start, length, stage at the start, where in that stage it started (0 = the
    attempt's first step in the stage, 1 = its last), whether it handed back, and its longest run of scores below the
    return threshold. Those are the scores the gate checks for a hand-back: of the observation after each expert step,
    the last of them the one the policy resumes from; unscored steps (a random move, an atomic grasp) neither count nor
    break a run."""
    out = []
    for a in rows:
        for start, length in _runs(a["controller"] != POLICY):
            k = int(a["stage"][start])
            idx = np.flatnonzero(a["stage"] == k)
            checked = a["score"][start + 1:start + length + 1]
            below = _runs(checked[np.isfinite(checked)] < settings["return_threshold"])
            out.append(dict(seed=a["seed"], success=a["success"], start=start, length=length, stage=k,
                            position=(start - idx[0]) / (idx[-1] - idx[0] + 1),
                            returned=start + length < len(a["controller"]),
                            longest_below=max([n for _, n in below], default=0)))
    return out


def _quantiles(values) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, p5=None, p25=None, median=None, p75=None, p95=None, mean=None)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=int(len(x)), p5=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p95=float(q[4]),
                mean=float(x.mean()))


def _share(num, den):
    return num / den if den else None


def stage_stats(rows: list[dict], settings: dict) -> list[dict]:
    """One row per group of attempts and stage: the numbers behind the stage figures."""
    take, ret = settings["take_threshold"], settings["return_threshold"]
    tos = takeovers(rows, settings)
    out = []
    for group in GROUPS:
        att = _group(rows, group)
        seeds = {a["seed"] for a in att}
        for k, name in enumerate(stages.STAGES):
            here = [a["stage"] == k for a in att]
            pol = [m & (a["controller"] == POLICY) for m, a in zip(here, att)]
            exp = [m & (a["controller"] == EXPERT) for m, a in zip(here, att)]
            move_steps = sum(int((m & (a["controller"] == MOVE)).sum()) for m, a in zip(here, att))
            steps = np.array([int(m.sum()) for m in here])
            p_steps, e_steps = sum(int(m.sum()) for m in pol), sum(int(m.sum()) for m in exp)
            above = sum(int((a["score"][m] > take).sum()) for a, m in zip(att, pol))
            below = sum(int((a["score"][m] < ret).sum()) for a, m in zip(att, exp))
            with_policy = [i for i, m in enumerate(pol) if m.any()]
            run = sum(bool(_runs(pol[i] & (att[i]["score"] > take))) for i in with_policy)
            started = [t for t in tos if t["seed"] in seeds and t["stage"] == k]
            taken = len({t["seed"] for t in started} & {att[i]["seed"] for i in with_policy})
            visited = steps[steps > 0]
            spent = np.percentile(visited, [25, 50, 75]) if len(visited) else [None] * 3
            longest = _quantiles([t["longest_below"] for t in started])
            row = dict(group=group, stage=name, attempts=len(att), reached=sum(a["furthest"] >= k for a in att),
                       failed_here=sum(not a["success"] and a["furthest"] == k for a in att), visited=int(len(visited)),
                       steps=int(steps.sum()), steps_p25=spent[0], steps_median=spent[1], steps_p75=spent[2],
                       policy_steps=p_steps, expert_steps=e_steps, move_steps=move_steps,
                       expert_share=_share(e_steps, int(steps.sum())), move_share=_share(move_steps, int(steps.sum())),
                       policy_above_take=_share(above, p_steps), expert_below_return=_share(below, e_steps),
                       attempts_with_policy_steps=len(with_policy), with_run_above_take=_share(run, len(with_policy)),
                       with_takeover=_share(taken, len(with_policy)), takeovers=len(started),
                       takeovers_per_100_policy_steps=_share(100 * len(started), p_steps),
                       returned=sum(t["returned"] for t in started), **{f"longest_below_{q}": v for q, v in longest.items()})
            for who, masks in (("policy", pol), ("expert", exp)):
                q = _quantiles(np.concatenate([a["score"][m] for a, m in zip(att, masks)]) if att else [])
                row.update({f"{who}_score_{key}": v for key, v in q.items()})
            row["fail_rate"] = _share(row["failed_here"], row["reached"])
            out.append(row)
    return out


def flag_stats(rows: list[dict], settings: dict) -> list[dict]:
    """One row per group of attempts and failure flag."""
    out = []
    for group in GROUPS:
        att = _group(rows, group)
        for i, name in enumerate(stages.FLAGS):
            on = [((a["flags"] >> i) & 1).astype(bool) for a in att]
            pol = [m & (a["controller"] == POLICY) for m, a in zip(on, att)]
            steps, e_steps, p_steps = (sum(int(m.sum()) for m in on), sum(int((m & (a["controller"] == EXPERT)).sum()) for m, a in zip(on, att)),
                                       sum(int(m.sum()) for m in pol))
            above = sum(int((a["score"][m] > settings["take_threshold"]).sum()) for a, m in zip(att, pol))
            out.append(dict(group=group, flag=name, attempts=len(att), with_flag=_share(sum(bool(m.any()) for m in on), len(att)),
                            steps=steps, expert_share=_share(e_steps, steps), policy_above_take=_share(above, p_steps)))
    return out
