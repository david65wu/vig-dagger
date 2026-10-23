"""Weights & Biases runs for the analysis steps (training runs are logged by the robopolicy trainer itself).

Every run of an experiment uses the config's ``wandb.entity`` / ``wandb.project`` and the experiment ``name`` as
its group. With ``wandb.enabled: false`` nothing is sent; the same numbers are always written to
``<work_dir>/results`` as JSON and CSV.
"""

from __future__ import annotations

from typing import Any


def start(cfg: dict, name: str, job_type: str, config: dict | None = None, tags: list[str] | None = None):
    """A W&B run, or None when W&B is disabled."""
    w = cfg["wandb"]
    if not w.get("enabled", True):
        return None
    import wandb

    return wandb.init(entity=w["entity"], project=w["project"], group=cfg["name"], name=f"{cfg['name']}/{name}",
                      job_type=job_type, config=config or {}, tags=tags or [], reinit="finish_previous",
                      settings=wandb.Settings(silent=True))


def flatten(values: dict[str, Any], prefix: str = "", sep: str = "/") -> dict[str, Any]:
    """Nested dicts to {"a/b/c": value} with only numbers, strings and booleans (None dropped)."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name, sep))
        elif isinstance(value, (int, float, str, bool)) and value is not None:
            out[name] = value
    return out


def table(rows: list[dict[str, Any]]):
    """A wandb.Table from a list of flat dicts (columns from the first row)."""
    import wandb

    columns = list(rows[0]) if rows else []
    return wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in rows])
