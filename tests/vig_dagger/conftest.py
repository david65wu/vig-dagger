from pathlib import Path

import pytest
import yaml

from vig_dagger import config as config_module

REPO = Path(__file__).resolve().parents[2]

# the plugcharger-moe-seeds grid (3 MoE seeds x VIG/flow x p90/p95/p99 x plain/random move, one threshold both ways):
# most tests exercise the multi-seed machinery (numbering, packing, the task board) on it
GRID = ["moe.seeds=[0, 1, 2]", "dagger.scorers=[vig, flow]", "dagger.thresholds=[p90, p95, p99]",
        "dagger.return_thresholds=[same]", "dagger.variants=[plain, random_move]"]


@pytest.fixture
def cfg(tmp_path):
    """configs/plugcharger.yaml with the 36-collection grid above and a temporary work directory."""
    return config_module.load(REPO / "configs" / "plugcharger.yaml", [*GRID, f"work_dir={tmp_path / 'work'}"])


@pytest.fixture
def sweep_cfg(tmp_path):
    """configs/plugcharger.yaml as it is (this branch's hand-back sweep) with a temporary work directory."""
    return config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path / 'work'}"])


@pytest.fixture
def write_cfg(tmp_path):
    def make(overrides: dict) -> dict:
        path = tmp_path / "exp.yaml"
        base = yaml.safe_load((REPO / "configs" / "plugcharger.yaml").read_text())
        base.update(overrides)
        base["work_dir"] = str(tmp_path / "work")
        path.write_text(yaml.safe_dump(base))
        return config_module.load(path)
    return make
