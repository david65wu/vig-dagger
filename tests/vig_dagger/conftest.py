from pathlib import Path

import pytest
import yaml

from vig_dagger import config as config_module

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path):
    """configs/plugcharger.yaml with a temporary work directory."""
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
