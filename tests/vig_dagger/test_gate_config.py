import pytest

from vig_dagger.config import Collection, collections, downloaded
from vig_dagger.gate import TwoThresholdGate


def test_one_threshold_takes_over_above_and_hands_back_below():
    gate = TwoThresholdGate(0.5, 0.5, take_count=2, return_count=16)
    assert gate.update(0.6) is None
    assert gate.update(0.7) == "take"
    for _ in range(15):
        assert gate.update(0.4) is None
    assert gate.update(0.4) == "return"
    assert gate.update(0.9) is None and gate.update(0.9) == "take"


def test_counts_need_consecutive_scores():
    gate = TwoThresholdGate(0.5, 0.5, take_count=2, return_count=3)
    assert [gate.update(x) for x in (0.6, 0.4, 0.6, 0.6)] == [None, None, None, "take"]
    assert [gate.update(x) for x in (0.4, 0.4, 0.6, 0.4, 0.4, 0.4)] == [None, None, None, None, None, "return"]


def test_collections_grid(cfg):
    cols = collections(cfg)
    assert len(cols) == 36 and len({c.id for c in cols}) == 36
    assert cols[0].id == "sd0_vig_p90_plain" and cols[-1].id == "sd2_flow_p99_random_move"
    assert [c.variant for c in cols[:12]] == ["plain"] * 6 + ["random_move"] * 6
    for c in cols:
        assert Collection.parse(c.id) == c
    assert downloaded(cfg, 0) and not downloaded(cfg, 1) and not downloaded(cfg, 2)


def test_config_rejects_unknown_threshold(write_cfg):
    with pytest.raises(ValueError):
        write_cfg({"dagger": {"thresholds": ["p42"]}})
    with pytest.raises(ValueError):
        write_cfg({"dagger": {"variants": ["shake"]}})
