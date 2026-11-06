import json

import pytest
import torch

from vig_dagger import board, plan
from vig_dagger.config import Paths


@pytest.fixture
def two_machines(cfg, tmp_path):
    """Two work directories sharing one board directory."""
    board._BOARDS.clear()
    a = dict(cfg, sharing=dict(cfg["sharing"], board=str(tmp_path / "board")))
    b = dict(a, work_dir=str(tmp_path / "other"))
    yield a, b
    board._BOARDS.clear()


def test_claims_publish_fetch_release(two_machines):
    a, b = two_machines
    ba, bb = board.get(a), board.get(b)
    assert ba.machine != bb.machine
    ba.ensure(); bb.ensure()
    assert ba.claim("moe/sd0") and not bb.claim("moe/sd0")  # one machine only
    assert bb.claim("moe/sd1")
    bb.refresh(force=True)
    assert bb.owner("moe/sd0") == ba.machine and bb.mine() == {"moe/sd1"}
    # machine a finishes moe/sd0 and publishes it (the checkpoint weights-only)
    run = Paths.of(a).moe_dir(0)
    run.mkdir(parents=True)
    torch.save({"model": {"w": torch.ones(2)}, "optimizer": {"s": torch.ones(100)}}, run / "latest.pt")
    (run / "thresholds.json").write_text(json.dumps({"vig": {"p90": 0.1}}))
    board.publish(a, "moe/sd0")
    assert bb.claim("dagger/sd0_vig_p90_plain") and board.task_done(b, "moe/sd0")
    board.fetch(b, "moe/sd0")  # machine b gets it
    payload = torch.load(Paths.of(b).moe_checkpoint(0), weights_only=True)
    assert payload["optimizer"] == {} and json.loads(Paths.of(b).thresholds(0).read_text()) == {"vig": {"p90": 0.1}}
    # a task whose machine is gone: released, then claimable again
    assert ba.release("moe/sd1") and ba.claim("moe/sd1")
    with pytest.raises(SystemExit):
        ba.release("moe/sd0")  # done


def test_other_settings_refused(two_machines):
    a, b = two_machines
    board.get(a).ensure()
    b["dagger"] = dict(b["dagger"], num_demos=7)
    board._BOARDS.pop(str(Paths.of(b).root), None)
    with pytest.raises(SystemExit, match="other settings"):
        board.get(b).ensure()


def test_graph_routes_through_fetch(two_machines):
    a, _ = two_machines
    jobs = {j.name: j for j in plan.build(a, Paths.of(a).config())}
    assert jobs["dagger_prepare_sd1_vig_p90_plain"].deps == ["fetch/moe/sd1"]
    assert "fetch/dagger/sd0_vig_p90_plain" in jobs["summarize"].deps
    s = board.Scheduler(a, jobs)
    assert s.needs["dagger/sd2_flow_p99_random_move"] == {"moe/sd2"} and len(s.needs["summary"]) == 36
