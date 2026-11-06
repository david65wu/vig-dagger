import json

import torch

from vig_dagger import plan, tasks
from vig_dagger.config import Paths


def _jobs(cfg):
    return plan.build(cfg, Paths.of(cfg).config())


def test_numbering_and_needs(cfg):
    jobs = _jobs(cfg)
    names = tasks.numbered(cfg, jobs)
    assert len(names) == 40
    assert names[:3] == ["moe/sd0", "moe/sd1", "moe/sd2"] and names[-1] == "summary"
    assert names[3] == "dagger/sd0_vig_p90_plain"
    by_name = {j.name: j for j in jobs}
    assert tasks.needs(by_name, "dagger/sd1_flow_p95_random_move") == {"moe/sd1"}
    assert tasks.needs(by_name, "summary") == {n for n in names if n.startswith("dagger/")}
    assert tasks.needs(by_name, "moe/sd1") == set()  # demos and their conversion are preparation, not a task


def test_inputs_pack_unpack(cfg, tmp_path):
    jobs = _jobs(cfg)
    paths = Paths.of(cfg)
    problems = tasks.check_inputs(cfg, jobs, ["dagger/sd0_vig_p90_plain"])
    assert len(problems) == 1 and "task 1 (moe/sd0)" in problems[0]
    # the MoE task's outputs on "machine A"
    run = paths.moe_dir(0)
    run.mkdir(parents=True)
    torch.save({"model": {"w": torch.ones(2)}, "optimizer": {"state": torch.ones(1000)}}, run / "latest.pt")
    (run / "thresholds.json").write_text(json.dumps({"vig": {"p90": 0.1}, "flow": {"p90": 0.2}}))
    (run / "metrics.jsonl").write_text("{}\n")
    assert tasks.check_inputs(cfg, jobs, ["dagger/sd0_vig_p90_plain"]) == []
    archive = tasks.pack(cfg, ["moe/sd0"], "dagger/sd0_vig_p90_plain", tmp_path / "a.tar")
    # on "machine B"
    other = dict(cfg, work_dir=str(tmp_path / "other"))
    files = tasks.unpack(other, archive)
    assert sorted(files) == ["policies/moe/sd0/latest.pt", "policies/moe/sd0/thresholds.json"]
    payload = torch.load(tmp_path / "other" / "policies/moe/sd0/latest.pt", weights_only=True)
    assert payload["optimizer"] == {} and torch.equal(payload["model"]["w"], torch.ones(2))  # weights only
    selected = tasks.select_jobs(other, _jobs(other), ["dagger/sd0_vig_p90_plain"])
    assert {j.name for j in selected} >= {"thresholds_sd0", "dagger_prepare_sd0_vig_p90_plain", "analyze_sd0_vig_p90_plain"}
    assert not any(j.name.startswith(("download_demos", "convert", "train/")) for j in selected)
    assert next(j for j in selected if j.name == "thresholds_sd0").done()
