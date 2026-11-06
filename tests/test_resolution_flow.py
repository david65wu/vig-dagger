import json

import numpy as np
import pytest
import torch

from vig_dagger.resolution_flow import score_query, query_seed, episode_dir, write_json, report


class FakePolicy:
    device = torch.device('cpu')

    def flow_score(self, state, images, num_action_samples, num_loss_samples, factor):
        sample = torch.randn(1, 16, 7) + factor
        return float(torch.rand(()) + factor), sample

    def denormalize_actions(self, x):
        return x * 2


def test_scoring_is_order_independent_and_does_not_change_rng_or_actor():
    policies = {f: FakePolicy() for f in [1, 2, 4]}
    before = torch.get_rng_state().clone()
    args = (None, None, 2, query_seed(60000, 8), 512)
    values, chunks, actor = score_query(policies, [1, 2, 4], *args)
    reversed_values, reversed_chunks, reversed_actor = score_query(policies, [4, 2, 1], *args)
    _, _, actor_only = score_query(policies, [2], *args)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(actor, actor_only) and torch.equal(actor, reversed_actor)
    np.testing.assert_array_equal(values, reversed_values[::-1])
    np.testing.assert_array_equal(chunks, reversed_chunks[::-1])
    np.testing.assert_allclose(chunks[1] - 2, chunks[0] - 1, atol=5e-7)
    np.testing.assert_array_equal(actor.numpy(), chunks[1] * 2)


def test_nonfinite_score_rejected():
    policy = FakePolicy()
    policy.flow_score = lambda *a, **k: (float('nan'), torch.zeros(1, 16, 7))
    with pytest.raises(ValueError, match='Nonfinite'):
        score_query({1: policy}, [1], None, None, 1, 0, 512)


def test_report_preserves_raw_values_failed_attempts_and_distinguishes_weighting(tmp_path):
    cfg = dict(work_dir=str(tmp_path), protocol_id='test', factors=[1, 2],
               groups=['moe', 'per_resolution'], acting_factors=[1], scene_seeds=[10, 11],
               loss_draws=512, actions_per_step=8)
    for group in cfg['groups']:
        for seed, values in [(10, [[1., 2.]]), (11, [[5., 10.]] * 3)]:
            dest = episode_dir(cfg, group, 1, seed)
            dest.mkdir(parents=True)
            np.savez(dest / 'trace.npz', flow=values, query_seconds=np.arange(len(values)),
                     state_stages=np.zeros((4, 4), dtype=bool))
            write_json(dest / 'episode.json', dict(protocol_id='test', queries=len(values),
                seed=seed, group=group, acting_factor=1, success=seed == 10, dt=1/60))
    assert report(cfg) == (4, 4)
    summary = json.loads((tmp_path / 'results/summary.json').read_text())
    for group in cfg['groups']:
        s = summary['groups'][group]
        assert s['successes'] == 1 and s['rollouts'] == 2
        assert s['levels']['1']['mean'] == 4
        assert s['levels']['1']['rollout_mean_average'] == 3
        for name in ['boxplot_queries', 'boxplot_rollout_means', 'boxplot_by_acting_resolution']:
            assert (tmp_path / 'results' / group / f'{name}.png').is_file()
    assert (tmp_path / 'results/report.done.json').is_file()
