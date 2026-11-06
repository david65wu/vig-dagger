"""Whole-rollout flow percentiles, calibrated separately for each dense policy."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

GROUPS = ('familiar_a_to_b', 'midpoint_id', 'midpoint_3cm', 'midpoint_8cm',
          'unseen_a_to_b', 'outside_sphere_a_to_b')
COLORS = ('#7ba9ce', '#4b83af', '#edb05c', '#d96b65', '#a697c8', '#a58b76')


def percentile_rank(reference, values):
    """Empirical CDF: percentage of calibration scores <= each rollout score."""
    reference = np.asarray(reference)
    assert reference.ndim == 1 and len(reference) and np.isfinite(reference).all()
    assert np.isfinite(values).all()
    return 100. * np.searchsorted(np.sort(reference), values, side='right') / len(reference)


def run(out, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out.mkdir(parents=True, exist_ok=True)
    tasks = [
        ('sphere', 'Random A/B sphere', runs/'toy_sphere_rollouts_300ep_v1'),
        ('fixed_goal_sphere', 'Fixed-goal hemisphere', runs/'toy_fixed_goal_sphere_v1/video_eval'),
    ]
    observations = []; summaries = []; provenance = []
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    for task, title, root in tasks:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        for ax, count in zip(axes, (5, 20)):
            name = f'demos{count}_dense_full'
            calibration_path = root/'flow_calibration'/f'{name}.json'
            calibration = json.loads(calibration_path.read_text())
            with np.load(calibration_path.with_suffix('.npz')) as data:
                reference = data['flow'].copy()
                assert np.allclose(reference, data['chunk_losses'].mean(axis=1))
            assert len(reference) == count * 20 == calibration['frames']
            assert calibration['checkpoint'].endswith('/epoch-300.pt')
            assert np.isclose(np.percentile(reference, 95), calibration['p95'])
            rows = json.loads((root/name/'rollouts.json').read_text())
            assert len(rows) == 30 and len({r['case_id'] for r in rows}) == 30
            grouped = {g: [] for g in GROUPS}
            for row in rows:
                stem = f"{row['case_id']:02d}_{row['kind']}"
                folder = root/name/'calibrated_chunk_videos'
                meta = json.loads((folder/f'{stem}.json').read_text())
                assert meta['checkpoint'] == calibration['checkpoint']
                assert meta['samples'] == calibration['samples'] == 32
                assert meta['noise_time_draws_per_chunk'] == calibration['draws'] == 64
                with np.load(folder/f'{stem}_chunks.npz') as data:
                    flow, steps = data['flow'].copy(), data['steps'].copy()
                    assert np.allclose(flow, data['chunk_losses'].mean(axis=1))
                assert np.array_equal(steps, np.arange(0, max(1, row['steps']), 8))
                assert len(flow) == len(steps)
                ranks = percentile_rank(reference, flow)
                grouped[row['kind']].extend(ranks.tolist())
                observations.extend(dict(task=task, policy=name, condition=row['kind'],
                    case_id=row['case_id'], control_step=int(step), time_s=float(step/60),
                    flow_loss=float(loss), training_percentile=float(rank))
                    for step, loss, rank in zip(steps, flow, ranks))
            values = [np.array(grouped[g]) for g in GROUPS]
            for group, values_in_group in zip(GROUPS, values):
                assert sum(r['kind'] == group for r in rows) == 5
                q = np.percentile(values_in_group, [5, 25, 50, 75, 95])
                summaries.append(dict(task=task, policy=name, condition=group,
                    episodes=5, scored_observations=len(values_in_group),
                    p05=float(q[0]), q25=float(q[1]), median=float(q[2]), q75=float(q[3]), p95=float(q[4])))
            boxes = ax.boxplot(values, positions=np.arange(1, 7), widths=.56,
                patch_artist=True, whis=(5, 95), showfliers=True,
                medianprops=dict(color='#202833', linewidth=2),
                whiskerprops=dict(color='#687585', linewidth=1.2),
                capprops=dict(color='#687585', linewidth=1.2),
                flierprops=dict(marker='o', markersize=2, markerfacecolor='#77818d',
                                markeredgewidth=0, alpha=.24))
            for box, color in zip(boxes['boxes'], COLORS):
                box.set(facecolor=color, edgecolor='#455466', linewidth=1.1, alpha=.8)
            labels = ['Familiar\nA → B', 'Midpoint\nID', 'Offset\n3 cm', 'Offset\n8 cm',
                      'Unseen A\n(same B)' if task == 'fixed_goal_sphere' else 'Unseen\nA/B',
                      'Both endpoints\noutside sphere']
            labels = [label+f'\nn={len(v):,}' for label, v in zip(labels, values)]
            ax.set(xticks=np.arange(1, 7), xticklabels=labels, ylim=(-2, 103),
                   yticks=[0, 20, 40, 60, 80, 95, 100],
                   title=f'{count} demos · dense_full\nTraining calibration: {len(reference)} observations')
            ax.tick_params(axis='x', labelsize=10, length=0, pad=10)
            ax.tick_params(axis='y', labelleft=True)
            ax.axhline(95, color='#824a98', linestyle='--', linewidth=1.5, zorder=0)
            ax.set_axisbelow(True); ax.grid(axis='y', color='#e3e6eb', linewidth=.8)
            ax.spines[['top', 'right']].set_visible(False)
            provenance.append(dict(task=task, policy=name, checkpoint=calibration['checkpoint'],
                calibration=str(calibration_path), calibration_frames=len(reference), source=str(root/name)))
        axes[0].set_ylabel('Flow-loss percentile in own training distribution')
        fig.suptitle(title+' — whole-rollout flow percentiles', fontsize=19, fontweight='semibold', y=.97)
        fig.legend(handles=[Line2D([0], [0], color='#824a98', linestyle='--', lw=1.5,
                                  label='95th training percentile')],
                   loc='upper center', bbox_to_anchor=(.5, .905), frameon=False)
        fig.text(.5, .075, 'Boxes: 25th–75th percentiles · Center line: median · Whiskers: 5th–95th percentiles · Dots: values beyond whiskers',
                 ha='center', fontsize=10, color='#46505c')
        fig.text(.5, .045, 'Five episodes per condition; all scored observations pooled (every 8 control steps). Longer episodes contribute more samples.',
                 ha='center', fontsize=10, color='#46505c')
        fig.text(.5, .016, 'Conditions describe rollout starts. Epoch 300 throughout. Percentile = 100 × fraction of own training scores ≤ rollout score.',
                 ha='center', fontsize=10, color='#46505c')
        fig.subplots_adjust(left=.07, right=.985, top=.79, bottom=.245, wspace=.15)
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'{task}_flow_percentiles.{extension}', dpi=180)
        plt.close(fig)
    with (out/'whole_rollout_flow_percentiles.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0])); writer.writeheader(); writer.writerows(observations)
    (out/'boxplot_summary.json').write_text(json.dumps(summaries, indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(
        percentile='100 * count(training calibration score <= rollout score) / number of training calibration scores',
        tie_rule='right-inclusive empirical CDF; tied reference scores are included',
        saturation='Values at or above the maximum training calibration score map to percentile 100',
        pooling='Every scored whole-rollout observation included once; episodes are not equally weighted',
        interval='Every 8 control steps until termination, without duplicated video end-hold frames',
        box='25th to 75th percentile of the pooled calibrated scores; median center line; 5th/95th percentile whiskers',
        samples=len(observations), conditions=list(GROUPS), policies=provenance), indent=2))
    print(json.dumps(dict(output=str(out), figures=2, policies=4, boxes=24,
                          scored_observations=len(observations)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('/cpfs02/user/caojianuo/vig-dagger-runs'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    run(args.out or args.runs/'toy_flow_percentile_comparison_v1', args.runs)
