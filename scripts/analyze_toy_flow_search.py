"""Score action samples, select states without outcomes, and report repeated trials."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch


def score(args):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_flow_search import sample_and_score, chunk_losses
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.root/args.run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    policy = LibraryPolicy(args.run_dir/'latest.pt', 'cuda:0')
    rows, chunks, losses, standard_errors, calibration_512 = [], [], [], [], []
    started = time.time()
    with h5py.File(args.root/'observations.hdf5') as f:
        records = json.loads(f.attrs['records'])
        states = f['states'][:]; references = f['references'][:]
        for index, record in enumerate(records):
            s = torch.tensor(states[index], device=policy.device)[None]
            im = torch.tensor(f['images'][index], device=policy.device).permute(0, 3, 1, 2)[None]
            values, raw, flow, se = sample_and_score(policy, s, im, seed=31000+index)
            if record['kind'] == 'calibration':
                inputs = policy.packer(policy._batch(s, im, [1]), device=str(policy.device))
                context, _ = policy.policy.encode(inputs)
                normalized = policy.normalize_actions(torch.tensor(raw[:8], device=policy.device))
                verified, _ = chunk_losses(policy, inputs, context, normalized, draws=512)
                calibration_512.extend(verified.tolist())
            error = float(np.linalg.norm(np.array(values['mean_action'])-references[index]))
            row = dict(record, **values, index=index, position=states[index, :3].tolist(),
                       reference=references[index].tolist(), reference_error_m=error,
                       error_to_spread=error/max(values['spread_m'], 1e-9))
            rows.append(row); chunks.append(raw); losses.append(flow); standard_errors.append(se)
            if (index+1) % 5 == 0 or index+1 == len(records):
                progress = dict(done=index+1, total=len(records), elapsed_seconds=time.time()-started)
                (out/'score_progress.json').write_text(json.dumps(progress))
                print('SCORE '+json.dumps(progress), flush=True)
    chunks, losses, standard_errors = np.array(chunks), np.array(losses), np.array(standard_errors)
    calibration = [r for r in rows if r['kind'] == 'calibration']
    thresholds = dict(state_flow_p95=float(np.percentile([r['flow_mean'] for r in calibration], 95)),
                      sample_flow_128_p95=float(np.percentile(losses[:len(calibration)], 95)),
                      sample_flow_512_p95=float(np.percentile(calibration_512, 95)),
                      spread_p95_m=float(np.percentile([r['spread_m'] for r in calibration], 95)))
    for row in rows:
        i = row['index']
        row['flow_ratio'] = row['flow_mean']/thresholds['state_flow_p95']
        row['sample_low_flow_fraction'] = float(np.mean(losses[i] <= thresholds['sample_flow_128_p95']))
        row['narrow'] = row['spread_m'] <= thresholds['spread_p95_m']
    selected = []
    def add(row, reason):
        if row['index'] not in [s['index'] for s in selected]:
            selected.append(dict(row, selection_reason=reason))
    probes = [r for r in rows if r['kind'] == 'probe']
    for row in probes:
        if row['radius'] == 0:
            add(row, 'predeclared_on_line_control')
    for progress in (.5, .9, .98, 1.02):
        group = [r for r in probes if r['progress'] == progress and r['radius'] > 0]
        add(min(group, key=lambda r:r['flow_mean']), 'lowest_mean_flow_at_this_progress')
    remaining = [r for r in probes if r['radius'] > 0 and r['index'] not in [s['index'] for s in selected]]
    for row in sorted(remaining, key=lambda r:r['reference_error_m']/max(r['flow_ratio'], 1e-12), reverse=True)[:2]:
        add(row, 'large_reference_disagreement_relative_to_flow')
    remaining = [r for r in probes if r['index'] not in [s['index'] for s in selected]]
    add(max(remaining, key=lambda r:r['flow_mean']), 'high_flow_comparison')
    assert len(selected) == 12
    (out/'selection.json').write_text(json.dumps(selected, indent=2))
    (out/'scores.json').write_text(json.dumps(dict(name=args.run_dir.name, thresholds=thresholds,
        samples=128, draws_per_sample=128, paired_rollout_draws=512, results=rows), indent=2))
    np.savez_compressed(out/'action_samples.npz', chunks=chunks, losses=losses, loss_standard_errors=standard_errors,
                        states=states, references=references, calibration_sample_losses_512=np.array(calibration_512))
    (out/'scores_done.json').write_text(json.dumps(dict(observations=len(rows), selected=len(selected))))


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from vig_dagger.toy_flow_search import wilson
    names = [f'demos{n}_{v}' for n in (5,20) for v in ('dense_full','moe_7level','dense_7level')]
    all_data, summary = {}, []
    for name in names:
        out = args.root/name
        data = json.loads((out/'scores.json').read_text())
        rollouts = json.loads((out/'rollouts.json').read_text())
        selected = json.loads((out/'selection.json').read_text())
        calibration_means = np.array([r['flow_mean'] for r in data['results'] if r['kind']=='calibration'])
        for row in selected:
            row['flow_id_percentile'] = float(100*np.mean(calibration_means<=row['flow_mean']))
        for trial in rollouts:
            with np.load(out/f"state_{trial['index']:03d}_trial_{trial['trial']:02d}.npz") as trace:
                start_position = trace['states'][0, :3]
                after_position = trace['states'][8, :3] if len(trace['states']) > 8 else trace['final_state'][:3]
                goal = np.array([.45, .15, .35])
                trial['goal_distance_change_first_8_m'] = float(np.linalg.norm(after_position-goal)-np.linalg.norm(start_position-goal))
                trial['initial_action_toward_goal_dot'] = float(trace['first_chunk'][0, :3].dot(goal-start_position))
        states = []
        for row in selected:
            trials = [r for r in rollouts if r['index'] == row['index']]
            assert len(trials) == 8
            failures = sum(not r['success'] for r in trials)
            low_failures = [r for r in trials if not r['success'] and r['first_chunk_flow'] <= data['thresholds']['sample_flow_512_p95']]
            states.append(dict(row, failures=failures, trials=8, failure_rate=failures/8,
                               failure_rate_wilson95=wilson(failures, 8),
                               low_flow_failed_rollouts=len(low_failures),
                               low_flow_failed_rollouts_moving_away_first_8=sum(r['goal_distance_change_first_8_m']>.0005 for r in low_failures),
                               low_flow_failure_candidate=row['flow_ratio']<=1 and failures>=6))
        all_data[name] = dict(data=data, states=states, rollouts=rollouts)
        summary.append(dict(name=name, evaluated_states=12, rollouts=96,
            low_flow_failure_candidates=[r['index'] for r in states if r['low_flow_failure_candidate']],
            narrow_low_flow_failure_candidates=[r['index'] for r in states if r['low_flow_failure_candidate'] and r['narrow']],
            failed_rollouts_with_low_first_chunk_flow=sum(r['low_flow_failed_rollouts'] for r in states),
            failed_rollouts=sum(r['failures'] for r in states), thresholds=data['thresholds'], states=states))
        (out/'paired_outcomes.json').write_text(json.dumps(rollouts,indent=2))
    fig, axes = plt.subplots(2,3,figsize=(17,9),sharey=True)
    for ax, name in zip(axes.flat, names):
        for label_index,r in enumerate(all_data[name]['states']):
            lo, hi = r['failure_rate_wilson95']
            ax.plot([r['flow_ratio']]*2, [lo,hi], color='gray',alpha=.35,lw=1)
            point = ax.scatter(r['flow_ratio'],r['failure_rate'],c=[r['spread_m']*1000],
                cmap='viridis',norm=LogNorm(.03,3),marker='o' if r['radius']==0 else '^',s=65,edgecolors='black',lw=.5)
            ax.annotate(str(r['index']), (r['flow_ratio'],r['failure_rate']),xytext=(3,5+10*(label_index%2)),textcoords='offset points',fontsize=7)
        ax.axvline(1,color='black',ls='--',lw=1)
        ax.set(xscale='log',xlabel='Mean flow / held-out state p95',ylabel='Empirical failure rate (8 trials)',
               title=name,ylim=(-.08,1.16))
        lower,upper=ax.get_xlim()
        ticks=[x for x in (.2,.3,.5,1,2,5,10,20) if lower<=x<=upper]
        ax.set_xticks(ticks,labels=[f'{x:g}' for x in ticks]);ax.minorticks_off();ax.grid(alpha=.2)
    fig.colorbar(point, ax=axes.ravel().tolist(),label='First-action XYZ spread (mm command)',fraction=.02,pad=.02)
    fig.suptitle('Flow versus repeated-rollout failure: circles = on-line controls, triangles = selected offsets\nGray bars: 95% Wilson intervals; labels: observation IDs; selected search set, not population prevalence')
    fig.subplots_adjust(top=.86,right=.87,hspace=.35,wspace=.25)
    fig.savefig(args.root/'flow_vs_failure.png',dpi=160,bbox_inches='tight');plt.close(fig)

    fig, axes = plt.subplots(6,2,figsize=(13,23))
    examples = []
    for i,name in enumerate(names):
        states = all_data[name]['states']; data = all_data[name]['data']
        arrays = np.load(args.root/name/'action_samples.npz')
        for j,kind in enumerate(('on_line','offset')):
            group = [r for r in states if (r['radius']==0)==(kind=='on_line')]
            failing = [r for r in group if r['failure_rate']>=.75]
            chosen = min(failing or group,key=lambda r:r['flow_ratio'])
            idx = chosen['index']; angle=chosen['direction']*np.pi/2
            radial=np.array([np.cos(angle),0,np.sin(angle)])
            samples=arrays['chunks'][idx,:,0,:3]*1000
            ratios=arrays['losses'][idx]/data['thresholds']['sample_flow_128_p95']
            ax=axes[i,j]
            points=ax.scatter(samples@radial,samples[:,1],c=ratios,cmap='coolwarm',norm=LogNorm(.1,10),s=18,alpha=.8)
            ref=arrays['references'][idx]*1000
            ax.scatter(ref@radial,ref[1],marker='x',c='black',s=80,label='Direct-to-B reference')
            for trial in [r for r in all_data[name]['rollouts'] if r['index']==idx]:
                a=np.load(args.root/name/f"state_{idx:03d}_trial_{trial['trial']:02d}.npz")['first_chunk'][0,:3]*1000
                ax.scatter(a@radial,a[1],marker='s',s=60,facecolors='none',edgecolors='green' if trial['success'] else 'red',lw=1.5)
            ax.set(title=f"{name} | state {idx} | {kind}\nprogress {chosen['progress']:.2f}, offset {chosen['radius']*1000:g} mm | failures {chosen['failures']}/8 | mean flow/ID {chosen['flow_ratio']:.2f}",
                   xlabel='Radial command (mm; outward positive)',ylabel='Along-line Y command (mm)')
            ax.grid(alpha=.2)
            examples.append(dict(name=name,index=idx,kind=kind,video=str(args.root/name/'videos'/f'state_{idx:03d}_trial_00.mp4')))
    fig.suptitle('Action distributions at the lowest-flow frequently failing state in each group (or lowest-flow state if none)\nDots: 128 chunks, colored by individual flow / ID sample p95; squares: actual initial rollout actions, green success / red failure\nBlack cross: direct-to-B reference; panels use independent axes; outcomes depend on subsequent sampled actions too',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.colorbar(points,ax=axes.ravel().tolist(),label='Individual chunk flow / ID sample p95',fraction=.015,pad=.02)
    fig.savefig(args.root/'candidate_action_distributions.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for ax,name in zip(axes.flat,names):
        data=all_data[name]
        for trial in data['rollouts']:
            ratio=trial['first_chunk_flow']/data['data']['thresholds']['sample_flow_512_p95']
            y=float(not trial['success'])+(trial['trial']-3.5)*.012
            ax.scatter(ratio,y,c='crimson' if not trial['success'] else 'seagreen',s=18,alpha=.6)
        ax.axvline(1,color='black',ls='--');ax.set(xscale='log',xlabel='Executed first-chunk flow / ID sample p95',
            title=name,yticks=[0,1],yticklabels=['Succeeded','Failed'],ylim=(-.1,1.1));ax.grid(alpha=.2)
    fig.suptitle('Flow of the exact first chunk executed, scored with 512 noise/time draws\nA failed continuation does not prove the first chunk alone caused failure')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(args.root/'executed_chunk_flow.png',dpi=150);plt.close(fig)
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.root/'examples.json').write_text(json.dumps(examples,indent=2))
    print(json.dumps([{k:v for k,v in s.items() if k!='states'} for s in summary],indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['score','report'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    with torch.inference_mode():
        (score if args.mode=='score' else report)(args)
