"""Offline uncertainty scoring and plots for the six trained toy policies."""
import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch


def score(args):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_evaluation import measure
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    policy=LibraryPolicy(args.run_dir/'latest.pt','cuda:0')
    with h5py.File(args.root/'observations.hdf5') as f:
        states=f['state'][:]; images=f['images'][:]; references=f['reference_action'][:]
        records=json.loads(f.attrs['records'])
    # Pair scores with actual outcomes using the exact recorded initial state and
    # the first review-video frame. RGB is H.264 decoded here, as in training.
    paired_states=[]; paired_images=[]; paired_refs=[]
    rollout_results=json.loads((args.out/'rollouts.json').read_text())
    for i,rollout in enumerate(rollout_results):
        path=args.out/'videos'/f"{i:02d}_{rollout['kind']}.mp4"
        if not path.exists(): continue
        raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-frames:v','1',
                                     '-f','rawvideo','-pix_fmt','rgb24','-'])
        panel=np.frombuffer(raw,dtype=np.uint8).reshape(284,512,3)
        rgb=np.stack([panel[28:,:256],panel[28:,256:]])
        state=np.load(args.out/f'rollout_{i:02d}.npz')['states'][0]
        delta=np.asarray([.45,.15,.35])-state[:3]
        ref=np.zeros(7,dtype=np.float32); ref[:3]=delta*min(1,.008/max(np.linalg.norm(delta),1e-12)); ref[-1]=1
        paired_states.append(state); paired_images.append(rgb); paired_refs.append(ref)
        records.append(dict(kind='rollout_start',condition=rollout['kind'],rollout_index=i,
                            success=rollout['success'],reason=rollout['reason'],seed=rollout['seed'],
                            image_source='first_decoded_review_mp4_frame'))
    states=np.concatenate([states,np.asarray(paired_states)])
    images=np.concatenate([images,np.asarray(paired_images)])
    references=np.concatenate([references,np.asarray(paired_refs)])
    dataset=Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_training_v1')/'datasets'/('train_5' if 'demos5_' in args.run_dir.name else 'train_20')
    train_positions=[]
    for path in sorted((dataset/'toy_reach/shards').glob('*.hdf5')):
        with h5py.File(path) as f:
            train_positions.append(next(iter(f['data'].values()))['obs/robot0_eef_pos'][:])
    train_positions=np.concatenate(train_positions)
    results=[]; chunks=[]; by_level=[]
    for i,(state,rgb,ref,record) in enumerate(zip(states,images,references,records)):
        s=torch.from_numpy(state).reshape(1,-1).to(policy.device)
        im=torch.from_numpy(rgb).permute(0,3,1,2)[None].to(policy.device)
        values,actions,level_actions=measure(policy,s,im,seed=6000+i)
        mean=np.asarray(values['mean_action'])
        error=float(np.linalg.norm(mean-ref[:3]))
        spread=values['action_spread_m']
        goal_direction=np.asarray([.45,.15,.35])-state[:3]
        cosine=float(mean.dot(goal_direction)/max(np.linalg.norm(mean)*np.linalg.norm(goal_direction),1e-12))
        row=dict(record,**values,index=i,position=state[:3].tolist(),reference_action=ref[:3].tolist(),
                 reference_error_m=error,error_to_spread_ratio=error/max(spread,1e-9),goal_direction_cosine=cosine,
                 nearest_demo_distance_m=float(np.linalg.norm(train_positions-state[:3],axis=1).min()))
        results.append(row); chunks.append(actions); by_level.append(level_actions)
        if (i+1)%10==0:
            (args.out/'score_progress.json').write_text(json.dumps({'done':i+1,'total':len(records)}))
            print(f'SCORED {args.run_dir.name} {i+1}/{len(records)}',flush=True)
    calibration=[r for r in results if r['kind']=='calibration']
    thresholds={key:float(np.percentile([r[key] for r in calibration],95)) for key in ('flow_mean','vig','action_spread_m')}
    for r in results:
        r['flow_relative_to_id_p95']=r['flow_mean']/max(thresholds['flow_mean'],1e-12)
        r['vig_relative_to_id_p95']=r['vig']/max(thresholds['vig'],1e-12)
        r['low_flow']=r['flow_mean']<=thresholds['flow_mean']
        r['low_spread']=r['action_spread_m']<=thresholds['action_spread_m']
        r['low_vig']=r['vig']<=thresholds['vig']
        # A diagnostic flag, not proof that a valid alternate path will fail.
        r['narrow_reference_disagreement']=(r['reference_error_m']>.001 and r['error_to_spread_ratio']>3)
    output=dict(name=args.run_dir.name,checkpoint=str(args.run_dir/'latest.pt'),thresholds=thresholds,
                vig_kind='unseen_image_degradation_sensitivity' if len(policy.factors)==1 else 'resolution_conditioned_vig',
                samples=128,flow_actions=8,flow_loss_draws=512,results=results)
    (args.out/'scores.json').write_text(json.dumps(output,indent=2))
    np.savez_compressed(args.out/'action_samples.npz',chunks=np.asarray(chunks),by_level=np.asarray(by_level),
                        states=states,references=references)
    (args.out/'score_progress.json').write_text(json.dumps({'done':len(results),'total':len(results)}))
    (args.out/'scores_done.json').write_text(json.dumps({'observations':len(results)}))


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[f'demos{n}_{v}' for n in (5,20) for v in ('dense_full','moe_7level','dense_7level')]
    colors=plt.get_cmap('tab10').colors
    all_scores={name:json.loads((args.root/name/'scores.json').read_text()) for name in names}
    summary=[]
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fields=[('flow_relative_to_id_p95','Flow / held-out ID p95',1),('vig_relative_to_id_p95','VIG or sensitivity / held-out ID p95',1),
            ('action_spread_m','Action spread (mm command)',1000),('reference_error_m','Reference disagreement (mm command)',1000),
            ('error_to_spread_ratio','Disagreement / spread',1),('nearest_demo_distance_m','Nearest training EEF position (mm)',1000)]
    radii=[0,.005,.01,.02,.03,.05,.08]
    for j,name in enumerate(names):
        score_data=all_scores[name]; rows=[r for r in score_data['results'] if r['kind']=='probe']
        rollouts=json.loads((args.root/name/'rollouts.json').read_text())
        selected={kind:[r for r in rollouts if r['kind']==kind] for kind in ('id','ood')}
        ood=[r for r in rows if r['requested_radius']>.01]
        flagged=[r for r in ood if r['narrow_reference_disagreement']]
        failed_starts=[r for r in score_data['results'] if r['kind']=='rollout_start' and r['condition']=='ood' and not r['success']]
        id_starts=[r for r in score_data['results'] if r['kind']=='rollout_start' and r['condition']=='id']
        successful_id_starts=[r for r in id_starts if r['success']]
        centers=[r for r in rows if r['requested_radius']==0]
        summary.append(dict(name=name,id_success=sum(r['success'] for r in selected['id']),id_total=len(selected['id']),
                            ood_success=sum(r['success'] for r in selected['ood']),ood_total=len(selected['ood']),
                            ood_probes=len(ood),narrow_disagreement=len(flagged),
                            low_flow_among_narrow_disagreement=sum(r['low_flow'] for r in flagged),
                            low_vig_among_narrow_disagreement=sum(r['low_vig'] for r in flagged),
                            flow_warning_rate_ood=float(np.mean([not r['low_flow'] for r in ood])),
                            vig_warning_rate_ood=float(np.mean([not r['low_vig'] for r in ood])),
                            failed_ood_rollout_starts=len(failed_starts),
                            failed_ood_starts_with_low_flow=sum(r['low_flow'] for r in failed_starts),
                            failed_ood_starts_with_low_vig=sum(r['low_vig'] for r in failed_starts),
                            failed_ood_starts_with_low_spread=sum(r['low_spread'] for r in failed_starts),
                            scored_id_starts=len(id_starts),
                            scored_id_starts_with_flow_warning=sum(not r['low_flow'] for r in id_starts),
                            scored_successful_id_starts=len(successful_id_starts),
                            successful_id_starts_with_flow_warning=sum(not r['low_flow'] for r in successful_id_starts),
                            center_probes=len(centers),
                            center_probes_with_flow_warning=sum(not r['low_flow'] for r in centers),
                            thresholds=score_data['thresholds']))
        label=name.replace('demos','').replace('_',' ')
        for ax,(key,title,scale) in zip(axes.flat,fields):
            groups=[np.asarray([r[key]*scale for r in rows if r['requested_radius']==radius]) for radius in radii]
            med=np.asarray([np.median(g) for g in groups])
            ax.plot(np.asarray(radii)*100,med,label=label,color=colors[j])
            ax.fill_between(np.asarray(radii)*100,[np.percentile(g,25) for g in groups],
                            [np.percentile(g,75) for g in groups],color=colors[j],alpha=.10)
            ax.set(xlabel='Distance from nominal line (cm)',ylabel=title)
            ax.axvline(1,color='black',ls='--',alpha=.4); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[0,0].axhline(1,color='black',ls=':',alpha=.5)
    axes[0,1].axhline(1,color='black',ls=':',alpha=.5)
    fig.suptitle('Final policies: medians and interquartile ranges across shared physical probes')
    fig.tight_layout(); fig.savefig(args.root/'ood_scores.png',dpi=150); plt.close(fig)

    # Representative action clouds: fixed midpoint, direction 0, matched across policies and radii.
    radii=[0,.01,.03,.05,.08]
    fig,axes=plt.subplots(6,5,figsize=(17,18),sharex=True,sharey=True)
    for row,name in enumerate(names):
        data=np.load(args.root/name/'action_samples.npz')
        results=all_scores[name]['results']
        angle=next(r['angle'] for r in results if r['kind']=='probe' and r['progress']==.5 and r['requested_radius']==.01 and r['direction']==0)
        radial=np.array([np.cos(angle),0,np.sin(angle)])
        for col,radius in enumerate(radii):
            result=next(r for r in results if r['kind']=='probe' and r['progress']==.5 and r['requested_radius']==radius and r['direction']==0)
            idx=result['index']; samples=data['chunks'][idx,:,0,:3]*1000
            ref=data['references'][idx,:3]*1000
            ax=axes[row,col]
            ax.scatter(samples@radial,samples[:,1],s=5,alpha=.4)
            ax.scatter(ref@radial,ref[1],marker='x',c='red',s=60)
            ax.set_title(f"{radius*100:g} cm | flow {result['flow_mean']:.3f}",fontsize=9)
            ax.grid(alpha=.2)
            if col==0: ax.set_ylabel(name+'\nAlong-line Y command (mm)',fontsize=8)
            if row==5: ax.set_xlabel('Radial command (mm; outward positive)')
    fig.suptitle('First-action distributions: 128 samples; red cross = direct-to-B reference')
    fig.tight_layout(rect=(0,0,1,.975)); fig.savefig(args.root/'action_distributions.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,name in zip(axes.flat,names):
        for r in all_scores[name]['results']:
            if r['kind']!='rollout_start': continue
            failed=not r['success']
            ax.scatter(max(r['flow_relative_to_id_p95'],1e-5),max(r['vig_relative_to_id_p95'],1e-5),
                       marker='x' if failed else 'o',color='crimson' if failed else 'seagreen',s=55)
        ax.axvline(1,color='black',ls='--',alpha=.5); ax.axhline(1,color='black',ls='--',alpha=.5)
        ax.set(xscale='log',yscale='log',xlabel='Flow / ID p95',ylabel='VIG or sensitivity / ID p95',title=name)
        ax.grid(alpha=.2)
    fig.suptitle('Scores at recorded rollout starts: red x = failed; green circle = succeeded\nBelow 1 means no warning; RGB decoded from saved initial video frames')
    fig.tight_layout(); fig.savefig(args.root/'failure_warnings.png',dpi=150); plt.close(fig)
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['score','report'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.mode=='score':
        with torch.inference_mode(): score(args)
    else: report(args)
