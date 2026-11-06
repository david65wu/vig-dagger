"""Generate simple whole-rollout flow and conditional action plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

KINDS=['familiar_id','new_id','position_ood','orientation_ood','combined_ood']
LABELS=['Familiar ID','New ID','Position OOD','Orientation OOD','Combined OOD']
COLORS=['#2474b5','#21a585','#ed9b30','#9b64bb','#d94a56']
EPOCHS=[10,25,50,100,200,300]


def policy_label(root):
    path=root/'protocol.json'
    return json.loads(path.read_text()).get('policy_label','Dense full-resolution policy') if path.exists() else 'Dense full-resolution policy'


def plot_success_rates(root,rows):
    """Plot observed success fractions, keeping the five conditions separate."""
    fig,axes=plt.subplots(1,3,figsize=(18,5.8),sharey=True)
    markers=['o','s','^','D','v']
    styles=['-','--','-.',':','-']
    for ax,count in zip(axes,[50,200,500]):
        for kind,label,color,marker,style in zip(KINDS,LABELS,COLORS,markers,styles):
            selected=sorted((r for r in rows if int(r['demos'])==count and r['case']==kind),key=lambda r:int(r['epoch']))
            assert [int(r['epoch']) for r in selected]==EPOCHS
            assert all(int(r['rollouts'])==10 and 0<=int(r['successes'])<=10 for r in selected)
            rate=[100*int(r['successes'])/int(r['rollouts']) for r in selected]
            ax.plot(EPOCHS,rate,label=label,color=color,marker=marker,linestyle=style,
                    linewidth=2,markersize=6,markerfacecolor='white',markeredgewidth=1.5)
        ax.set_title(f'{count} training demos',fontsize=14)
        ax.set_xlabel('Training epoch',fontsize=12)
        ax.set_xticks(EPOCHS);ax.set_xlim(0,310);ax.set_ylim(-3,103)
        ax.set_yticks(np.arange(0,101,10));ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Rollout success rate (%)',fontsize=12)
    fig.suptitle('Plug insertion: success rate across training checkpoints\n'+policy_label(root),fontsize=15,y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.88),ncol=5,frameon=False)
    fig.text(.5,.025,'10 matched rollouts per condition and checkpoint; one success = 10 percentage points.\n'
             'Success requires 10 consecutive steps within the original insertion tolerance while maintaining the grasp.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.105,1,.79))
    for suffix in ['png','pdf']:fig.savefig(root/f'success_rates.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot(root):
    rows=[];data={};videos=[]
    protocol=json.loads((root/'protocol.json').read_text())
    has_vig=protocol.get('vig_enabled',False)
    for count in [50,200,500]:
        for epoch in EPOCHS:
            run=root/'runs'/f'demos{count}_epoch{epoch:03d}'
            assert (run/'complete.json').exists(),run
            episodes=json.loads((run/'rollouts.json').read_text());assert len(episodes)==50
            if has_vig:
                flow_cal=np.load(run/'calibration.npz');vig_cal=np.load(run/'calibration_vig.npz')
                assert bool(vig_cal.get('temporal_average',False))==protocol.get('vig_temporal_average',False)
                assert np.array_equal(flow_cal['indices'],vig_cal['indices'])
                flow_reference=np.sort(flow_cal['flow']);vig_reference=np.sort(vig_cal['vig'])
                if protocol.get('vig_both',False):
                    original_cal=np.load(run/'calibration_vig_original.npz')
                    assert np.array_equal(flow_cal['indices'],original_cal['indices'])
                    original_reference=np.sort(original_cal['vig'])
            for kind in KINDS:
                selected=[r for r in episodes if r['kind']==kind];assert len(selected)==10
                traces=[np.load(run/r['trace']) for r in selected]
                if protocol.get('score_stride',8)==1:
                    for trace in traces:
                        assert np.array_equal(trace['score_steps'],np.arange(len(trace['actions'])))
                        assert np.array_equal(trace['replan_steps'],np.arange(0,len(trace['actions']),8))
                        for step,action in enumerate(trace['actions']):
                            assert np.array_equal(action,trace['replan_chunks'][step//8,step%8])
                raw=np.concatenate([x['flow'] for x in traces]);percentile=np.concatenate([x['percentile'] for x in traces])
                assert np.isfinite(raw).all() and np.isfinite(percentile).all()
                data[(count,epoch,kind)]=(raw,percentile)
                rows.append(dict(demos=count,epoch=epoch,case=kind,successes=sum(r['success'] for r in selected),
                    rollouts=len(selected),scored_observations=len(raw),median_flow=float(np.median(raw)),
                    median_percentile=float(np.median(percentile))))
                if has_vig:
                    for trace in traces:
                        assert len(trace['vig'])==len(trace['flow'])==len(trace['score_steps'])
                        assert trace['vig_gap'].shape==(len(trace['vig']),6)
                        assert np.allclose(trace['vig'],trace['vig_gap'].max(axis=1))
                        assert np.allclose(trace['percentile'],100*np.searchsorted(flow_reference,trace['flow'],side='right')/len(flow_reference))
                        assert np.allclose(trace['vig_percentile'],100*np.searchsorted(vig_reference,trace['vig'],side='right')/len(vig_reference))
                        if protocol.get('vig_both',False):
                            assert len(trace['vig_original'])==len(trace['flow'])
                            assert np.allclose(trace['vig_original'],trace['vig_original_gap'].max(axis=1))
                            assert np.allclose(trace['vig_original_percentile'],100*np.searchsorted(original_reference,trace['vig_original'],side='right')/len(original_reference))
                    vig=np.concatenate([x['vig'] for x in traces]);vig_percentile=np.concatenate([x['vig_percentile'] for x in traces])
                    assert len(vig)==len(raw) and np.isfinite(vig).all() and np.isfinite(vig_percentile).all()
                    data[(count,epoch,kind)]+=(vig,vig_percentile)
                    rows[-1].update(median_vig=float(np.median(vig)),median_vig_percentile=float(np.median(vig_percentile)))
                for r in selected:
                    if 'video' in r:videos.append(dict(demos=count,epoch=epoch,case=kind,success=r['success'],
                        case_id=r['case_id'],path=str((run/r['video']).relative_to(root))))
                    if 'video_raw' in r:videos[-1]['raw_path']=str((run/r['video_raw']).relative_to(root))
    with (root/'success_rates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'summary.json').write_text(json.dumps(rows,indent=2))
    (root/'video_index.json').write_text(json.dumps(videos,indent=2))
    plot_success_rates(root,rows)
    score_figures=[(1,'Flow percentile relative to each checkpoint own training demos','flow_percentiles.png'),
                   (0,'Raw flow self-consistency loss','flow_raw.png')]
    if has_vig:score_figures += [(3,'VIG percentile relative to each checkpoint own training demos','vig_percentiles.png'),
                                (2,('Raw VIG: Wasserstein distance after averaging each 16-step chunk' if protocol.get('vig_temporal_average',False) else 'Raw VIG: maximum resolution-dependent action-distribution distance'),'vig_raw.png')]
    for field,title,filename in score_figures:
        fig,axes=plt.subplots(3,1,figsize=(16,13),sharex=True)
        for ax,count in zip(axes,[50,200,500]):
            for j,kind in enumerate(KINDS):
                arrays=[data[(count,e,kind)][field] for e in EPOCHS]
                boxes=ax.boxplot(arrays,positions=np.arange(6)+(j-2)*.15,widths=.12,patch_artist=True,
                    showfliers=False,manage_ticks=False,whis=(5,95))
                for box in boxes['boxes']:box.set(facecolor=COLORS[j],alpha=.7)
                for median in boxes['medians']:median.set(color='black')
            ax.set_title(f'{count} training demos');ax.set_ylabel('Training percentile' if field in [1,3] else ('Raw VIG' if field==2 else 'Raw flow loss'))
            ax.grid(axis='y',alpha=.2)
            if field in [1,3]:ax.set_ylim(-2,102)
            else:ax.set_yscale('log')
        axes[-1].set_xticks(range(6),[str(e) for e in EPOCHS]);axes[-1].set_xlabel('Training epoch')
        fig.suptitle(policy_label(root)+'\n'+title+f'\nWhole rollouts, scored every {protocol.get("score_stride",8)} step(s); boxes: 25–75%, median; whiskers: 5–95%',y=.995)
        fig.legend([Patch(color=c,alpha=.7) for c in COLORS],LABELS,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5)
        fig.text(.5,.014,'Pooled observations: longer rollouts contribute more scores. Calibrated percentiles are not failure probabilities.',ha='center')
        fig.tight_layout(rect=(0,.035,1,.89));fig.savefig(root/filename,dpi=160);plt.close(fig)
    for count in [50,200,500]:
        for mode in ['translation','rotation']:
            fig,axes=plt.subplots(2,3,figsize=(15,10),sharex=True,sharey=True)
            all_values=[]
            for ax,epoch in zip(axes.flat,EPOCHS):
                for k,kind in enumerate(KINDS):
                    trace=np.load(root/'runs'/f'demos{count}_epoch{epoch:03d}'/'traces'/f'{k*10:02d}_{kind}.npz')
                    if mode=='translation':v=trace['initial_projected_displacements_mm'][:,:2]
                    else:v=trace['initial_angular_displacements_deg'][:,1:3]
                    all_values.append(v)
                    ax.scatter(v[:,0],v[:,1],s=12,alpha=.5,c=COLORS[k],label=LABELS[k])
                ax.set_title(f'Epoch {epoch}');ax.axhline(0,c='gray',lw=.5);ax.axvline(0,c='gray',lw=.5);ax.grid(alpha=.2)
                ax.set_xlabel('Insertion-axis command (mm)' if mode=='translation' else 'Pitch-axis rotation command (deg)')
                ax.set_ylabel('Lateral command (mm)' if mode=='translation' else 'Yaw-axis rotation command (deg)')
            values=np.concatenate(all_values)
            lower=values.min(axis=0);upper=values.max(axis=0);pad=np.maximum((upper-lower)*.08,.1)
            for ax in axes.flat:ax.set_xlim(lower[0]-pad[0],upper[0]+pad[0]);ax.set_ylim(lower[1]-pad[1],upper[1]+pad[1])
            fig.suptitle(policy_label(root)+f' | {count} demos: first command from 128 sampled 16-action chunks at the initial observation\n'
                'FK displacement in socket frame; illustrative pair 0. New ID and OOD share the same socket; familiar ID uses a training scene.',fontsize=12)
            fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.92),ncol=5)
            fig.tight_layout(rect=(0,0,1,.86));fig.savefig(root/f'action_{mode}_demos{count}.png',dpi=160);plt.close(fig)
    # Verify the same saved start was used across all 18 policies/checkpoints.
    starts={}
    for run in sorted((root/'runs').glob('demos*_epoch*')):
        for row in json.loads((run/'rollouts.json').read_text()):
            starts.setdefault(row['case_id'],[]).append(np.asarray(row['start_tcp'])[:3])
    spread={str(k):float(np.max(np.linalg.norm(np.asarray(v)-v[0],axis=1))) for k,v in starts.items()}
    assert max(spread.values())<.001,'Physical starting states differ across policies'
    (root/'matched_start_validation.json').write_text(json.dumps(dict(max_position_difference_m=max(spread.values()),per_case=spread),indent=2))
    # Lightweight browseable index links every representative outcome video.
    html=['<html><meta charset="utf-8"><title>Insertion ID/OOD analysis</title><body>',
          '<h1>Insertion ID/OOD analysis</h1><p>'+policy_label(root)+'</p><p>18 checkpoints; 50 matched starts each; 900 rollouts.</p>',
          '<p><a href="success_rates.csv">Success rates and flow summary CSV</a></p>']
    for image in ['success_rates.png','flow_percentiles.png','flow_raw.png']+(['vig_percentiles.png','vig_raw.png'] if has_vig else [])+[f'action_{m}_demos{n}.png' for n in [50,200,500] for m in ['translation','rotation']]:
        html.append(f'<p><a href="{image}"><img src="{image}" style="max-width:100%"></a></p>')
    if protocol.get('full_rollout_videos',False):
        html.append('<h2>All rollout videos by checkpoint</h2><p>50 videos per checkpoint, 10 per condition. Flow and VIG are measured on the same observations.</p><ul>')
        for count in [50,200,500]:
            for epoch in EPOCHS:
                html.append(f'<li><a href="runs/demos{count}_epoch{epoch:03d}/index.html">{count} demos / epoch {epoch}: all 50 videos</a></li>')
    else:
        html.append('<h2>Representative videos</h2><ul>')
        for row in videos:
            html.append(f'<li><a href="{row["path"]}">{row["demos"]} demos / epoch {row["epoch"]} / {row["case"]} / {"success" if row["success"] else "failure"}</a></li>')
    html.append('</ul></body></html>');(root/'index.html').write_text('\n'.join(html))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    plot(parser.parse_args().root)
