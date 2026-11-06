"""Four-panel videos and whole-rollout boxplots for fresh four-score rollouts."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from matplotlib.ticker import MaxNLocator
from render_insertion_raw_videos import render,SCORES
from render_insertion_ema_videos import ema
from run_insertion_experiment import write_json

FOUR_SCORES=SCORES+[('vig_lowest','VIG (lowest ref)','#ffc16e')]
PERCENTILES={'flow':'percentile','vig_original':'vig_original_percentile','vig':'vig_percentile','vig_lowest':'vig_lowest_percentile'}
EPOCHS=[10,25,50,100,200,300]


def render_row(root,name,row,preview=False):
    run=root/'runs'/name;src=run
    with np.load(run/row['trace']) as saved:trace={k:saved[k] for k in saved.files}
    raw={k:ema(trace[k]) for k,_,_ in FOUR_SCORES};raw['score_steps']=trace['score_steps']
    ticks={k:MaxNLocator(nbins=5).tick_values(0,max(float(raw[k].max())*1.05,1e-6)).tolist() for k,_,_ in FOUR_SCORES}
    args=[('raw',raw,ticks)]
    if not preview:
        pct={k:ema(trace[PERCENTILES[k]]) for k,_,_ in FOUR_SCORES};pct['score_steps']=trace['score_steps']
        args.append(('percentile',pct,{k:list(range(0,101,20)) for k,_,_ in FOUR_SCORES}))
    files={}
    for kind,values,axis in args:
        path=run/('preview_raw' if preview else f'videos_ema01_{kind}')/Path(row['video']).name;path.parent.mkdir(exist_ok=True)
        if not path.exists():render(src/row['video'],path,values,axis,row['steps'],value_label=f'EMA {kind}; alpha=0.1',axis_label=f'EMA {kind}',scores=FOUR_SCORES)
        files['video_'+kind]=str(path.relative_to(root))
    if not preview:
        dest=run/'ema01_traces'/Path(row['trace']).name;dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,score_steps=trace['score_steps'],alpha=.1,**{k+'_ema':raw[k] for k,_,_ in FOUR_SCORES},**{PERCENTILES[k]+'_ema':pct[k] for k,_,_ in FOUR_SCORES})
    return dict(case_id=row['case_id'],kind=row['kind'],success=row['success'],steps=row['steps'],raw_ticks=ticks,**files)


def render_run(root,name):
    run=root/'runs'/name
    if (run/'complete.json').exists():return
    assert (run/'scoring_complete.json').exists()
    rows=json.loads((root/'runs'/name/'rollouts.json').read_text());assert len(rows)==50
    reference=np.sort(np.load(run/'calibration_vig_lowest.npz')['vig'])
    records=[]
    for row in rows:
        with np.load(run/row['trace']) as t:
            assert len(t['vig_lowest'])==row['steps']
            assert np.allclose(t['vig_lowest'],t['vig_lowest_gap'].max(axis=1))
            assert np.array_equal(t['vig_lowest_percentile'],100*np.searchsorted(reference,t['vig_lowest'],side='right')/len(reference))
        records.append(render_row(root,name,row))
        print('FOUR PANEL VIDEO',name,row['case_id'],flush=True)
    write_json(run/'videos.json',records)
    html=['<html><meta charset="utf-8"><body>',f'<h1>{name}: flow + three VIG variants</h1>',
          '<p>Fresh policy rollouts; all four scores computed on the same observations. Fourth panel: VIG with lowest resolution (factor 64) as reference, original per-action aggregation. EMA alpha=0.1, independently on raw values and training-calibrated percentiles. Raw axes fit each episode. Success is measured in these fresh rollouts. Observations and all seven-resolution sample sets are retained.</p>']
    for record in records:
        html.append(f'<details><summary>Case {record["case_id"]:02d} / {record["kind"]} / {"success" if record["success"] else "failure"}</summary>')
        for k in ['percentile','raw']:
            link=os.path.relpath(root/record['video_'+k],run)
            html.append(f'<p><a href="{link}">EMA {k}</a></p><video controls preload="none" width="512" src="{link}"></video>')
        html.append('</details>')
    (run/'index.html').write_text('\n'.join(html+['</body></html>']))
    write_json(run/'complete.json',dict(episodes=50,mp4s=100,alpha=.1,finished_at=time.time()))


def publish(root):
    records=[];links=[]
    for n in [50,200,500]:
        for e in EPOCHS:
            name=f'demos{n}_epoch{e:03d}';run=root/'runs'/name
            if (run/'complete.json').exists():
                records+=json.loads((run/'videos.json').read_text())
                links.append(f'<li><a href="runs/{name}/index.html">{name}: 50 episodes / both versions</a></li>')
            else:links.append(f'<li>{name}: processing</li>')
    plot_links=''.join(f'<li><a href="{filename}">{filename}</a></li>' for filename in ['success_rates.png','flow_raw.png','flow_percentiles.png','vig_original_raw.png','vig_original_percentiles.png','vig_raw.png','vig_percentiles.png','vig_lowest_raw.png','vig_lowest_percentiles.png'] if (root/filename).exists())
    (root/'index.html').write_text('<html><meta charset="utf-8"><body><h1>Flow, VIG, chunk-mean VIG, and lowest-reference VIG</h1><p>Fresh policy rollouts using the same 50 evaluation starting cases per checkpoint. All four curves use EMA alpha=0.1 in videos; summary boxplots use unsmoothed scores.</p><ul>'+plot_links+''.join(links)+'</ul></body></html>')
    write_json(root/'video_index.json',records)


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from plot_insertion_analysis import KINDS,LABELS,COLORS,plot as plot_existing
    plot_existing(root)
    data={};summary=[]
    for n in [50,200,500]:
        for e in EPOCHS:
            name=f'demos{n}_epoch{e:03d}';run=root/'runs'/name
            assert (run/'scoring_complete.json').exists()
            rows=json.loads((root/'runs'/name/'rollouts.json').read_text())
            for kind in KINDS:
                selected=[r for r in rows if r['kind']==kind];assert len(selected)==10
                arrays=[];percentiles=[];original=[];original_percentiles=[]
                for row in selected:
                    with np.load(run/row['trace']) as t:
                        arrays.append(t['vig_lowest']);percentiles.append(t['vig_lowest_percentile'])
                        original.append(t['vig_original']);original_percentiles.append(t['vig_original_percentile'])
                raw=np.concatenate(arrays);pct=np.concatenate(percentiles)
                data[n,e,kind]=(raw,pct,np.concatenate(original),np.concatenate(original_percentiles))
                summary.append(dict(demos=n,epoch=e,condition=kind,rollouts=10,frames=len(raw),successes=sum(r['success'] for r in selected),raw_quantiles=np.percentile(raw,[5,25,50,75,95]).tolist(),percentile_quantiles=np.percentile(pct,[5,25,50,75,95]).tolist()))
    for field,filename,label in [(0,'vig_lowest_raw.png','Raw VIG (lowest-resolution reference)'),(1,'vig_lowest_percentiles.png','Training percentile'),(2,'vig_original_raw.png','Raw VIG (full-resolution reference)'),(3,'vig_original_percentiles.png','Training percentile')]:
        fig,axes=plt.subplots(3,1,figsize=(16,13),sharex=True)
        for ax,n in zip(axes,[50,200,500]):
            for j,kind in enumerate(KINDS):
                boxes=ax.boxplot([data[n,e,kind][field] for e in EPOCHS],positions=np.arange(6)+(j-2)*.15,widths=.12,patch_artist=True,showfliers=False,manage_ticks=False,whis=(5,95))
                for b in boxes['boxes']:b.set(facecolor=COLORS[j],alpha=.7)
                for b in boxes['medians']:b.set(color='black')
            ax.set_title(f'{n} training demos');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
            if field%2:ax.set_ylim(-2,102);ax.set_yticks(range(0,101,20))
            else:ax.set_yscale('log')
        axes[-1].set_xticks(range(6),[str(e) for e in EPOCHS]);axes[-1].set_xlabel('Training epoch')
        reference_label='lowest-resolution' if field<2 else 'full-resolution'
        fig.suptitle(f'Plug insertion: VIG with {reference_label} reference\nWhole fresh rollouts; every frame; unsmoothed scores\nBoxes: 25–75%, median; whiskers: 5–95%',y=.995)
        fig.legend([Patch(color=c,alpha=.7) for c in COLORS],LABELS,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5)
        fig.text(.5,.014,'Pooled frames: longer episodes contribute more scores. Percentiles use each checkpoint’s own training demos.',ha='center')
        fig.tight_layout(rect=(0,.035,1,.89));fig.savefig(root/filename,dpi=160);fig.savefig((root/filename).with_suffix('.pdf'));plt.close(fig)
    write_json(root/'vig_lowest_summary.json',summary)
    publish(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name');p.add_argument('--preview',action='store_true');p.add_argument('--plot',action='store_true');a=p.parse_args()
    if a.plot:plot(a.root)
    elif a.preview:
        row=next(r for r in json.loads((a.root/'runs'/a.name/'rollouts.json').read_text()) if r['case_id']==22)
        record=render_row(a.root,a.name,row,True);write_json(a.root/'preview_video.json',record)
    else:render_run(a.root,a.name)
