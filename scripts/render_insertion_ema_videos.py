"""Display-only EMA of raw scores and calibrated percentile curves."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from matplotlib.ticker import MaxNLocator
from render_insertion_raw_videos import render,SCORES
from run_insertion_experiment import REPO,write_json

ALPHA=.1
PERCENTILES={'flow':'percentile','vig_original':'vig_original_percentile','vig':'vig_percentile'}


def ema(values):
    values=np.asarray(values,dtype=np.float64)
    assert values.ndim==1 and len(values)>0 and np.isfinite(values).all()
    result=np.empty_like(values);result[0]=values[0]
    for i in range(1,len(values)):result[i]=ALPHA*values[i]+(1-ALPHA)*result[i-1]
    return result


def gallery(root,run,records):
    dest=root/'ema_01'/run.name;dest.mkdir(parents=True,exist_ok=True)
    import os
    html=['<html><meta charset="utf-8"><title>EMA rollout videos</title><body>',f'<h1>{run.name}: EMA alpha = 0.1</h1>',
        '<p>Three signals: flow, original VIG, and chunk-mean VIG. EMA resets at each episode. Raw scores and existing percentiles are smoothed separately. Raw axes fit this episode\'s smoothed values.</p>']
    for row in sorted(records,key=lambda r:r['case_id']):
        html.append(f'<details><summary>Case {row["case_id"]:02d} / {row["kind"]} / {"success" if row["success"] else "failure"}</summary>')
        for key,label in [('percentile_video','EMA percentile axes'),('raw_video','EMA raw axes')]:
            path=os.path.relpath(root/row[key],dest)
            html.append(f'<p><a href="{path}">{label}</a></p><video controls preload="none" width="512" src="{path}"></video>')
        html.append('</details>')
    (dest/'index.html').write_text('\n'.join(html+['</body></html>']))


def worker(root,run,case_id=None):
    marker=run/'ema01_complete.json'
    if marker.exists() and case_id is None:return
    rows=json.loads((run/'rollouts.json').read_text())
    assert len(rows)==50
    if case_id is not None:rows=[r for r in rows if r['case_id']==case_id]
    records=[];limits={}
    for row in rows:
        raw={'score_steps':None};pct={'score_steps':None};saved={}
        with np.load(run/row['trace']) as trace:
            steps=trace['score_steps'];assert np.array_equal(steps,np.arange(row['steps']))
            raw['score_steps']=steps;pct['score_steps']=steps;saved['score_steps']=steps
            for key,_,_ in SCORES:
                raw[key]=ema(trace[key]);pct[key]=ema(trace[PERCENTILES[key]])
                assert np.all(pct[key]>=-1e-10) and np.all(pct[key]<=100+1e-10)
                saved[key+'_ema']=raw[key];saved[PERCENTILES[key]+'_ema']=pct[key]
        ticks={key:MaxNLocator(nbins=5).tick_values(0,float(values.max())*1.05 if values.max()>0 else 1e-6).tolist() for key,values in raw.items() if key!='score_steps'}
        limits[str(row['case_id'])]=ticks
        videos={}
        for kind,values,axis_ticks,label in [('percentile',pct,{key:list(range(0,101,20)) for key,_,_ in SCORES},'EMA percentile'),('raw',raw,ticks,'EMA raw')]:
            target=run/f'videos_ema01_{kind}'/Path(row['video']).name;target.parent.mkdir(exist_ok=True)
            if not target.exists():render(run/row['video'],target,values,axis_ticks,row['steps'],value_label=label+'; alpha=0.1',axis_label=label)
            videos[kind+'_video']=str(target.relative_to(root))
        trace_out=run/'ema01_traces'/Path(row['trace']).name;trace_out.parent.mkdir(exist_ok=True)
        np.savez_compressed(trace_out,**saved,alpha=ALPHA)
        records.append(dict(case_id=row['case_id'],kind=row['kind'],success=row['success'],steps=row['steps'],**videos,
            source_trace=str((run/row['trace']).relative_to(root)),source_sha256=hashlib.sha256((run/row['trace']).read_bytes()).hexdigest()))
        print('EMA',run.name,row['case_id'],flush=True)
    if case_id is not None:
        write_json(root/'ema01_preview.json',dict(alpha=ALPHA,records=records));return
    write_json(run/'ema01_videos.json',records);write_json(run/'ema01_raw_axis_limits.json',dict(alpha=ALPHA,per_case_ticks=limits))
    gallery(root,run,records)
    write_json(marker,dict(alpha=ALPHA,episodes=50,mp4s=100,finished_at=time.time()))


def coordinate(root):
    assert np.allclose(ema([10,0,0]),[10,9,8.1])
    assert np.allclose(ema([5,5,5]),[5,5,5])
    assert np.array_equal(ema([2]),[2])
    runs=sorted((root/'runs').glob('demos*_epoch*'));assert len(runs)==18
    output=root/'ema_01';output.mkdir(exist_ok=True)
    def publish():
        links=[];records=[]
        for run in runs:
            if (run/'ema01_complete.json').exists():
                records+=json.loads((run/'ema01_videos.json').read_text())
                links.append(f'<li><a href="{run.name}/index.html">{run.name}: 50 episodes, both versions</a></li>')
            else:links.append(f'<li>{run.name}: rendering</li>')
        (output/'index.html').write_text('<html><meta charset="utf-8"><body><h1>EMA alpha = 0.1: flow and both VIG signals</h1><p>Raw and percentile curves are smoothed independently. One causal update per video frame, initialized with the first value; reset per episode.</p><ul>'+''.join(links)+'</ul></body></html>')
        write_json(output/'video_index.json',records)
    publish();write_json(root/'ema01_status.json',dict(phase='rendering',target_mp4s=1800,alpha=ALPHA,workers=8,updated_at=time.time()))
    def launch(run):
        with (root/'logs'/f'{run.name}_ema01.log').open('a') as log:
            subprocess.run([sys.executable,'-u','scripts/render_insertion_ema_videos.py','--root',str(root),'--run-dir',str(run)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (run/'ema01_complete.json').exists()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(launch,run) for run in runs]
        for future in as_completed(futures):
            future.result();publish()
            done=sum((run/'ema01_complete.json').exists() for run in runs)
            write_json(root/'ema01_status.json',dict(phase='rendering',completed_checkpoints=done,target_checkpoints=18,alpha=ALPHA,updated_at=time.time()))
    page=root/'index.html';text=page.read_text();link='<p><a href="ema_01/index.html">EMA-smoothed videos (alpha = 0.1): percentile and raw axes</a></p>'
    if link not in text:page.write_text(text.replace('<body>','<body>'+link))
    write_json(root/'ema01_status.json',dict(phase='complete',checkpoints=18,episodes=900,mp4s=1800,alpha=ALPHA,finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-dir',type=Path);p.add_argument('--case-id',type=int);a=p.parse_args()
    if a.run_dir:worker(a.root,a.run_dir,a.case_id)
    else:
        with (a.root/'ema01.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:coordinate(a.root)
            except BaseException as exc:
                write_json(a.root/'ema01_status.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
