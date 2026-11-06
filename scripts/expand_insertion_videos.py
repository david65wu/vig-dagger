"""Expand dense checkpoint videos with fresh matched-state policy rollouts.

Retain existing representative recordings; create missing videos with newly
measured flow and the checkpoint's saved calibration, separate from original results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_insertion_experiment import run_job,write_json,REPO

EPOCHS=[10,25,50,100,200,300]


def publish(root,out):
    records=[];pages=[]
    for count in [50,200,500]:
        for epoch in EPOCHS:
            name=f'demos{count}_epoch{epoch:03d}';manifest=out/'runs'/name/'videos.json'
            rows=json.loads(manifest.read_text()) if manifest.exists() else []
            records.extend(rows)
            page=['<html><meta charset="utf-8"><title>Dense insertion videos</title><body>',
                f'<h1>{name}: {len(rows)}/50 videos</h1>',
                '<p>Hand + side cameras; 16 action targets; raw flow and training percentile. Percentile ticks: 0, 20, 40, 60, 80, 100.</p>',
                '<p>Original representative recordings are retained. Additional videos are fresh policy rollouts from the same saved starts, with newly measured flow and the same checkpoint calibration. Original summary plots still describe the original 900 evaluations.</p>']
            for kind in ['familiar_id','new_id','position_ood','orientation_ood','combined_ood']:
                page.append(f'<h2>{kind}</h2>')
                for row in sorted((x for x in rows if x['kind']==kind),key=lambda x:x['case_id']):
                    path=os.path.relpath(out/row['video'],out/'runs'/name)
                    label=f'Case {row["case_id"]:02d} | {"success" if row["success"] else "failure"} | {row["source"]}'
                    page.append(f'<details><summary>{html.escape(label)}</summary><p><a href="{path}">Open MP4</a></p><video controls preload="none" width="512" src="{path}"></video></details>')
            page.append('</body></html>');target=out/'runs'/name/'index.html';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('\n'.join(page))
            pages.append(f'<li><a href="runs/{name}/index.html">{count} demos / epoch {epoch}: {len(rows)}/50 videos</a></li>')
    write_json(out/'video_index.json',records)
    (out/'index.html').write_text('<html><meta charset="utf-8"><title>All dense insertion videos</title><body><h1>Dense full-resolution insertion: all rollout videos</h1>'
        f'<p>{len(records)}/900 videos available. 50 starts per checkpoint: 10 per ID/OOD condition.</p><ul>'+''.join(pages)+'</ul></body></html>')
    index=root/'index.html';link='<p><a href="full_rollout_videos/index.html">All 50 rollout videos per dense checkpoint (expanded gallery)</a></p>'
    content=index.read_text()
    if 'full_rollout_videos/index.html' not in content:index.write_text(content.replace('</body>',link+'</body>'))
    return records


def worker(root,training,out,count,epoch,limit=None):
    import numpy as np
    import torch
    from vig_dagger.insertion_env import make_insertion_env
    torch.set_num_threads(2)
    name=f'demos{count}_epoch{epoch:03d}';source=root/'runs'/name;dest=out/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
    original=json.loads((source/'rollouts.json').read_text());cases={x['case_id']:x for x in json.loads((root/'cases.json').read_text())}
    manifest=dest/'videos.json';rows=json.loads(manifest.read_text()) if manifest.exists() else []
    done={x['case_id'] for x in rows}
    for row in original:
        if row['case_id'] not in done and 'video' in row:
            rows.append(dict(case_id=row['case_id'],kind=row['kind'],success=row['success'],demos=count,epoch=epoch,
                source='original recording',video=os.path.relpath(source/row['video'],out)))
            done.add(row['case_id'])
    write_json(manifest,rows)
    env=make_insertion_env(render_backend='cuda:0');policy=None;generated=0
    reference=np.sort(np.load(source/'calibration.npz')['flow'])
    calibration_sha=hashlib.sha256((source/'calibration.npz').read_bytes()).hexdigest()
    try:
        for row in original:
            if row['case_id'] in done:continue
            stem=f'{row["case_id"]:02d}_{row["kind"]}';path=dest/'videos'/f'{stem}.mp4'
            from vig_dagger.policy import LibraryPolicy
            from vig_dagger.insertion_analysis import run_rollout
            if policy is None:policy=LibraryPolicy(training/'policies'/f'demos{count}_dense_full'/f'epoch-{epoch:03d}.pt','cuda:0')
            fresh=run_rollout(policy,env,cases[row['case_id']],root,dest,reference,name+' NEW',{})
            write_json(dest/'fresh_rollouts'/f'{stem}.json',fresh)
            result=dict(case_id=row['case_id'],kind=row['kind'],success=fresh['success'],demos=count,epoch=epoch,
                source='fresh policy rollout',video=os.path.relpath(dest/fresh['video'],out),calibration_sha256=calibration_sha)
            path=dest/fresh['video']
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,check=True)
            rows.append(result);write_json(manifest,rows);generated+=1
            write_json(dest/'status.json',dict(phase='rendering',videos=len(rows),total=50,updated_at=time.time()))
            print(f'VIDEO {name} {stem}: {result["source"]}; {len(rows)}/50',flush=True)
            if limit and generated>=limit:break
    finally:env.close()
    if len(rows)==50:write_json(dest/'complete.json',dict(videos=50,finished_at=time.time()))


def coordinate(root,training,out):
    jobs=[(n,e) for n in [50,200,500] for e in EPOCHS];import queue
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    write_json(out/'pipeline.json',dict(phase='rendering',pid=os.getpid(),gpus=[1,2,3,4,5],target_videos=900,updated_at=time.time()))
    def lane(gpu):
        while True:
            try:count,epoch=pending.get_nowait()
            except queue.Empty:return
            name=f'demos{count}_epoch{epoch:03d}';dest=out/'runs'/name
            if not (dest/'complete.json').exists():
                run_job([sys.executable,'-u','scripts/expand_insertion_videos.py','worker','--root',str(root),'--training',str(training),'--out',str(out),'--count',str(count),'--epoch',str(epoch)],
                    out/'logs'/f'{name}.log',gpu,out/'processes'/f'{name}.json',f'--out {out} --count {count} --epoch {epoch}')
            assert (dest/'complete.json').exists()
    publish(root,out)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(lane,gpu) for gpu in [1,2,3,4,5]]
        while not all(f.done() for f in futures):
            time.sleep(15);records=publish(root,out)
            write_json(out/'progress.json',dict(videos=len(records),total=900,updated_at=time.time()))
        for f in futures:f.result()
    records=publish(root,out);assert len(records)==900
    write_json(out/'pipeline.json',dict(phase='complete',videos=900,generated=sum(r['source']!='original recording' for r in records),
        fresh_rollouts=sum(r['source']=='fresh policy rollout' for r in records),finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['worker','coordinate']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int);p.add_argument('--epoch',type=int);p.add_argument('--limit',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.command=='worker':worker(a.root,a.training,a.out,a.count,a.epoch,a.limit)
    else:
        with (a.out/'coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:coordinate(a.root,a.training,a.out)
            except BaseException as exc:
                write_json(a.out/'pipeline.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
