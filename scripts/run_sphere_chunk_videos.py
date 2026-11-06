"""Regenerate all sphere video overlays with complete 16-command sampled chunks."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from run_sphere_rollouts import NAMES,write


def validate(path):
    result=subprocess.run(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],
        capture_output=True,text=True,check=True)
    assert not result.stderr.strip(),(path,result.stderr)
    stream=json.loads(result.stdout)['streams'][0]
    data=np.load(path.with_name(path.stem+'_chunks.npz'))
    old=np.load(path.parent.parent/'annotated_videos'/f'{path.stem}_flow_actions.npz')
    assert stream['width']==1024 and stream['height']==800 and stream['r_frame_rate']=='15/2'
    assert int(stream['nb_read_frames'])==len(data['steps'])+8
    assert data['sampled_chunks'].shape==(len(data['steps']),32,16,7)
    assert np.isfinite(data['sampled_chunks']).all()
    assert np.array_equal(data['flow'],old['flow']) and np.array_equal(data['steps'],old['steps'])
    error=float(np.abs(data['sampled_chunks'][:,:,0,:3]-old['sampled_first_xyz']).max())
    assert error<1e-7
    return dict(path=str(path),decoded_frames=int(stream['nb_read_frames']),horizon=16,
                first_command_max_error_m=error,flow_unchanged=True)


def run(root):
    repo=Path(__file__).resolve().parents[1]
    jobs=[(i,n,0,15 if 'moe' in n else 30) for i,n in enumerate(NAMES)]
    jobs += [(6,'demos5_moe_7level',15,30),(7,'demos20_moe_7level',15,30)]
    processes=[];started=time.time()
    for gpu,name,start,stop in jobs:
        cmd=[sys.executable,str(repo/'scripts/annotate_sphere_chunks.py'),'--root',str(root),'--name',name,
             '--gpu',str(gpu),'--case-start',str(start),'--case-stop',str(stop)]
        with (root/'logs'/f'chunks_{name}_{start}_{stop}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,start,stop,p))
    write(root/'chunk_processes.json',[dict(name=n,start=a,stop=b,pid=p.pid) for n,a,b,p in processes])
    write(root/'chunk_video_protocol.json',dict(videos=180,samples=32,horizon=16,magnification=1,
        path='Cumulative sum of all 16 base-frame translation commands, starting at current EEF position',
        interpretation='Nominal command paths, not simulated future EEF trajectories',
        cyan='Up to eight commands actually executed from the rollout chunk; final eight predictions were not saved',
        orange='All sixteen translations for each of 32 regenerated chunks',
        flow='Copied unchanged from previous videos; same observations, seeds and generated chunks',
        verification='Regenerated first translation commands must match saved originals within 1e-7 m'))
    for rel in ['scripts/annotate_sphere_chunks.py','scripts/annotate_sphere_rollouts.py','scripts/run_sphere_chunk_videos.py']:
        shutil.copy2(repo/rel,root/'source_snapshot'/Path(rel).name)
    while True:
        complete=len(list(root.glob('demos*/chunk_videos/*.json')))
        write(root/'chunk_pipeline.json',dict(state='rendering',completed=complete,total=180,elapsed_seconds=time.time()-started))
        for name,a,b,p in processes:
            if p.poll() is not None and p.returncode:raise RuntimeError(f'{name} cases {a}:{b} exited {p.returncode}')
        if all(p.poll() is not None for _,_,_,p in processes):break
        time.sleep(5)
    paths=sorted(root.glob('demos*/chunk_videos/*.mp4'));assert len(paths)==180
    write(root/'chunk_pipeline.json',dict(state='validating',completed=180,total=180))
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(validate,paths))
    write(root/'chunk_video_validation.json',dict(videos=180,all_frames_decoded=True,all_horizons=16,checks=checks))
    write(root/'chunk_pipeline.json',dict(state='complete',completed=180,total=180,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:run(args.root)
    except BaseException as exc:
        write(args.root/'chunk_pipeline.json',dict(state='failed',error=repr(exc)));raise
