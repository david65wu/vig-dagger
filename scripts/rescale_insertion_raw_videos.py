"""Refresh raw MP4s with ranges fitted separately to each rollout and metric."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from run_insertion_experiment import REPO,write_json


def coordinate(root):
    runs=sorted((root/'runs').glob('demos*_epoch*'))
    assert len(runs)==18
    status=root/'raw_rescale.json'
    write_json(status,dict(phase='rendering',target_videos=900,workers=8,updated_at=time.time()))
    def worker(run):
        log=root/'logs'/f'{run.name}_raw_rollout_scale.log'
        with log.open('a') as stream:
            subprocess.run([sys.executable,'-u','scripts/render_insertion_raw_videos.py','--run-dir',str(run),'--scale','rollout'],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert (run/'raw_rollout_scale_complete.json').exists()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,run) for run in runs]
        for future in as_completed(futures):
            future.result()
            done=sum((run/'raw_rollout_scale_complete.json').exists() for run in runs)
            write_json(status,dict(phase='rendering',complete_checkpoints=done,target_checkpoints=18,updated_at=time.time()))
    video_index=json.loads((root/'video_index.json').read_text())
    lookup={(int(run.name.split('_')[0][5:]),int(run.name.split('epoch')[1]),r['case_id']):(run,r) for run in runs for r in json.loads((run/'rollouts.json').read_text())}
    for item in video_index:
        run,row=lookup[item['demos'],item['epoch'],item['case_id']]
        item['raw_path']=str((run/row['video_raw']).relative_to(root))
    write_json(root/'video_index.json',video_index)
    protocol=json.loads((root/'protocol.json').read_text());protocol['raw_axis_scope']='Separately fitted to each score in each rollout; zero baseline, 5% margin and readable tick rounding'
    write_json(root/'protocol.json',protocol)
    page=root/'index.html';text=page.read_text();note='<p>Raw MP4 y-axis ranges are fitted separately to each score in each rollout, with a small margin, and stay fixed during playback.</p>'
    text=text.replace('Raw versions appear when the checkpoint finishes, using fixed per-score axis limits across all 50 episodes.', 'Raw versions use per-rollout, per-score axis ranges.')
    if note not in text:text=text.replace('</body>',note+'</body>')
    page.write_text(text)
    write_json(status,dict(phase='complete',videos=900,checkpoints=18,axis_scope='rollout',finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'raw_rescale.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:coordinate(a.root)
        except BaseException as exc:
            write_json(a.root/'raw_rescale.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
