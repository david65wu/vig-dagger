"""Decode every completed annotated video and verify frame/flow alignment."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def check(path):
    result=subprocess.run(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],
        capture_output=True,text=True,check=True)
    assert not result.stderr.strip(),(path,result.stderr)
    stream=json.loads(result.stdout)['streams'][0]
    arrays=np.load(path.with_name(path.stem+'_flow_actions.npz'))
    assert stream['width']==1024 and stream['height']==800 and stream['r_frame_rate']=='15/2'
    assert int(stream['nb_read_frames'])==len(arrays['steps'])+8
    assert np.all(np.diff(arrays['steps'])==8) and np.isfinite(arrays['flow']).all()
    assert arrays['sampled_first_xyz'].shape==(len(arrays['steps']),32,3)
    return dict(path=str(path),decoded_frames=int(stream['nb_read_frames']),flow_frames=len(arrays['steps']))


def run(root):
    submitted=set();pending={};results=[];start=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        while len(results)<180:
            for meta in root.glob('demos*/annotated_videos/*.json'):
                path=meta.with_suffix('.mp4')
                if path not in submitted:
                    pending[pool.submit(check,path)]=path;submitted.add(path)
            for future in list(pending):
                if future.done():
                    results.append(future.result());del pending[future]
            (root/'decode_progress.json').write_text(json.dumps(dict(decoded=len(results),total=180,pending=len(pending))))
            if time.time()-start>3600:raise TimeoutError('Video validation exceeded one hour')
            time.sleep(2)
    (root/'full_decode_validation.json').write_text(json.dumps(dict(videos=180,all_frames_decoded=True,results=results),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
