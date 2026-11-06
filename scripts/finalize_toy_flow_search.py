"""Detached final report and artifact checks after all six search workers finish."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

import h5py
import numpy as np


def finish(root):
    repo=Path(__file__).resolve().parents[1]
    names=[f'demos{n}_{v}' for n in (5,20) for v in ('dense_full','moe_7level','dense_7level')]
    while True:
        status={name:json.loads((root/name/'pipeline.json').read_text()) for name in names}
        failed=[name for name,data in status.items() if data['state']=='failed']
        if failed:
            raise RuntimeError(f'Failed workers: {failed}')
        count=sum(data['state']=='complete' for data in status.values())
        (root/'status.json').write_text(json.dumps(dict(state='running',completed_policies=count,total_policies=6)))
        if count==6:
            break
        time.sleep(15)
    subprocess.run([str(repo/'.venv/bin/python'),str(repo/'scripts/analyze_toy_flow_search.py'),
                    'report','--root',str(root)],cwd=repo,check=True)
    checks=dict(policies={},total_rollouts=0,decoded_videos=0)
    for name in names:
        out=root/name
        data=json.loads((out/'scores.json').read_text())
        assert len(data['results'])==169
        with np.load(out/'action_samples.npz') as arrays:
            assert arrays['chunks'].shape==(169,128,16,7)
            assert arrays['losses'].shape==(169,128)
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
        rollouts=json.loads((out/'rollouts.json').read_text())
        assert len(rollouts)==96
        assert len({(r['index'],r['trial']) for r in rollouts})==96
        for row in rollouts:
            with np.load(out/f"state_{row['index']:03d}_trial_{row['trial']:02d}.npz") as arrays:
                assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
                assert np.array_equal(arrays['actions'][:min(8,len(arrays['actions']))],
                    arrays['first_chunk'][:min(8,len(arrays['actions']))])
        checks['policies'][name]=dict(observations=169,rollouts=96,all_finite=True,
            max_restore_state_error=max(r['restore_max_state_error'] for r in rollouts),
            max_initial_rgb_mae=max(r['initial_rgb_mae'] for r in rollouts))
        checks['total_rollouts']+=len(rollouts)
        videos=list((out/'videos').glob('*.mp4'));assert len(videos)==12
        for video in videos:
            subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],
                check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            checks['decoded_videos']+=1
    with h5py.File(root/'observations.hdf5') as f:
        records=json.loads(f.attrs['records'])
        checks['max_probe_placement_error_m']=max(r['placement_error'] for r in records if r['kind']=='probe')
    (root/'validation.json').write_text(json.dumps(checks,indent=2))
    (root/'status.json').write_text(json.dumps(dict(state='complete',completed_policies=6,total_rollouts=576)))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    try:
        finish(args.root)
    except BaseException as exc:
        (args.root/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc))))
        traceback.print_exc();raise
