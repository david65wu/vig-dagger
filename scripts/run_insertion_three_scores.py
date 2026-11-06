"""One rollout and MP4 containing flow plus both VIG definitions every frame."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time

from run_insertion_experiment import REPO,run_job,write_json,alive
from run_insertion_mean_chunk_vig import BASE,OLD,TRAINING,EPOCHS,prepare as prepare_base
MEAN=BASE/'charger_insertion_moe7_mean_chunk_vig_v1'


def prepare(root):
    prepare_base(root)
    protocol=json.loads((root/'protocol.json').read_text())
    protocol.update(vig_both=True,policy_label='7-level MoE | flow + original VIG + chunk-mean VIG',
        vig_sampling='Both VIG definitions computed from the identical 7 x 32 x 16 x 8 sample tensor at each frame',
        calibration='Reuse each score own every-frame training reference after checkpoint hash and calibration-frame verification',
        video_panels=['flow','original VIG','chunk-mean VIG'],video_size=[512,840],
        video_versions=['percentile axes','raw axes'],raw_axis_scope='Fixed per score across all 50 episodes of a checkpoint')
    write_json(root/'protocol.json',protocol)
    import numpy as np
    for count in [50,200,500]:
        for epoch in EPOCHS:
            name=f'demos{count}_epoch{epoch:03d}';dest=root/'runs'/name
            sha=json.loads((dest/'calibration.json').read_text())['sha256']
            for source,suffix in [(MEAN,''),(OLD,'_original')]:
                source=source/'runs'/name;meta=json.loads((source/'calibration_vig.json').read_text())
                assert meta['sha256']==sha and meta['stride']==1
                assert meta.get('temporal_average',False)==(suffix=='')
                for ext in ['npz','json']:
                    target=dest/f'calibration_vig{suffix}.{ext}'
                    if not target.exists():shutil.copy2(source/f'calibration_vig.{ext}',target)
            arrays=[np.load(dest/name) for name in ['calibration.npz','calibration_vig.npz','calibration_vig_original.npz']]
            assert all(np.array_equal(arrays[0]['indices'],a['indices']) for a in arrays[1:])
    if not (root/'preflight.json').exists():
        links=''.join(f'<li><a href="runs/demos{n}_epoch{e:03d}/index.html">{n} demos / epoch {e}</a></li>' for n in [50,200,500] for e in EPOCHS)
        (root/'index.html').write_text('<html><meta charset="utf-8"><body><h1>Flow, VIG, and chunk-mean VIG together</h1><p>All three scores on every frame of each MP4; separate calibrated percentile plots. 50 videos per checkpoint, 900 total.</p><ul>'+links+'</ul></body></html>')


def preflight(root):
    import numpy as np
    import torch
    from vig_dagger.scoring import VIGScorer
    from vig_dagger.insertion_analysis import worker
    class Fake:
        device=torch.device('cpu');horizon=16;action_dim=8;factors=(1,2);calls=0
        def vig_samples(self,*args):
            self.calls+=1
            result=torch.zeros(2,2,16,8);result[1,:,:8]=1;result[1,:,8:]=-1
            return result
    fake=Fake();score=VIGScorer(2,True,True)(fake,torch.zeros(1,9),torch.zeros(1,2,3,2,2))
    assert fake.calls==1 and score.value==0 and score.details['original_value']==1
    worker(root,TRAINING,50,300,case_ids=[0])
    out=root/'runs/demos50_epoch300';row=json.loads((out/'rollouts.json').read_text())[0];trace=np.load(out/row['trace'])
    assert row['vig_both'] and len(trace['score_steps'])==row['steps']
    assert np.array_equal(trace['score_steps'],np.arange(row['steps']))
    for raw,percentile,reference in [('flow','percentile','calibration.npz'),('vig','vig_percentile','calibration_vig.npz'),('vig_original','vig_original_percentile','calibration_vig_original.npz')]:
        calibration=np.load(out/reference);values=np.sort(calibration['flow' if raw=='flow' else 'vig'])
        assert len(trace[raw])==row['steps']
        assert np.allclose(trace[percentile],100*np.searchsorted(values,trace[raw],side='right')/len(values))
    assert np.allclose(trace['vig'],trace['vig_gap'].max(axis=1))
    assert np.allclose(trace['vig_original'],trace['vig_original_gap'].max(axis=1))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames','-of','json',str(out/row['video'])]))['streams'][0]
    assert (probe['width'],probe['height'],int(probe['nb_frames']))==(512,840,row['steps'])
    write_json(root/'preflight.json',dict(passed=True,single_vig_sampling_call_verified=True,three_score_alignment_verified=True,
        all_percentiles_verified=True,video_dimensions=[512,840],frames=row['steps'],video=str((out/row['video']).relative_to(root)),finished_at=time.time()))


def coordinate(root):
    prepare(root);write_json(root/'pipeline.json',dict(phase='preflight',updated_at=time.time()))
    if not (root/'preflight.json').exists():run_job([sys.executable,'-u','scripts/run_insertion_three_scores.py','--root',str(root),'--preflight'],root/'preflight.log',0,root/'preflight_process.json','--preflight')
    assert json.loads((root/'preflight.json').read_text())['passed']
    jobs=queue.Queue();adopt={}
    for n in [500,200,50]:
        for e in EPOCHS:
            registry=root/'processes'/f'demos{n}_epoch{e:03d}.json'
            process=json.loads(registry.read_text()) if registry.exists() else {}
            if process and alive(process['pid'],str(root)):
                assert process['gpu'] not in adopt
                adopt[process['gpu']]=(n,e)
            else:jobs.put((n,e))
    write_json(root/'pipeline.json',dict(phase='evaluating',pid=__import__('os').getpid(),gpus=list(range(8)),checkpoints=18,rollouts=900,updated_at=time.time()))
    def lane(gpu):
        initial=adopt.pop(gpu,None)
        while True:
            if initial is not None:n,e=initial;initial=None
            else:
                try:n,e=jobs.get_nowait()
                except queue.Empty:return
            name=f'demos{n}_epoch{e:03d}';out=root/'runs'/name
            if not (out/'complete.json').exists():
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','worker','--root',str(root),'--training',str(TRAINING),'--count',str(n),'--epoch',str(e)],root/'logs'/f'{name}.log',gpu,root/'processes'/f'{name}.json',f'--root {root} --training {TRAINING} --count {n} --epoch {e}')
            assert (out/'complete.json').exists()
            if not (out/'raw_complete.json').exists():
                with (root/'logs'/f'{name}_raw.log').open('a') as log:
                    subprocess.run([sys.executable,'-u','scripts/render_insertion_raw_videos.py','--run-dir',str(out)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert (out/'raw_complete.json').exists()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,g) for g in range(8)]
        for f in futures:f.result()
    write_json(root/'pipeline.json',dict(phase='plotting',updated_at=time.time()))
    subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(root)],cwd=REPO,check=True)
    subprocess.run([sys.executable,'scripts/compare_insertion_variants.py','--dense',str(BASE/'charger_insertion_dense_every_frame_v1'),'--moe',str(root)],cwd=REPO,check=True)
    write_json(root/'pipeline.json',dict(phase='complete',checkpoints=18,rollouts=900,mp4s=1800,finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.preflight:preflight(a.root)
    else:
        with (a.root/'coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:coordinate(a.root)
            except BaseException as exc:
                write_json(a.root/'pipeline.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
