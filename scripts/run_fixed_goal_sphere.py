"""Detached, strictly serial fixed-goal hemisphere experiment on one physical GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import psutil

from run_sphere_rollouts import NAMES, KINDS, write, report as rollout_report

REPO = Path(__file__).resolve().parents[1]
EPOCHS = (100, 150, 200, 250, 300)


def prepare_cases(root):
    from vig_dagger.toy_reach_env.sphere_sampling import sample_fixed_goal_pair
    video = root/'video_eval'; static = root/'static_eval'
    with h5py.File(static/'observations.hdf5') as f:records=json.loads(f.attrs['records'])
    cases=[]
    for demo in range(5):
        for radius,kind in [(0.,'midpoint_id'),(.03,'midpoint_3cm'),(.08,'midpoint_8cm')]:
            record=next(r for r in records if r['demo']==demo and r['progress']==.5 and r['direction']==0
                and r['radius_m']==radius and r['kind']==('recorded_id' if radius==0 else 'ood'))
            cases.append(dict(kind=kind,demo=demo,frame=record['frame'],point_a=record['point_a'],
                point_b=record['point_b'],start=record['anchor'] if radius==0 else record['requested_position'],
                reset_seed=20000+demo))
    for demo in range(5):
        r=next(r for r in records if r['demo']==demo)
        cases.append(dict(kind='familiar_a_to_b',demo=demo,point_a=r['point_a'],point_b=r['point_b'],
                          start=r['point_a'],reset_seed=100+demo))
    for seed in range(6000,6005):
        a,b=sample_fixed_goal_pair(seed=seed)
        cases.append(dict(kind='unseen_a_to_b',point_a=a.tolist(),point_b=b.tolist(),
                          start=a.tolist(),reset_seed=seed))
    cases.extend(json.loads((video/'outside_cases.json').read_text()))
    for i,case in enumerate(cases):case.update(case_id=i,policy_seed=81000+i)
    assert len(cases)==30
    assert all(np.allclose(c['point_b'],[.45,0,.5]) for c in cases[:25])
    assert all(np.allclose(c['point_b'],[.45,0,.55]) for c in cases[25:])
    write(video/'cases.json',cases)
    common=dict(task='fixed_goal_sphere',training=str(root/'training_300ep'),checkpoint='epoch-300.pt',
        sphere_center=[.45,0,.35],sphere_radius=.15,fixed_b=[.45,0,.5],
        training_a='Uniform opposite hemisphere; dot(A-C,B-C)<=0',
        image_factor=1,action_horizon=16,execute_steps=8,
        scope='Off-path probes keep the original A marker and fixed B; offsets are from that demonstrated path, not minimum distances to every training trajectory')
    write(video/'protocol.json',dict(common,policies=NAMES,groups=KINDS+['outside_sphere_a_to_b'],
        cases_per_policy=30,total_rollouts=180,trials_per_start=1,
        unseen_a_to_b='New hemisphere A with the same fixed B',
        outside_sphere_a_to_b='Separate goal-OOD stress test: radius .20, B moved outward to (.45,0,.55), A on its opposite hemisphere',
        success='Within 5 mm of B for 15 consecutive steps',max_steps=1200,control_hz=60,
        video_flow='32 newly generated 16-action chunks x 64 fresh noise/time draws, every 8 control steps',
        video_arrows='All 16 cumulative translations for each sample at scale 1; cyan up to 8 executed commands',
        calibration='Fresh per-policy training p95; 20 frames per own training demo; same 32 x 64 recipe',
        video_fps=7.5))
    write(static/'protocol.json',dict(common,
        observations='15 recorded training ID + 15 physically recreated on-path controls + 60 offsets at 3 cm + 60 at 8 cm',
        flow='128 sampled chunks x 128 fresh noise/time draws per observation',
        action_plot='First translation of each of 128 chunks; all 16 commands are preserved and shown in rollout videos',
        examples='Predetermined: first five demos, halfway along the chord, positive first transverse direction'))


def validate_training(out, epoch):
    assert (out/f'epoch-{epoch:03d}.pt').is_file() and (out/'latest.pt').is_file()
    rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    losses=[v for row in rows for key,v in row.items() if key in ('train/loss','val/loss')]
    assert losses and all(math.isfinite(v) for v in losses)
    if epoch==300:
        check=json.loads((out/'initialization_check.json').read_text())
        assert check['weights_equal'] and check['statistics_equal'] and check['selection_equal']


class Queue:
    def __init__(self, root):
        self.root=root
        self.settings=json.loads((root/'experiment.json').read_text())
        self.gpu=self.settings['gpu'];self.uuid=self.settings['gpu_uuid']
        actual=subprocess.check_output(['nvidia-smi',f'--id={self.gpu}','--query-gpu=uuid',
                                        '--format=csv,noheader'],text=True).strip()
        assert actual==self.uuid, 'Physical GPU mapping changed; refusing another GPU'
        setup=root.parent/'setup'
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.uuid,PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
            HF_HOME='/cpfs02/user/caojianuo/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            WANDB_DISABLED='true',OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',
            VK_ICD_FILENAMES=str(setup/'nvidia-570/nvidia_icd_local.json'),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(setup/'nvidia-570/10_nvidia_local.json'))
        self.env['LD_LIBRARY_PATH']=str(setup/'nvidia-graphics')+':'+self.env.get('LD_LIBRARY_PATH','')
        self.jobs=json.loads((root/'jobs.json').read_text()) if (root/'jobs.json').exists() else {}
        self.started=time.time()

    def status(self, stage, **extra):
        write(self.root/'pipeline.json',dict(state='running',stage=stage,pid=os.getpid(),
            gpu=self.gpu,gpu_uuid=self.uuid,concurrency=1,updated_at=time.time(),
            elapsed_seconds=time.time()-self.started,jobs_complete=sum(j['state']=='complete' for j in self.jobs.values()),**extra))

    def alive(self, pid):
        try:
            p=psutil.Process(pid)
            return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and str(self.root) in ' '.join(p.cmdline())
        except psutil.Error:return False

    def job(self, key, command, artifact):
        artifact=Path(artifact)
        previous=self.jobs.get(key,{})
        if previous.get('state')=='complete' and artifact.exists():return
        if previous.get('pid') and self.alive(previous['pid']):
            while self.alive(previous['pid']):
                self.status(key,worker_pid=previous['pid'],adopted=True);time.sleep(10)
            if not artifact.exists():raise RuntimeError(f'Adopted job stopped without {artifact}')
        elif not artifact.exists():
            # One blocking child at a time. No multi-GPU launch helper is called.
            with (self.root/'logs'/f'{key}.log').open('a') as log:
                process=subprocess.Popen(list(map(str,command)),cwd=REPO,env=self.env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.jobs[key]=dict(state='running',pid=process.pid,command=list(map(str,command)),
                                started_at=time.time(),gpu=self.gpu)
            write(self.root/'jobs.json',self.jobs)
            while process.poll() is None:
                self.status(key,worker_pid=process.pid);time.sleep(10)
            if process.returncode:
                self.jobs[key].update(state='failed',returncode=process.returncode)
                write(self.root/'jobs.json',self.jobs)
                raise RuntimeError(f'{key} failed ({process.returncode}); see logs/{key}.log')
            if not artifact.exists():raise RuntimeError(f'{key} exited without {artifact}')
        self.jobs[key]=dict(previous,state='complete',artifact=str(artifact),completed_at=time.time(),gpu=self.gpu)
        write(self.root/'jobs.json',self.jobs)
        print('COMPLETE '+key,flush=True)

    def python(self, script, *args, sim=False):
        return [REPO/('.venv-isaac/bin/python' if sim else '.venv/bin/python'),REPO/'scripts'/script,*args]


def run(root):
    from vig_dagger.toy_training import prepare as prepare_training
    from continue_sphere_training import prepare as prepare_continuation, report as checkpoint_report
    from calibrate_sphere_flow import prepare as prepare_calibration, report as calibration_report
    from run_sphere_policy_videos import validate_video

    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # The GPU lock serializes any restarts of this experiment as well.
        with (root.parent/'toy_gpu7.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            queue=Queue(root)
            train=root/'train_demos';validation=root/'validation_demos'
            first=root/'training_100ep';final=root/'training_300ep'
            static=root/'static_eval';video=root/'video_eval'
            for path in (static,video):path.mkdir(exist_ok=True)
            (video/'logs').mkdir(exist_ok=True)
            # Adopt the collection started interactively, without starting a second simulator.
            if 'collect_train' not in queue.jobs:
                initial=json.loads((root/'collection_process.json').read_text())
                queue.jobs['collect_train']=dict(initial,state='running',gpu=queue.gpu)
                write(root/'jobs.json',queue.jobs)
            for label,destination,count,seed in [('train',train,20,100),('validation',validation,5,1000)]:
                queue.job(f'collect_{label}',queue.python('collect_toy_reach.py','--out',destination,
                    '--gpu',queue.gpu,'--num-demos',count,'--seed',seed,'--task','fixed_goal_sphere',sim=True),
                    destination/'summary.json')
                queue.job(f'validate_{label}',queue.python('validate_toy_reach_dataset.py',destination),destination/'validation.json')
            queue.job('capture_static',queue.python('capture_sphere_simple.py','--root',static,'--source',train,
                      '--gpu',queue.gpu,sim=True),static/'capture_done.json')
            queue.job('camera_outside_preflight',queue.python('prepare_sphere_video_eval.py','--root',video,
                      '--gpu',queue.gpu,'--endpoint-mode','fixed_goal',sim=True),video/'preflight.json')
            prepare_cases(root)
            queue.status('prepare_training')
            if not (first/'experiment.json').exists():
                prepare_training(argparse.Namespace(root=first,source=train/'dataset',
                    validation=validation/'dataset',task='fixed_goal_sphere'))
                manifest=json.loads((first/'experiment.json').read_text())
                for spec in manifest['runs']:
                    spec['gpu']=queue.gpu
                    write(Path(spec['run_dir'])/'experiment.json',spec)
                write(first/'experiment.json',manifest)
            for name in NAMES:
                out=first/'policies'/name
                queue.job(f'train100_{name}',[REPO/'.venv/bin/python','-m','vig_dagger.toy_training',
                    'worker','--run-dir',out],out/'latest.pt')
                validate_training(out,100)
            if not (final/'experiment.json').exists():prepare_continuation(final,first)
            specs=json.loads((final/'experiment.json').read_text())['runs']
            assert all(s['gpu']==queue.gpu for s in specs)
            for spec in specs:
                name=spec['name'];out=final/'policies'/name
                queue.job(f'train300_{name}',[REPO/'.venv/bin/python','-m','vig_dagger.toy_training',
                    'worker','--run-dir',out],out/'latest.pt')
                validate_training(out,300)
                for epoch in EPOCHS:
                    dest=final/'eval'/f'epoch-{epoch:03d}'
                    queue.job(f'check{epoch}_{name}',queue.python('rollout_sphere_simple.py',
                        '--root',dest,'--name',name,'--gpu',queue.gpu,'--policy-device','cuda:0',
                        '--training',final,'--checkpoint',f'epoch-{epoch:03d}.pt',
                        '--case-count',spec['num_demos'],'--source',train/'dataset/toy_reach/shards',sim=True),
                        dest/name/'completed.json')
                checkpoint_report(final,specs)
            # Reuse the original recipes with explicit new data/checkpoint provenance.
            queue.status('prepare_calibration')
            if not (video/'flow_calibration/training_observations.hdf5').exists():
                prepare_calibration(video,train/'dataset/toy_reach')
            protocol=json.loads((video/'flow_calibration/protocol.json').read_text())
            protocol.update(training_root=str(final),checkpoint='epoch-300.pt')
            write(video/'flow_calibration/protocol.json',protocol)
            for name in NAMES:
                queue.job(f'static_{name}',queue.python('analyze_sphere_simple.py','score','--root',static,
                    '--training',final,'--checkpoint','epoch-300.pt','--name',name),static/name/'scores.json')
                queue.job(f'calibrate_{name}',queue.python('calibrate_sphere_flow.py','score','--root',video,
                    '--training',final,'--checkpoint','epoch-300.pt','--name',name,'--gpu',0),
                    video/'flow_calibration'/f'{name}.json')
                queue.job(f'rollout_{name}',queue.python('rollout_sphere_simple.py','--root',video,
                    '--training',final,'--checkpoint','epoch-300.pt','--name',name,'--gpu',queue.gpu,
                    '--policy-device','cuda:0','--source',train/'dataset/toy_reach/shards',sim=True),
                    video/name/'completed.json')
                queue.job(f'annotate_{name}',queue.python('run_sphere_policy_videos.py','annotate','--root',video,
                    '--training',final,'--checkpoint','epoch-300.pt','--name',name,'--gpu',0),
                    video/name/'annotation_done.json')
            queue.job('static_report',queue.python('analyze_sphere_simple.py','report','--root',static,
                      '--training',final,'--checkpoint','epoch-300.pt'),static/'summary.json')
            calibration_report(video);rollout_report(video)
            write(video/'flow_calibration/status.json',dict(state='complete',policies=6,observations=1500,checkpoint='epoch-300.pt'))
            protocol=json.loads((video/'protocol.json').read_text())
            protocol['checkpoint_sha256']={name:hashlib.sha256((final/'policies'/name/'epoch-300.pt').read_bytes()).hexdigest() for name in NAMES}
            write(video/'protocol.json',protocol)
            queue.status('validate_videos')
            paths=sorted(video.glob('demos*/calibrated_chunk_videos/*.mp4'));assert len(paths)==180
            with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(validate_video,paths))
            for name in NAMES:
                cal=json.loads((video/'flow_calibration'/f'{name}.json').read_text())
                with np.load(video/'flow_calibration'/f'{name}.npz') as data:
                    assert np.allclose(data['flow'],data['chunk_losses'].mean(axis=1))
                    assert np.allclose(np.percentile(data['flow'],[50,95]),[cal['median'],cal['p95']])
            write(video/'calibrated_chunk_validation.json',dict(videos=180,all_frames_decoded=True,
                horizon=16,per_policy_training_reference=True,checkpoint='epoch-300.pt',checks=checks))
            write(root/'pipeline.json',dict(state='complete',gpu=queue.gpu,gpu_uuid=queue.uuid,
                concurrency=1,updated_at=time.time(),elapsed_seconds=time.time()-queue.started,
                training_demos=20,validation_demos=5,policies=6,epochs=300,
                checkpoint_rollouts=375,analysis_rollouts=180,calibrated_videos=180,
                static_analysis=str(static),video_analysis=str(video)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.root.resolve())
    except BaseException as exc:
        write(args.root/'pipeline.json',dict(state='failed',pid=os.getpid(),error=repr(exc),updated_at=time.time()))
        traceback.print_exc();raise
