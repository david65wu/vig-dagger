"""Reuse initial sphere rollouts, add outside-sphere cases, annotate every video."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_sphere_rollouts import NAMES,write,report


def validate_video(path):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_frames','-of','json',str(path)],capture_output=True,text=True,check=True)
    stream=json.loads(result.stdout)['streams'][0]
    assert stream['width']==1024 and stream['height']==800 and int(stream['nb_frames'])>8,(path,stream)
    return dict(path=str(path),frames=int(stream['nb_frames']))


def run(root):
    repo=Path(__file__).resolve().parents[1];source=root.parent/'toy_sphere_rollouts_v1';setup=root.parent/'setup'
    started=time.time();write(root/'pipeline.json',dict(state='waiting_for_preflight',pid=os.getpid()))
    while not (root/'preflight.json').exists():
        pid=json.loads((root/'preflight_process.json').read_text())['pid']
        if not Path(f'/proc/{pid}').exists():raise RuntimeError('Preflight failed; see logs/preflight.log')
        time.sleep(5)
    cases=json.loads((source/'cases.json').read_text());outside=json.loads((root/'outside_cases.json').read_text())
    for i,case in enumerate(outside):cases.append(dict(case,case_id=25+i,policy_seed=81025+i))
    write(root/'cases.json',cases)
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(cases_per_policy=30,total_rollouts=180,groups=protocol['groups']+['outside_sphere_a_to_b'],
        reused_rollouts=str(source),outside_endpoints='Both A/B on 20 cm sphere, 5 cm outside the training sphere; both reachability-checked',
        video_flow='Recomputed from exact recorded states and decoded H264 camera images; 32 generated chunks, 64 noise/time draws per chunk, every 8 control steps',
        video_arrows='Orange sampled first XYZ commands; cyan actual executed XYZ command; world command lengths magnified 5x before camera projection',
        video_cameras='Calibrated intrinsics and ROS extrinsics; wrist rigid transform reconstructed from recorded TCP pose',
        video_fps=7.5)
    write(root/'protocol.json',protocol)
    (root/'source_snapshot').mkdir(exist_ok=True)
    for rel in ['scripts/rollout_sphere_simple.py','scripts/extend_sphere_rollouts.py','scripts/annotate_sphere_rollouts.py',
                'scripts/prepare_sphere_video_eval.py','scripts/run_sphere_rollouts.py','src/vig_dagger/toy_projection.py']:
        shutil.copy2(repo/rel,root/'source_snapshot'/Path(rel).name)
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED='0',OMP_NUM_THREADS='2',VK_ICD_FILENAMES=str(setup/'nvidia-570/nvidia_icd_local.json'),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(setup/'nvidia-570/10_nvidia_local.json'))
    env['LD_LIBRARY_PATH']=str(setup/'nvidia-graphics')+':'+env.get('LD_LIBRARY_PATH','');env.pop('CUDA_VISIBLE_DEVICES',None)
    stages={name:'waiting_for_original_rollouts' for name in NAMES};processes={}
    def launch(name,stage,cmd):
        with (root/'logs'/f'{name}_{stage}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;stages[name]=stage;print(f'{stage} {name} pid={p.pid}',flush=True)
    while True:
        for gpu,name in enumerate(NAMES):
            out=root/name
            if stages[name]=='waiting_for_original_rollouts':
                status=source/name/'status.json'
                if not status.exists() or json.loads(status.read_text())['state']!='complete':continue
                out.mkdir(exist_ok=True);(out/'videos').mkdir(exist_ok=True)
                for p in (source/name).glob('*.npz'):
                    dest=out/p.name
                    if not dest.exists():dest.symlink_to(p)
                for p in (source/name/'videos').glob('*.mp4'):
                    dest=out/'videos'/p.name
                    if not dest.exists():dest.symlink_to(p)
                shutil.copy2(source/name/'rollouts.json',out/'rollouts.json')
                launch(name,'outside_rollouts',[str(repo/'.venv-isaac/bin/python'),str(repo/'scripts/rollout_sphere_simple.py'),
                    '--root',str(root),'--name',name,'--gpu',str(gpu)])
            elif stages[name] in ('outside_rollouts','annotating'):
                p=processes[name]
                if p.poll() is None:continue
                if p.returncode:raise RuntimeError(f'{name} {stages[name]} failed, code {p.returncode}')
                if stages[name]=='outside_rollouts':
                    launch(name,'annotating',[sys.executable,str(repo/'scripts/annotate_sphere_rollouts.py'),
                        '--root',str(root),'--name',name,'--gpu',str(gpu)])
                else:stages[name]='complete'
        write(root/'pipeline.json',dict(state='running',pid=os.getpid(),elapsed_seconds=time.time()-started,
                                       stages=stages,pids={n:p.pid for n,p in processes.items()}))
        if all(s=='complete' for s in stages.values()):break
        time.sleep(5)
    report(root)
    videos=sorted(root.glob('demos*/annotated_videos/*.mp4'));assert len(videos)==180
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(validate_video,videos))
    write(root/'video_validation.json',dict(videos=len(checks),checks=checks))
    write(root/'pipeline.json',dict(state='complete',rollouts=180,annotated_videos=180,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:run(args.root)
    except BaseException as exc:
        write(args.root/'pipeline.json',dict(state='failed',error=repr(exc)));raise
