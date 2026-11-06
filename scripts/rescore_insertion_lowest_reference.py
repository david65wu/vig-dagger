"""Rescore recorded actions, without generating any new policy rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from run_insertion_experiment import write_json
from vig_dagger.insertion_analysis import restore,inputs,CAMERAS
from vig_dagger.policy import LibraryPolicy,PolicySet,robot_state,camera_images
from vig_dagger.scoring import VIGScorer

BASE=Path('/cpfs02/user/caojianuo/vig-dagger-runs')
SOURCE=BASE/'charger_insertion_moe7_three_scores_v1'
TRAINING=BASE/'charger_insertion_moe7_v1'
FACTORS=[1,2,4,8,16,32]


def score(context,state,images,seed):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        result=context['scorer'](context['policies'],state,images)
    d=result.details
    gaps=np.array([d['lowest_reference_w1_by_factor'][f] for f in FACTORS])
    assert np.isfinite(gaps).all() and np.all(gaps>=0)
    assert d['lowest_reference_factor']==64
    assert d['lowest_reference_value']==max(gaps)
    # The full-vs-lowest edge must agree in both reference definitions.
    assert np.isclose(gaps[0],d['original_w1_by_factor'][64],rtol=0,atol=1e-8)
    return result.value,d['original_value'],d['lowest_reference_value'],gaps


def context_for(run):
    meta=json.loads((SOURCE/'runs'/run/'calibration.json').read_text())
    path=Path(meta['checkpoint'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    policy=LibraryPolicy(path,'cuda:0')
    assert policy.factors==(1,2,4,8,16,32,64) and policy.cameras==CAMERAS
    return dict(policies=PolicySet({1:policy}),scorer=VIGScorer(32,True,True,True)),meta


def compare(actual,expected,label):
    # Small cross-device floating-point differences are permitted; changes
    # in the observation or scoring protocol must not silently pass.
    if not np.allclose(actual,expected,rtol=1e-4,atol=1e-6):
        raise RuntimeError(f'{label} mismatch: max absolute error {np.max(np.abs(np.asarray(actual)-expected))}')


def replay(root,name,context,case_id,reference=None):
    from vig_dagger.insertion_env import make_insertion_env
    src=SOURCE/'runs'/name;out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    row=next(r for r in json.loads((src/'rollouts.json').read_text()) if r['case_id']==case_id)
    path=out/row['trace'];path.parent.mkdir(exist_ok=True)
    if path.exists():
        assert (out/'replay_audits'/f'{case_id:02d}.json').exists(), 'Missing replay audit'
        if reference is not None:
            with np.load(path) as cached:trace={k:cached[k] for k in cached.files}
            if 'vig_lowest_percentile' not in trace:
                trace['vig_lowest_percentile']=100*np.searchsorted(reference,trace['vig_lowest'],side='right')/len(reference)
                temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**trace);temp.replace(path)
        return
    case=next(c for c in json.loads((SOURCE/'cases.json').read_text()) if c['case_id']==case_id)
    with np.load(src/row['trace']) as saved:trace={k:saved[k] for k in saved.files}
    assert np.array_equal(trace['score_steps'],np.arange(row['steps']))
    env=make_insertion_env(render_backend='cuda:0')
    policy=context['policies'].actor
    values=[];gaps=[];checks=[];tcp_errors=[];tcp_rotation_errors=[];distance_errors=[];angle_errors=[];stable=0;replayed_success=False
    started=time.time()
    try:
        payload=torch.load(SOURCE/case['state_file'],map_location='cpu',weights_only=False)
        obs,_=restore(env,payload)
        compare(env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy(),np.array(row['start_qpos']),'start qpos')
        for step,action in enumerate(trace['actions']):
            mean,original,value,gap=score(context,robot_state(obs,policy.device),camera_images(obs,CAMERAS,policy.device),70000000+case_id*1000+step)
            values.append(value);gaps.append(gap);checks.append([mean,original])
            # Replay only the stored action. Never call the acting policy.
            obs,_,_,truncated,info=env.step(action)
            tcp=env.unwrapped.agent.tcp.pose.raw_pose[0].cpu().numpy()
            error=float(np.linalg.norm(tcp[:3]-trace['tcp'][step,:3]));tcp_errors.append(error)
            tcp_rotation_errors.append(float(np.max(np.abs(tcp[3:]-trace['tcp'][step,3:]))))
            d,a=env.unwrapped._compute_distance()
            distance_errors.append(abs(float(d[0])-trace['distance_m'][step]))
            angle_errors.append(abs(float(a[0])-trace['angle_rad'][step]))
            # Contact impulses can differ by a few micrometres in replay.
            # Use explicit physical tolerances, also recorded in the audit.
            stable=stable+1 if bool(info['success'][0]) and bool(env.unwrapped.agent.is_grasping(env.unwrapped.charger)[0]) else 0
            if stable>=10:replayed_success=True
            if step%50==0:print('REPLAY',name,case_id,step,'/',row['steps'],flush=True)
    finally:env.close()
    values=np.asarray(values);gaps=np.asarray(gaps)
    audit=dict(case_id=case_id,steps=row['steps'],success=row['success'],replayed_success=replayed_success,
        max_tcp_error_m=max(tcp_errors),max_tcp_quaternion_error=max(tcp_rotation_errors),
        max_plug_distance_error_m=max(distance_errors),max_plug_angle_error_rad=max(angle_errors),
        max_original_vig_error=float(np.max(np.abs(np.asarray(checks)[:,1]-trace['vig_original']))),
        max_chunk_mean_vig_error=float(np.max(np.abs(np.asarray(checks)[:,0]-trace['vig']))),
        source_trace_sha256=hashlib.sha256((src/row['trace']).read_bytes()).hexdigest(),
        source=str(src/row['trace']),no_new_policy_actions=True,seconds=time.time()-started)
    write_json(out/'replay_diagnostics'/f'{case_id:02d}.json',audit)
    compare(np.asarray(checks)[:,0],trace['vig'],'whole-episode chunk mean VIG')
    compare(np.asarray(checks)[:,1],trace['vig_original'],'whole-episode original VIG')
    assert max(tcp_errors)<=1e-5 and max(tcp_rotation_errors)<=1e-5, audit
    assert max(distance_errors)<=1e-5 and max(angle_errors)<=1e-3, audit
    assert replayed_success==row['success'],audit
    trace.update(vig_lowest=values,vig_lowest_gap=gaps,vig_lowest_factors=FACTORS,
                 replay_vig=np.asarray(checks)[:,0],replay_vig_original=np.asarray(checks)[:,1])
    if reference is not None:trace['vig_lowest_percentile']=100*np.searchsorted(reference,values,side='right')/len(reference)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**trace);temp.replace(path)
    write_json(out/'replay_audits'/f'{case_id:02d}.json',audit)
    print('REPLAY VERIFIED',audit,flush=True)


def calibrate(root,name,context,meta,preview=False):
    src=SOURCE/'runs'/name;out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    target=out/('calibration_preview.npz' if preview else 'calibration_vig_lowest.npz')
    if target.exists() and target.with_suffix('.json').exists():return np.sort(np.load(target)['vig'])
    count=meta['training_demo_count']
    order=json.loads((TRAINING/'demo_order.json').read_text())[:count]
    old=np.load(src/'calibration_vig_original.npz');old_mean=np.load(src/'calibration_vig.npz')
    indices=[];values=[];gaps=[];original=[];means=[];done=0;offset=0
    partial=out/'calibration_lowest_partial.npz'
    if partial.exists() and not preview:
        with np.load(partial) as p:
            indices=p['indices'].tolist();values=p['vig'].tolist();gaps=p['gaps'].tolist()
            original=p['original'].tolist();means=p['means'].tolist();done=int(p['demos_done']);offset=len(indices)
    started=time.time()
    for i,item in enumerate(order):
        if i<done:continue
        with h5py.File(Path(item['source']).with_suffix('.h5')) as h5:
            g=h5[f'traj_{item["episode_id"]}']
            for frame in range(min(16,len(g['actions'])) if preview else len(g['actions'])):
                state,images=inputs(context['policies'].actor,g['obs/agent/qpos'][frame],np.stack([g[f'obs/sensor_data/{c}/rgb'][frame] for c in CAMERAS]))
                mean,orig,value,gap=score(context,state,images,80000000+item['source_index']*1000+frame)
                assert np.array_equal(old['indices'][offset],[item['source_index'],frame])
                compare(orig,old['vig'][offset],'training original VIG')
                compare(mean,old_mean['vig'][offset],'training chunk mean VIG')
                indices.append([item['source_index'],frame]);values.append(value);gaps.append(gap);original.append(orig);means.append(mean);offset+=1
        print('CALIBRATE LOWEST',name,i+1,'/',count,len(values),'frames',flush=True)
        if preview:break
        temp=partial.with_suffix('.tmp.npz');np.savez_compressed(temp,indices=indices,vig=values,gaps=gaps,original=original,means=means,demos_done=i+1);temp.replace(partial)
        write_json(out/'status.json',dict(phase='calibrating',demos_done=i+1,total_demos=count,frames=len(values),seconds=time.time()-started))
    if not preview:assert np.array_equal(indices,old['indices']) and np.array_equal(indices,old_mean['indices'])
    temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,indices=indices,vig=values,vig_gap=gaps,reference_factor=64,comparison_factors=FACTORS,frame_stride=1);temp.replace(target)
    metadata={k:meta[k] for k in ['checkpoint','sha256','training_demo_count','variant','stride']}
    metadata.update(reference_factor=64,comparison_factors=FACTORS,num_samples=32,temporal_average=False,preview_only=preview,
        scores=len(values),percentiles={str(p):float(np.percentile(values,p)) for p in [0,25,50,75,95,100]},
        max_original_vig_error=float(np.max(np.abs(np.asarray(original)-old['vig'][:len(original)]))),
        max_chunk_mean_vig_error=float(np.max(np.abs(np.asarray(means)-old_mean['vig'][:len(means)]))),seconds=time.time()-started)
    write_json(target.with_suffix('.json'),metadata)
    return np.sort(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--preview',action='store_true');p.add_argument('--calibration-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    context,meta=context_for(a.name)
    if a.preview:
        calibrate(a.root,a.name,context,meta,preview=True)
        replay(a.root,a.name,context,22)
        write_json(a.root/'preflight.json',dict(passed=True,checkpoint=a.name,case_id=22,full_episode_replayed=True,calibration_frames_verified=16,finished_at=time.time()))
    else:
        reference=calibrate(a.root,a.name,context,meta)
        write_json(a.root/'runs'/a.name/'calibration_complete.json',dict(scores=len(reference),finished_at=time.time()))
        if a.calibration_only:return
        for row in json.loads((SOURCE/'runs'/a.name/'rollouts.json').read_text()):
            replay(a.root,a.name,context,row['case_id'],reference)
        write_json(a.root/'runs'/a.name/'scoring_complete.json',dict(episodes=50,finished_at=time.time()))

if __name__=='__main__':main()
