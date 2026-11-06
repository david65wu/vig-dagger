"""Matched insertion ID/OOD rollouts, checkpoint-specific flow calibration and plots."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from vig_dagger.insertion_training import CAMERAS, CHECKPOINT_EPOCHS, write_json

KINDS=['familiar_id','new_id','position_ood','orientation_ood','combined_ood']
LABELS=['Familiar ID','New ID','Position OOD','Orientation OOD','Combined OOD']
COLORS=['#2474b5','#21a585','#ed9b30','#9b64bb','#d94a56']


def cpu_tree(value):
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    return torch.as_tensor(value).detach().cpu().clone()


def h5_state(group):
    return {k:h5_state(v) if isinstance(v,h5py.Group) else torch.from_numpy(v[0:1]) for k,v in group.items()}


def clock_reset(env):
    import gymnasium as gym
    env.unwrapped._elapsed_steps.zero_()
    cursor=env
    while isinstance(cursor,gym.Wrapper):
        if '_elapsed_steps' in cursor.__dict__:cursor._elapsed_steps=0
        cursor=cursor.env


def restore(env,payload):
    from mani_skill.utils.structs.pose import Pose
    env.env.reset(seed=payload['seed'])
    base=env.unwrapped
    base.set_state_dict(copy.deepcopy(payload['state']))
    base.goal_pose=Pose.create(torch.tensor(payload['goal_pose'])[None])
    base.agent.controller.reset()
    q=base.agent.robot.get_qpos()[0,:7].cpu().numpy()
    # One physics substep rebuilds contact impulses after state restore.
    # This is identical across policies and excluded from scored rollout steps.
    base.agent.set_action(torch.tensor(np.r_[q,-1.],dtype=torch.float32)[None])
    base.agent.before_simulation_step()
    base.scene.step()
    obs=base.get_obs()
    drift=float(np.linalg.norm(base.charger.pose.raw_pose[0,:3].cpu().numpy()-np.asarray(payload['charger_pose'])[:3]))
    if drift>.001:raise RuntimeError(f'State restoration moved charger by {drift} m')
    if not bool(base.agent.is_grasping(base.charger)[0]):raise RuntimeError('Restored start is not grasped')
    if bool(base.evaluate()['success'][0]):raise RuntimeError('Restored start already successful')
    clock_reset(env)
    return obs,drift


class FK:
    def __init__(self,base):
        self.base=base
        art=base.agent.robot._objs[0]
        self.model=art.create_pinocchio_model()
        links=art.get_links()
        # Match the physical link object; ManiSkill's logical names omit the
        # environment prefixes attached to SAPIEN entity names.
        target=base.agent.tcp._objs[0]
        self.index=next(i for i,l in enumerate(links) if l==target)
    def poses(self,actions):
        actions=np.asarray(actions);flat=actions.reshape(-1,actions.shape[-1])
        current=self.base.agent.robot.get_qpos()[0].cpu().numpy()
        root=self.base.agent.robot.pose.sp
        result=[]
        for action in flat:
            q=current.copy();q[:7]=action[:7]
            self.model.compute_forward_kinematics(q)
            pose=root*self.model.get_link_pose(self.index)
            result.append(np.r_[pose.p,pose.q])
        return np.asarray(result).reshape(*actions.shape[:-1],7)
    def check(self):
        q=self.base.agent.robot.get_qpos()[0].cpu().numpy()
        predicted=self.poses(np.r_[q[:7],-1.][None])[0,:3]
        error=float(np.linalg.norm(predicted-self.base.agent.tcp.pose.sp.p))
        if error>1e-5:raise RuntimeError(f'FK mismatch {error} m')
        return error


def save_case(env,root,row,payload=None):
    base=env.unwrapped
    if payload is None:
        payload=dict(seed=row['seed'],state=cpu_tree(base.get_state_dict()),
            goal_pose=base.goal_pose.raw_pose[0].cpu().tolist(),
            charger_pose=base.charger.pose.raw_pose[0].cpu().tolist())
    obs,drift=restore(env,payload)
    fk=FK(base);error=fk.check()
    row.update(restore_drift_m=drift,fk_error_m=error,
        state_file=f'states/{row["case_id"]:02d}_{row["kind"]}.pt')
    path=root/row['state_file'];path.parent.mkdir(exist_ok=True)
    torch.save(payload,path)
    # Preview and distribution samples use the same restoration protocol.
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(512,280),(20,24,30))
    for i,c in enumerate(CAMERAS):im.paste(Image.fromarray(obs['sensor_data'][c]['rgb'][0].cpu().numpy()),(i*256,24))
    ImageDraw.Draw(im).text((5,6),f'{row["case_id"]}: {row["kind"]} seed {row["seed"]}',fill='white')
    (root/'start_images').mkdir(exist_ok=True);im.save(root/'start_images'/f'{row["case_id"]:02d}.png')
    return row


def prepare(root,training):
    from vig_dagger.insertion_env import make_insertion_env
    if (root/'cases.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    env=make_insertion_env(render_backend='cuda:0')
    order=json.loads((training/'demo_order.json').read_text());rows=[];rejected=[]
    try:
        for i,item in enumerate(order[:10]):
            raw=Path(item['source']).with_suffix('.h5')
            with h5py.File(raw) as h5:
                g=h5[f'traj_{item["episode_id"]}']
                payload=dict(seed=item['seed'],state=h5_state(g['env_states']),
                    goal_pose=g['task_state/goal_pose'][:].tolist(),charger_pose=g['task_state/charger_pose'][0].tolist())
            rows.append(save_case(env,root,dict(case_id=i,kind='familiar_id',pair=i,seed=item['seed'],
                source=item['source'],source_episode=item['episode_id'],initialization=item['initialization']),payload))
        accepted=0
        for candidate in range(100):
            if accepted==10:break
            seed=200000+candidate
            rng=np.random.default_rng(900000+seed)
            d=float(rng.uniform(.03,.08));yz=rng.uniform(-.015,.015,2);rpy=rng.uniform(-8,8,3)
            ood_rng=np.random.default_rng(700000+seed)
            shifted=yz.copy();shifted[0]=float(ood_rng.choice([-1,1])*ood_rng.uniform(.03,.05))
            rotated=rpy.copy();axis=1+(accepted%2);rotated[axis]=float(ood_rng.choice([-1,1])*ood_rng.uniform(15,25))
            staged=[]
            try:
                for k,lat,ang in [('new_id',yz,rpy),('position_ood',shifted,rpy),('orientation_ood',yz,rotated),('combined_ood',shifted,rotated)]:
                    requested=dict(standoff_m=d,lateral_yz_m=lat.tolist(),rpy_deg=ang.tolist())
                    _,info=env.reset(seed=seed,options={'insertion_pose':requested})
                    init=info['insertion_initialization']
                    if 'position' in k or k=='combined_ood':
                        assert abs(init['actual_goal_relative_xyz_m'][1])>.015
                    if 'orientation' in k or k=='combined_ood':
                        assert abs(init['actual_goal_relative_rpy_deg'][axis])>8
                    row=dict(case_id=KINDS.index(k)*10+accepted,kind=k,pair=accepted,seed=seed,
                        requested=requested,initialization=init)
                    staged.append(save_case(env,root,row))
                rows.extend(staged);accepted+=1
                print(f'PREPARED matched group {accepted}/10',flush=True)
            except Exception as exc:
                rejected.append(dict(seed=seed,error=repr(exc)))
                print(f'REJECTED setup {seed}: {exc}',flush=True)
            write_json(root/'rejected_setups.json',rejected)
        if accepted!=10:raise RuntimeError('Could not construct ten valid matched ID/OOD groups')
    finally:env.close()
    rows.sort(key=lambda x:x['case_id']);write_json(root/'cases.json',rows)
    write_json(root/'protocol.json',dict(training_root=str(training),checkpoints=CHECKPOINT_EPOCHS,demo_counts=[50,200,500],
        cases=KINDS,rollouts_per_case=10,total_rollouts=900,cameras=CAMERAS,
        score_stride=8,executed_actions_per_replan=8,horizon=16,loss_draws=512,action_distribution_samples=128,
        max_steps=300,success='10 consecutive steps meeting original goal tolerance and maintaining grasp',
        calibration='All raw recorded observations before actions, stride 8, in each checkpoint own training subset',
        percentile='100 * searchsorted(sorted_training_scores, score, side=right) / number_training_scores',
        state_restore='Recorded physical state + one fixed-target physics substep to rebuild contacts; excluded from scoring; drift <= 1 mm',
        distribution_projection='First joint target converted to TCP displacement by FK, projected on socket insertion/lateral plane',
        videos='First success and first failure per case and checkpoint, when available; all 16 commanded TCP targets projected with current camera extrinsics',
        seed=20261103))


def inputs(policy,state,images):
    return torch.as_tensor(state,dtype=torch.float32,device=policy.device).reshape(1,9),torch.as_tensor(images,device=policy.device).permute(0,3,1,2)[None]


def calibrate(policy,out,training,count,stride=8):
    target=out/'calibration.npz'
    if target.exists():
        cached=np.load(target);assert int(cached.get('frame_stride',8))==stride
        return cached['flow']
    order=json.loads((training/'demo_order.json').read_text())[:count]
    values=[];indices=[]
    partial=out/'calibration_partial.npz'
    done=0
    if partial.exists():
        saved=np.load(partial);assert int(saved.get('frame_stride',8))==stride
        values=saved['flow'].tolist();indices=saved['indices'].tolist();done=int(saved['demos_done'])
    started=time.time()
    for i,item in enumerate(order):
        if i<done:continue
        with h5py.File(Path(item['source']).with_suffix('.h5')) as h5:
            g=h5[f'traj_{item["episode_id"]}']
            for frame in range(0,len(g['actions']),stride):
                state=g['obs/agent/qpos'][frame];images=np.stack([g[f'obs/sensor_data/{c}/rgb'][frame] for c in CAMERAS])
                s,im=inputs(policy,state,images)
                torch.manual_seed(40000000+item['source_index']*1000+frame)
                loss,_=policy.flow_score(s,im,num_loss_samples=512)
                if not math.isfinite(loss):raise RuntimeError('Nonfinite calibration score')
                values.append(loss);indices.append([item['source_index'],frame])
        if (i+1)%10==0 or i+1==count:
            np.savez_compressed(partial,flow=np.asarray(values),indices=np.asarray(indices),demos_done=i+1,frame_stride=stride)
            write_json(out/'status.json',dict(phase='calibrating',demos_done=i+1,total_demos=count,scores=len(values),elapsed=time.time()-started))
            print(f'CALIBRATE {i+1}/{count}, {len(values)} scores',flush=True)
    np.savez_compressed(target,flow=np.asarray(values),indices=np.asarray(indices),frame_stride=stride)
    return np.asarray(values)


def project(points,params):
    points=np.asarray(points);ext=np.asarray(params['extrinsic_cv']);K=np.asarray(params['intrinsic_cv'])
    xyz=np.c_[points,np.ones(len(points))]@ext.T
    pixel=xyz@K.T
    return pixel[:,:2]/np.maximum(pixel[:,2:3],1e-8),xyz[:,2]>1e-5


def score_vig(context,state,images,seed):
    policies=context['policies']
    devices=[policies.device.index or 0] if policies.device.type=='cuda' else []
    verify_rng=not context.get('rng_preservation_verified',False)
    if verify_rng:
        cpu_before=torch.get_rng_state();gpu_before=[torch.cuda.get_rng_state(d) for d in devices]
    # Diagnostic sampling must not change the random stream used for actions.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        result=context['scorer'](policies,state,images)
    if verify_rng:
        assert torch.equal(cpu_before,torch.get_rng_state())
        assert all(torch.equal(before,torch.cuda.get_rng_state(d)) for before,d in zip(gpu_before,devices))
        context['rng_preservation_verified']=True
    gaps=[result.details['w1_by_factor'][f] for f in policies.factors[1:]]
    if not math.isfinite(result.value) or not np.isfinite(gaps).all():raise RuntimeError('Nonfinite VIG')
    if 'original_value' in result.details:
        context['last_original_value']=result.details['original_value']
        context['last_original_gaps']=[result.details['original_w1_by_factor'][f] for f in policies.factors[1:]]
        assert math.isfinite(context['last_original_value'])
    if 'lowest_reference_value' in result.details:
        context['last_lowest_value']=result.details['lowest_reference_value']
        context['last_lowest_gaps']=[result.details['lowest_reference_w1_by_factor'][f] for f in policies.factors[:-1]]
        assert math.isfinite(context['last_lowest_value'])
    if 'samples' in result.details:
        context['last_samples']=result.details['samples'].cpu().numpy()
    return result.value,gaps


def calibrate_vig(context,out,training,count,stride=8):
    target=out/'calibration_vig.npz'
    if target.exists():
        cached=np.load(target);assert int(cached.get('frame_stride',8))==stride
        assert bool(cached.get('temporal_average',False))==context['scorer'].temporal_average
        return cached['vig']
    partial=out/'calibration_vig_partial.npz';values=[];indices=[];gaps=[];done=0;started=time.time()
    if partial.exists():
        saved=np.load(partial);assert int(saved.get('frame_stride',8))==stride
        assert bool(saved.get('temporal_average',False))==context['scorer'].temporal_average
        values=saved['vig'].tolist();indices=saved['indices'].tolist()
        gaps=saved['vig_gap'].tolist();done=int(saved['demos_done'])
    order=json.loads((training/'demo_order.json').read_text())[:count]
    for i,item in enumerate(order):
        if i<done:continue
        with h5py.File(Path(item['source']).with_suffix('.h5')) as h5:
            g=h5[f'traj_{item["episode_id"]}']
            for frame in range(0,len(g['actions']),stride):
                state=g['obs/agent/qpos'][frame];images=np.stack([g[f'obs/sensor_data/{c}/rgb'][frame] for c in CAMERAS])
                s,im=inputs(context['policies'].actor,state,images)
                value,gap=score_vig(context,s,im,80000000+item['source_index']*1000+frame)
                values.append(value);gaps.append(gap);indices.append([item['source_index'],frame])
        if (i+1)%10==0 or i+1==count:
            temp=partial.with_suffix('.tmp.npz')
            np.savez_compressed(temp,vig=np.asarray(values),vig_gap=np.asarray(gaps),indices=np.asarray(indices),demos_done=i+1,frame_stride=stride,temporal_average=context['scorer'].temporal_average)
            temp.replace(partial)
            write_json(out/'status.json',dict(phase='calibrating_vig',demos_done=i+1,total_demos=count,scores=len(values),elapsed=time.time()-started))
            print(f'CALIBRATE VIG {i+1}/{count}, {len(values)} scores',flush=True)
    assert np.array_equal(np.asarray(indices),np.load(out/'calibration.npz')['indices'])
    temp=target.with_suffix('.tmp.npz')
    np.savez_compressed(temp,vig=np.asarray(values),vig_gap=np.asarray(gaps),indices=np.asarray(indices),factors=context['policies'].factors,frame_stride=stride,temporal_average=context['scorer'].temporal_average)
    temp.replace(target)
    return np.asarray(values)


def write_checkpoint_gallery(out,rows,has_vig=True):
    description='flow percentile above VIG percentile' if has_vig else 'flow percentile'
    if rows and rows[0].get('vig_both',False):description='three stacked plots: flow, original VIG, and chunk-mean VIG'
    page=['<html><meta charset="utf-8"><title>Insertion rollouts</title><body>',
        f'<h1>{out.name}: {len(rows)}/50 rollout videos</h1>',
        f'<p>Hand and side cameras, all 16 action targets, {description}. Scores are calibrated on this checkpoint\'s training demos. Percentile ticks: 0, 20, 40, 60, 80, 100.</p>']
    if any(r.get('raw_axis_scope')=='rollout' for r in rows):
        page.append('<p>Raw y-axis ranges are fitted separately to each score in each rollout, with a small margin. Ranges stay fixed during playback.</p>')
    for kind,label in zip(KINDS,LABELS):
        page.append(f'<h2>{label}</h2>')
        for row in sorted((r for r in rows if r['kind']==kind),key=lambda r:r['case_id']):
            page.append(f'<details><summary>Case {row["case_id"]:02d}: {"success" if row["success"] else "failure"}</summary>'
                f'<a href="{row["video"]}">Percentile-axis MP4</a><br><video controls preload="none" width="512" src="{row["video"]}"></video>')
            if 'video_raw' in row:
                page.append(f'<p><a href="{row["video_raw"]}">Raw-axis MP4</a></p><video controls preload="none" width="512" src="{row["video_raw"]}"></video>')
            page.append('</details>')
    (out/'index.html').write_text('\n'.join(page+['</body></html>']))


def write_video(path,frames,paths,score_steps,flows,percentiles,states,meta,vig=None,vig_percentiles=None,vig_original=None,vig_original_percentiles=None):
    from PIL import Image,ImageDraw
    from vig_dagger.dataset import _ffmpeg_writer
    from vig_dagger.insertion_video_plot import VIDEO_HEIGHT,VIDEO_HEIGHT_VIG,draw_percentile_plot
    height=840 if vig_original is not None else (VIDEO_HEIGHT_VIG if vig is not None else VIDEO_HEIGHT)
    proc=_ffmpeg_writer(path,height,512)
    try:
        for t,(rgb,params) in enumerate(frames):
            im=Image.new('RGB',(512,height),(22,26,33));draw=ImageDraw.Draw(im)
            idx=max(0,bisect_right(score_steps,t)-1)
            for ci,c in enumerate(CAMERAS):
                im.paste(Image.fromarray(rgb[ci]),(ci*256,24));draw.text((ci*256+6,6),c,fill='white')
                xy,valid=project(paths[idx],params[c])
                # Draw into a clipped camera tile to avoid crossing panel borders.
                tile=im.crop((ci*256,24,(ci+1)*256,280));td=ImageDraw.Draw(tile)
                for j in range(1,len(xy)):
                    if valid[j-1] and valid[j] and np.max(np.abs(xy[j-1:j+1]))<10000:
                        td.line([tuple(xy[j-1]),tuple(xy[j])],fill='#ffaa35',width=2)
                for j in range(1,len(xy)):
                    if valid[j] and np.all(np.abs(xy[j])<10000):
                        x,y=xy[j];td.ellipse((x-2,y-2,x+2,y+2),fill='#ffcf78')
                im.paste(tile,(ci*256,24))
            draw.text((6,286),f'{meta["name"]} | {meta["kind"]} | step {t} | {"SUCCESS" if meta["success"] else "FAILURE"}',fill='white')
            draw.text((6,304),f'Flow {flows[idx]:.5f} | training percentile {percentiles[idx]:.1f} | 16 FK targets (orange)',fill='white')
            draw_percentile_plot(im,t,score_steps,percentiles,len(frames),label='Flow percentile' if vig is not None else None)
            if vig is not None:
                vig_label='VIG (chunk mean)' if meta.get('vig_temporal_average',False) else 'VIG'
                offset=180 if vig_original is not None else 0
                if vig_original is not None:
                    draw.text((6,484),f'VIG {vig_original[idx]:.5f} | training percentile {vig_original_percentiles[idx]:.1f}',fill='white')
                    draw_percentile_plot(im,t,score_steps,vig_original_percentiles,len(frames),top=500,label='VIG percentile',color='#a9d879')
                draw.text((6,484+offset),f'{vig_label} {vig[idx]:.5f} | training percentile {vig_percentiles[idx]:.1f}',fill='white')
                draw_percentile_plot(im,t,score_steps,vig_percentiles,len(frames),top=500+offset,label=vig_label+' percentile',color='#e5a3ff')
            proc.stdin.write(np.asarray(im).tobytes())
    finally:
        proc.stdin.close();err=proc.stderr.read().decode(errors='replace');code=proc.wait()
        if code:raise RuntimeError(err)


def bisect_right(values,value):
    import bisect
    return bisect.bisect_right(values,value)


def run_rollout(policy,env,case,root,out,reference,name,video_slots,*,vig_context=None,full_video=False,score_stride=8,max_steps=300,save_observations=False):
    from vig_dagger.policy import robot_state,camera_images
    assert score_stride in [1,2,4,8]
    payload=torch.load(root/case['state_file'],map_location='cpu',weights_only=False)
    obs,drift=restore(env,payload);base=env.unwrapped;fk=FK(base);fk_error=fk.check()
    start_q=base.agent.robot.get_qpos()[0].cpu().numpy();start_tcp=base.agent.tcp.pose.raw_pose[0].cpu().numpy()
    goal_rotation=base.goal_pose.sp.to_transformation_matrix()[:3,:3]
    torch.manual_seed(61000000+case['case_id'])
    noise=torch.randn(128,16,8,device=policy.device)
    samples=policy.sample_normalized(robot_state(obs,policy.device),camera_images(obs,CAMERAS,policy.device),[1],128,noise)[0]
    raw_samples=policy.denormalize_actions(samples).cpu().numpy()
    first_poses=fk.poses(raw_samples[:,0])
    projected=(first_poses[:,:3]-start_tcp[:3])@goal_rotation*1000
    from scipy.spatial.transform import Rotation
    rotations=Rotation.from_quat(first_poses[:,[4,5,6,3]]).as_matrix()
    current_rotation=Rotation.from_quat(start_tcp[[4,5,6,3]]).as_matrix()
    angular=Rotation.from_matrix(goal_rotation.T@rotations@current_rotation.T@goal_rotation).as_rotvec(degrees=True)
    frames=[];paths=[];score_steps=[];flows=[];percentiles=[];actions_log=[];tcp_log=[];distances=[];angles=[]
    original_values=[];original_percentiles=[];original_gaps=[]
    lowest_values=[];lowest_percentiles=[];lowest_gaps=[]
    saved_states=[];saved_samples=[];saved_flow_chunks=[]
    if save_observations:
        assert full_video and score_stride==1 and vig_context['scorer'].capture_samples
    vig_values=[];vig_percentiles=[];vig_gaps=[];replan_steps=[];replan_chunks=[];last_path=None
    steps=0;stable=0;success=False;started=time.time();last_chunk=None
    # Keep candidate footage until success/failure is known. Only the first
    # example of each outcome for each condition is exported.
    capture=full_video or not (video_slots.get((case['kind'],True)) and video_slots.get((case['kind'],False)))
    while steps<max_steps:
        if steps%score_stride==0:
            from contextlib import nullcontext
            devices=[policy.device.index or 0] if policy.device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices) if steps%8 else nullcontext():
                torch.manual_seed(50000000+case['case_id']*1000+steps)
                loss,chunks=policy.flow_score(robot_state(obs,policy.device),camera_images(obs,CAMERAS,policy.device),num_loss_samples=512)
            if save_observations:
                saved_states.append(robot_state(obs,policy.device)[0].cpu().numpy().copy())
                saved_flow_chunks.append(chunks[0].cpu().numpy().copy())
            if steps%8==0:
                last_chunk=policy.denormalize_actions(chunks)[0].cpu().numpy()
                last_path=np.vstack([base.agent.tcp.pose.sp.p,fk.poses(last_chunk)[:,:3]])
                replan_steps.append(steps);replan_chunks.append(last_chunk.copy())
            if not np.isfinite(last_chunk).all() or not math.isfinite(loss):raise RuntimeError('Nonfinite policy outputs')
            paths.append(last_path.copy())
            score_steps.append(steps);flows.append(loss)
            percentiles.append(float(np.searchsorted(reference,loss,side='right')*100/len(reference)))
            if vig_context is not None:
                value,gaps=score_vig(vig_context,robot_state(obs,policy.device),camera_images(obs,CAMERAS,policy.device),
                    70000000+case['case_id']*1000+steps)
                vig_values.append(value);vig_gaps.append(gaps)
                vig_percentiles.append(float(np.searchsorted(vig_context['reference'],value,side='right')*100/len(vig_context['reference'])))
                if 'original_reference' in vig_context:
                    original_values.append(vig_context['last_original_value']);original_gaps.append(vig_context['last_original_gaps'])
                    original_percentiles.append(float(np.searchsorted(vig_context['original_reference'],original_values[-1],side='right')*100/len(vig_context['original_reference'])))
                if 'lowest_reference' in vig_context:
                    lowest_values.append(vig_context['last_lowest_value']);lowest_gaps.append(vig_context['last_lowest_gaps'])
                    lowest_percentiles.append(float(np.searchsorted(vig_context['lowest_reference'],lowest_values[-1],side='right')*100/len(vig_context['lowest_reference'])))
                if save_observations:saved_samples.append(vig_context['last_samples'])
        if capture:
            rgb=np.stack([obs['sensor_data'][c]['rgb'][0].cpu().numpy() for c in CAMERAS])
            params={c:{k:v[0].cpu().numpy() for k,v in obs['sensor_param'][c].items() if k in ['intrinsic_cv','extrinsic_cv']} for c in CAMERAS}
            frames.append((rgb,params))
        action=last_chunk[steps%8]
        obs,_,terminated,truncated,info=env.step(action)
        steps+=1;actions_log.append(action.copy());tcp_log.append(base.agent.tcp.pose.raw_pose[0].cpu().numpy().copy())
        d,a=base._compute_distance();distances.append(float(d[0]));angles.append(float(a[0]))
        stable=stable+1 if bool(info['success'][0]) and bool(base.agent.is_grasping(base.charger)[0]) else 0
        if stable>=10:success=True;break
        if bool(truncated[0]):break
    stem=f'{case["case_id"]:02d}_{case["kind"]}'
    record=dict(case_id=case['case_id'],kind=case['kind'],pair=case['pair'],seed=case['seed'],success=success,steps=steps,
        final_distance_mm=distances[-1]*1000,final_angle_deg=float(np.rad2deg(angles[-1])),
        restore_drift_m=drift,fk_error_m=fk_error,start_qpos=start_q.tolist(),start_tcp=start_tcp.tolist(),
        seconds=time.time()-started,name=name,score_stride=score_stride,replan_stride=8,trace=f'traces/{stem}.npz')
    if vig_context is not None:record['vig_temporal_average']=vig_context['scorer'].temporal_average
    (out/'traces').mkdir(exist_ok=True)
    extra=dict(vig=np.asarray(vig_values),vig_percentile=np.asarray(vig_percentiles),vig_gap=np.asarray(vig_gaps)) if vig_context is not None else {}
    if original_values:
        extra.update(vig_original=np.asarray(original_values),vig_original_percentile=np.asarray(original_percentiles),vig_original_gap=np.asarray(original_gaps))
        record['vig_both']=True
    if lowest_values:
        extra.update(vig_lowest=np.asarray(lowest_values),vig_lowest_percentile=np.asarray(lowest_percentiles),vig_lowest_gap=np.asarray(lowest_gaps),vig_lowest_factors=np.asarray(policy.factors[:-1]))
        record['vig_lowest_reference_factor']=max(policy.factors)
    if save_observations:
        target=out/'observations'/f'{stem}.h5';target.parent.mkdir(exist_ok=True)
        temp=target.with_suffix('.partial.h5')
        with h5py.File(temp,'w') as h5:
            h5.attrs.update(cameras=json.dumps(CAMERAS),factors=json.dumps(list(policy.factors)),case_id=case['case_id'],pre_action=True,normalized_samples=True)
            for key,array in [('state',saved_states),('rgb',[f[0] for f in frames]),('vig_samples',saved_samples),('flow_chunks',saved_flow_chunks),
                              ('actions',actions_log),('intrinsic_cv',[[f[1][c]['intrinsic_cv'] for c in CAMERAS] for f in frames]),
                              ('extrinsic_cv',[[f[1][c]['extrinsic_cv'] for c in CAMERAS] for f in frames])]:
                array=np.asarray(array);assert len(array)==steps
                h5.create_dataset(key,data=array,chunks=(1,*array.shape[1:]),compression='lzf')
            h5.create_dataset('flow_seeds',data=50000000+case['case_id']*1000+np.arange(steps))
            h5.create_dataset('vig_seeds',data=70000000+case['case_id']*1000+np.arange(steps))
        temp.replace(target)
        record['observations']=str(target.relative_to(out))
    np.savez_compressed(out/record['trace'],score_steps=score_steps,flow=flows,percentile=percentiles,
        actions=np.asarray(actions_log),tcp=np.asarray(tcp_log),commanded_paths=np.asarray(paths),
        replan_steps=np.asarray(replan_steps),replan_chunks=np.asarray(replan_chunks),
        initial_sampled_chunks=raw_samples,initial_first_tcp_poses=first_poses,
        initial_projected_displacements_mm=projected,initial_angular_displacements_deg=angular,
        distance_m=distances,angle_rad=angles,**extra)
    if capture and (full_video or not video_slots.get((case['kind'],success))):
        (out/'videos').mkdir(exist_ok=True)
        path=out/'videos'/f'{stem}.mp4'
        write_video(path,frames,paths,score_steps,flows,percentiles,tcp_log,record,
            vig_values if vig_context is not None else None,vig_percentiles if vig_context is not None else None,
            original_values or None,original_percentiles or None)
        record['video']=str(path.relative_to(out));video_slots[(case['kind'],success)]=True
    return record


def worker(root,training,count,epoch,pilot=False,case_ids=None):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.insertion_env import make_insertion_env
    torch.set_num_threads(2)
    experiment=json.loads((training/'experiment.json').read_text())
    spec=next(r for r in experiment['runs'] if r['num_demos']==count)
    checkpoint=Path(spec['run_dir'])/f'epoch-{epoch:03d}.pt'
    name=f'demos{count}_epoch{epoch:03d}';out=root/('pilot' if pilot else 'runs')/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    policy=LibraryPolicy(checkpoint,'cuda:0');assert policy.cameras==CAMERAS and policy.horizon==16
    expected_factors=(1,2,4,8,16,32,64) if spec.get('variant')=='moe_7level' else (1,)
    assert policy.factors==expected_factors
    protocol=json.loads((root/'protocol.json').read_text())
    vig_enabled=protocol.get('vig_enabled',False)
    calibration_stride=protocol.get('calibration_stride',8)
    score_stride=protocol.get('score_stride',8)
    if pilot:
        # Small calibration only checks functionality. Never used by final runs.
        loss=calibrate(policy,out,training,1,calibration_stride)
    else:loss=calibrate(policy,out,training,count,calibration_stride)
    reference=np.sort(loss)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(out/'calibration.json',dict(checkpoint=str(checkpoint),sha256=sha,training_demo_count=1 if pilot else count,
        variant=spec.get('variant','dense_full'),evaluation_factor=1,
        scores=len(reference),stride=calibration_stride,loss_draws=512,percentiles={str(p):float(np.percentile(reference,p)) for p in [0,25,50,75,95,100]}))
    vig_context=None
    if vig_enabled:
        assert spec.get('variant')=='moe_7level' and not pilot
        from vig_dagger.policy import PolicySet
        from vig_dagger.scoring import VIGScorer
        vig_context=dict(policies=PolicySet({1:policy}),scorer=VIGScorer(num_samples=32,
            temporal_average=protocol.get('vig_temporal_average',False),include_original=protocol.get('vig_both',False)))
        vig_reference=calibrate_vig(vig_context,out,training,count,calibration_stride)
        vig_context['reference']=np.sort(vig_reference)
        if protocol.get('vig_both',False):
            original_cal=np.load(out/'calibration_vig_original.npz')
            assert not bool(original_cal.get('temporal_average',False))
            assert np.array_equal(original_cal['indices'],np.load(out/'calibration_vig.npz')['indices'])
            vig_context['original_reference']=np.sort(original_cal['vig'])
        previous_vig_meta=json.loads((out/'calibration_vig.json').read_text()) if (out/'calibration_vig.json').exists() else {}
        write_json(out/'calibration_vig.json',dict(checkpoint=str(checkpoint),sha256=sha,training_demo_count=count,
            samples_per_level=32,factors=list(policy.factors),scores=len(vig_reference),stride=calibration_stride,
            diagnostic_rng_preservation_verified=vig_context.get('rng_preservation_verified',False) or previous_vig_meta.get('diagnostic_rng_preservation_verified',False),
            temporal_average=protocol.get('vig_temporal_average',False),
            metric=protocol.get('vig_metric','Maximum over degraded levels of mean per-action-cell 1D Wasserstein distance in normalized actions'),
            percentiles={str(p):float(np.percentile(vig_reference,p)) for p in [0,25,50,75,95,100]}))
    cases=json.loads((root/'cases.json').read_text())
    if pilot:cases=[c for c in cases if c['pair']==0]
    if case_ids is not None:cases=[c for c in cases if c['case_id'] in case_ids]
    results=[]
    if (out/'rollouts.json').exists():results=json.loads((out/'rollouts.json').read_text())
    done={r['case_id'] for r in results};slots={(r['kind'],r['success']):True for r in results if 'video' in r}
    env=make_insertion_env(render_backend='cuda:0')
    try:
        for case in cases:
            if case['case_id'] in done:continue
            label=('MoE7 | ' if spec.get('variant')=='moe_7level' else '')+name
            record=run_rollout(policy,env,case,root,out,reference,label,slots,vig_context=vig_context,
                full_video=protocol.get('full_rollout_videos',False),score_stride=score_stride);results.append(record)
            write_json(out/'rollouts.json',results)
            if protocol.get('full_rollout_videos',False):write_checkpoint_gallery(out,results,vig_enabled)
            write_json(out/'status.json',dict(phase='rollouts',completed=len(results),total=len(cases),updated_at=time.time()))
            print(f'ROLLOUT {name} {case["case_id"]}: success={record["success"]} steps={record["steps"]}',flush=True)
    finally:env.close()
    import subprocess
    for video in (out/'videos').glob('*.mp4'):
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,capture_output=True)
    marker='complete.json' if case_ids is None or len(results)==50 else 'preflight_cases.json'
    write_json(out/marker,dict(name=name,demos=count,epoch=epoch,rollouts=len(results),successes=sum(r['success'] for r in results),finished_at=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','pilot'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--count',type=int,default=50);p.add_argument('--epoch',type=int,default=300)
    p.add_argument('--case-ids',type=int,nargs='+')
    args=p.parse_args()
    if args.command=='prepare':prepare(args.root,args.training)
    else:worker(args.root,args.training,args.count,args.epoch,args.command=='pilot',args.case_ids)


if __name__=='__main__':main()
