"""Project sampled/executed commands onto rollout cameras and add a flow trace."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from vig_dagger.toy_projection import project


def decoded_inputs(path, steps):
    cap=cv2.VideoCapture(str(path));wanted={s//4:s for s in steps};images={};frame=0
    try:
        while wanted:
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Missing frames in {path}: {wanted}')
            if frame in wanted:
                step=wanted.pop(frame)
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                images[step]=np.stack((rgb[28:284,:256],rgb[28:284,256:512]))
            frame+=1
    finally:cap.release()
    return images


def cumulative_command_paths(position,commands):
    """Nominal base-frame command sums, not forward-simulated robot positions."""
    commands=np.asarray(commands)
    zeros=np.zeros((*commands.shape[:-2],1,3),dtype=commands.dtype)
    return np.asarray(position)+np.concatenate((zeros,np.cumsum(commands,axis=-2)),axis=-2)


def chunk_overlay(rgb,state,camera,calibration,samples,executed):
    panel=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(512,512),interpolation=cv2.INTER_LINEAR)
    assert samples.shape==(32,16,3)
    paths=cumulative_command_paths(state[:3],samples)
    actual=cumulative_command_paths(state[:3],executed)
    def draw_path(canvas,points,color,thickness):
        pixels,depth=project(calibration,state,camera,points);pixels*=2
        coordinates=np.clip(np.rint(pixels),-1000000,1000000).astype(int)
        for j in range(len(points)-1):
            if depth[j]<=.001 or depth[j+1]<=.001:continue
            valid,start,end=cv2.clipLine((0,0,512,512),tuple(coordinates[j]),tuple(coordinates[j+1]))
            if valid:cv2.arrowedLine(canvas,start,end,color,thickness,cv2.LINE_AA,tipLength=.25)
        end=coordinates[-1]
        if depth[-1]>.001 and 0<=end[0]<512 and 0<=end[1]<512:
            cv2.circle(canvas,tuple(end),2,color,-1,cv2.LINE_AA)
    fan=panel.copy()
    for path in paths:draw_path(fan,path,(0,155,255),1)
    panel=cv2.addWeighted(fan,.65,panel,.35,0)
    draw_path(panel,actual,(255,235,0),2)
    pixels,depth=project(calibration,state,camera,state[None,:3]);pixels*=2
    origin=tuple(np.clip(np.rint(pixels[0]),-1000000,1000000).astype(int))
    if depth[0]>.001 and 0<=origin[0]<512 and 0<=origin[1]<512:
        cv2.circle(panel,origin,4,(255,255,255),-1,cv2.LINE_AA)
    else:cv2.putText(panel,'TCP outside this view',(12,490),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(panel,camera.replace('_',' '),(12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),2,cv2.LINE_AA)
    return panel


def camera_overlay(rgb,state,camera,calibration,samples,executed):
    panel=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(512,512),interpolation=cv2.INTER_LINEAR)
    points=np.concatenate((state[None,:3],state[None,:3]+5*samples,state[None,:3]+5*executed[None]))
    pixels,depth=project(calibration,state,camera,points);pixels*=2
    origin=tuple(np.clip(np.rint(pixels[0]),-1000000,1000000).astype(int))
    fan=panel.copy()
    if depth[0]>.001:
        for j in range(1,len(points)-1):
            if depth[j]<=.001:continue
            end=tuple(np.clip(np.rint(pixels[j]),-1000000,1000000).astype(int))
            valid,start,end=cv2.clipLine((0,0,512,512),origin,end)
            if valid:cv2.arrowedLine(fan,start,end,(0,175,255),1,cv2.LINE_AA,tipLength=.2)
        panel=cv2.addWeighted(fan,.65,panel,.35,0)
        if depth[-1]>.001:
            end=tuple(np.clip(np.rint(pixels[-1]),-1000000,1000000).astype(int))
            valid,start,end=cv2.clipLine((0,0,512,512),origin,end)
            if valid:cv2.arrowedLine(panel,start,end,(255,235,0),3,cv2.LINE_AA,tipLength=.22)
        if 0<=origin[0]<512 and 0<=origin[1]<512:cv2.circle(panel,origin,4,(255,255,255),-1,cv2.LINE_AA)
        else:cv2.putText(panel,'TCP outside this view',(12,490),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),2,cv2.LINE_AA)
    else:cv2.putText(panel,'TCP behind camera',(12,490),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(panel,camera.replace('_',' '),(12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),2,cv2.LINE_AA)
    return panel


def render_video(path,images,steps,states,actions,samples,flows,calibration,title,reason,full_chunks=False,training_p95=None):
    width,height=1024,800;fps=7.5
    cmd=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),
         '-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    ymax=max(.01,float(np.ceil(np.max(flows)*110)/100))
    left,right,top,bottom=65,995,645,758
    times=np.array(steps)/60
    xs=left+times/20*(right-left);ys=bottom-np.array(flows)/ymax*(bottom-top)
    font=cv2.FONT_HERSHEY_SIMPLEX
    try:
        for i,step in enumerate(steps):
            frame=np.full((height,width,3),248,np.uint8)
            cv2.putText(frame,title,(12,23),font,.64,(20,20,20),1,cv2.LINE_AA)
            label=(f't = {step/60:.2f}s | scale x1 | orange: 32 x 16-step command paths | cyan: up to 8 executed commands'
                   if full_chunks else f't = {step/60:.2f} s | arrows x5 | orange: sampled actions | cyan: executed action')
            cv2.putText(frame,label,(12,50),font,.51 if full_chunks else .59,(20,20,20),1,cv2.LINE_AA)
            for j,camera in enumerate(['third_person','wrist']):
                if full_chunks:
                    overlay=chunk_overlay(images[step][j],states[step],camera,calibration,samples[i],actions[step:step+8,:3])
                else:
                    overlay=camera_overlay(images[step][j],states[step],camera,calibration,samples[i],actions[step,:3])
                frame[64:576,j*512:(j+1)*512]=overlay
            cv2.putText(frame,f'Flow loss: {flows[i]:.4f}',(15,610),font,.8,(25,25,25),2,cv2.LINE_AA)
            cv2.putText(frame,'Recomputed from saved video frames; 32 chunks x 64 noise/time draws',(310,608),font,.51,(70,70,70),1,cv2.LINE_AA)
            for x in [0,5,10,15,20]:
                px=int(left+x/20*(right-left));cv2.line(frame,(px,top),(px,bottom),(225,225,225),1)
                cv2.putText(frame,str(x),(px-8,bottom+22),font,.48,(50,50,50),1,cv2.LINE_AA)
            for y in [0,ymax/2,ymax]:
                py=int(bottom-y/ymax*(bottom-top));cv2.line(frame,(left,py),(right,py),(225,225,225),1)
                cv2.putText(frame,f'{y:.3f}',(7,py+5),font,.42,(50,50,50),1,cv2.LINE_AA)
            curve=np.stack((xs[:i+1],ys[:i+1]),axis=1).round().astype(np.int32)
            if len(curve)>1:cv2.polylines(frame,[curve],False,(190,90,30),2,cv2.LINE_AA)
            cv2.circle(frame,tuple(curve[-1]),4,(50,50,210),-1,cv2.LINE_AA)
            cv2.putText(frame,'Time (s)',(475,793),font,.48,(50,50,50),1,cv2.LINE_AA)
            if training_p95 is not None:
                from add_sphere_flow_reference import panel
                panel(frame,np.asarray(flows),steps,i,training_p95)
            if i==0:cv2.imwrite(str(path.with_suffix('.png')),frame)
            proc.stdin.write(frame.tobytes())
        cv2.rectangle(frame,(0,0),(1023,28),(248,248,248),-1)
        cv2.putText(frame,title+' | '+reason,(12,23),font,.6,(20,20,20),1,cv2.LINE_AA)
        for _ in range(8):proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait():raise RuntimeError(f'Video encoding failed: {path}')


def run(args):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_flow_search import sample_and_score
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/args.name;dest=out/'annotated_videos';dest.mkdir(exist_ok=True)
    policy=LibraryPolicy(args.root.parent/'toy_sphere_training_v1/policies'/args.name/'epoch-100.pt',f'cuda:{args.gpu}')
    calibration=json.loads((args.root/'camera_calibration.json').read_text())
    rows=json.loads((out/'rollouts.json').read_text())
    if args.case_id is not None:rows=[r for r in rows if r['case_id']==args.case_id]
    if args.case_start is not None:rows=[r for r in rows if args.case_start<=r['case_id']<args.case_stop]
    progress_prefix='annotation' if args.case_start is None else f'annotation_part_{args.case_start}_{args.case_stop}'
    started=time.time()
    for count,row in enumerate(rows):
        stem=f"{row['case_id']:02d}_{row['kind']}";target=dest/f'{stem}.mp4'
        if target.exists() and (dest/f'{stem}.json').exists():continue
        a=np.load(out/f'{stem}.npz');states=a['states'];actions=a['actions']
        steps=list(range(0,len(actions),8));assert steps
        images=decoded_inputs(out/row['video'],steps)
        samples=[];flows=[]
        for step in steps:
            s=torch.tensor(states[step],device=policy.device)[None]
            im=torch.tensor(images[step],device=policy.device).permute(0,3,1,2)[None]
            values,raw,losses,_=sample_and_score(policy,s,im,seed=91000+row['case_id']*10000+step,samples=32,draws=64)
            samples.append(raw[:,0,:3]);flows.append(values['flow_mean'])
        samples=np.array(samples);flows=np.array(flows)
        assert np.isfinite(samples).all() and np.isfinite(flows).all()
        np.savez_compressed(dest/f'{stem}_flow_actions.npz',steps=steps,flow=flows,sampled_first_xyz=samples)
        render_video(target,images,steps,states,actions,samples,flows,calibration,
                     args.name+' | '+row['kind']+f" | case {row['case_id']}",row['reason'])
        meta=dict(case_id=row['case_id'],flow_source='exact recorded state plus decoded H264 camera frames',
                  samples=32,noise_time_draws_per_chunk=64,every_control_steps=8,arrow_magnification=5,
                  orange='32 independently sampled first translation commands',cyan='actual executed translation command',
                  camera_projection='Calibrated world/ROS transforms; wrist pose reconstructed from rigid TCP attachment',
                  outcome=row['reason'],success=row['success'])
        (dest/f'{stem}.json').write_text(json.dumps(meta,indent=2))
        (out/f'{progress_prefix}_progress.json').write_text(json.dumps(dict(completed=count+1,total=len(rows),elapsed_seconds=time.time()-started)))
        print('ANNOTATED '+stem,flush=True)
    (out/f'{progress_prefix}_done.json').write_text(json.dumps(dict(episodes=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--case-id',type=int)
    p.add_argument('--case-start',type=int);p.add_argument('--case-stop',type=int,default=30)
    run(p.parse_args())
