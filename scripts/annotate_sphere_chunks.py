"""Recover all 16 sampled commands and render full-chunk paths with original flow traces."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from annotate_sphere_rollouts import decoded_inputs,render_video,cumulative_command_paths


@torch.inference_mode()
def regenerate(policy,state,images,seed):
    torch.manual_seed(seed)
    inputs=policy.packer(policy._batch(state,images,[1]),device=str(policy.device))
    context,_=policy.policy.encode(inputs)
    noise=torch.randn(32,policy.horizon,policy.action_dim,device=policy.device,dtype=context.dtype)
    chunks=policy.policy.algorithm.generate_samples(policy._field(inputs,32),noise,context.repeat_interleave(32,dim=0))
    return policy.denormalize_actions(chunks).cpu().numpy()


def run(args):
    from vig_dagger.policy import LibraryPolicy
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/args.name;dest=out/'chunk_videos';dest.mkdir(exist_ok=True)
    policy=LibraryPolicy(args.root.parent/'toy_sphere_training_v1/policies'/args.name/'epoch-100.pt',f'cuda:{args.gpu}')
    calibration=json.loads((args.root/'camera_calibration.json').read_text())
    rows=json.loads((out/'rollouts.json').read_text())
    rows=[r for r in rows if args.case_start<=r['case_id']<args.case_stop]
    started=time.time()
    for count,row in enumerate(rows):
        stem=f"{row['case_id']:02d}_{row['kind']}";target=dest/f'{stem}.mp4'
        if target.exists() and target.with_suffix('.json').exists():continue
        trace=np.load(out/f'{stem}.npz');states=trace['states'];actions=trace['actions']
        previous=np.load(out/'annotated_videos'/f'{stem}_flow_actions.npz')
        steps=previous['steps'].tolist();flows=previous['flow'];images=decoded_inputs(out/row['video'],steps)
        chunks=[];max_error=0.
        for index,step in enumerate(steps):
            s=torch.tensor(states[step],device=policy.device)[None]
            im=torch.tensor(images[step],device=policy.device).permute(0,3,1,2)[None]
            raw=regenerate(policy,s,im,seed=91000+row['case_id']*10000+step)
            error=float(np.abs(raw[:,0,:3]-previous['sampled_first_xyz'][index]).max())
            assert error<1e-7,(row['case_id'],step,error)
            max_error=max(max_error,error);chunks.append(raw)
        chunks=np.array(chunks)
        assert chunks.shape==(len(steps),32,16,7) and np.isfinite(chunks).all()
        paths=cumulative_command_paths(states[steps,None,None,:3],chunks[:,:,:,:3])
        assert np.allclose(np.diff(paths,axis=2),chunks[:,:,:,:3],atol=1e-7)
        np.savez_compressed(dest/f'{stem}_chunks.npz',steps=steps,flow=flows,sampled_chunks=chunks)
        temporary=target.with_name(target.stem+'.partial.mp4')
        render_video(temporary,images,steps,states,actions,chunks[:,:,:,:3],flows,calibration,
                     args.name+' | '+row['kind']+f" | case {row['case_id']}",row['reason'],full_chunks=True)
        temporary.replace(target);temporary.with_suffix('.png').replace(target.with_suffix('.png'))
        target.with_suffix('.json').write_text(json.dumps(dict(case_id=row['case_id'],samples=32,horizon=16,
            path='p[k] = current EEF position + sum of translation commands 0..k-1; nominal commands, not simulated motion',
            magnification=1,orange='All 16 translation commands from each of 32 regenerated chunks',
            cyan='Up to 8 commands actually executed before the next replan; discarded predictions were not recorded',
            regenerated_first_action_max_error_m=max_error,flow='Unchanged from annotated_videos; no rescoring',
            source_observation='Exact saved states plus decoded original video images',success=row['success']),indent=2))
        progress=dict(completed=count+1,total=len(rows),elapsed_seconds=time.time()-started)
        (out/f'chunk_progress_{args.case_start}_{args.case_stop}.json').write_text(json.dumps(progress))
        print('FULL_CHUNK '+stem,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--case-start',type=int,default=0);p.add_argument('--case-stop',type=int,default=30)
    run(p.parse_args())
