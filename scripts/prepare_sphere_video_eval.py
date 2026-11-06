"""Validate camera transforms and five reachable endpoint pairs outside the training sphere."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch


def prepare(args):
    from vig_dagger.toy_reach_env.sphere_config import SphereReachEnvCfg
    from vig_dagger.toy_reach_env.sphere_environment import SphereReachEnv
    from vig_dagger.toy_reach_env.sphere_sampling import sample_sphere_pair, sample_fixed_goal_pair
    from vig_dagger.toy_projection import rotation,camera_pose,project
    import omni.replicator.core as rep
    cfg=SphereReachEnvCfg();cfg.sphere_radius=.20
    cfg.scene.wrist_cam.update_latest_camera_pose=True
    cfg.scene.third_person_cam.update_latest_camera_pose=True
    env=SphereReachEnv(cfg);calibration=None;checks=[];cases=[];rejected=[]
    annotators={}
    for name in ['third_person','wrist']:
        annotators[name]=rep.AnnotatorRegistry.get_annotator('CameraParams')
        annotators[name].attach(env.scene[name+'_cam'].render_product_paths[0])
    def rendered_pose(name):
        # USD sensor metadata can remain at the authored pose when PhysX uses
        # Fabric. Read the render product's actual view matrix instead.
        data=annotators[name].get_data()
        transform=np.linalg.inv(np.array(data['cameraViewTransform']).reshape(4,4)).T
        return transform[:3,3],transform[:3,:3]@np.diag([1,-1,-1]),data
    try:
        for seed in range(7000,7030):
            a,b=(sample_fixed_goal_pair(cfg.sphere_center,.20,seed=seed) if args.endpoint_mode=='fixed_goal'
                 else sample_sphere_pair(cfg.sphere_center,.20,.10,seed))
            try:
                for label,point in [('A',a),('B',b)]:
                    if label=='A':obs,_=env.reset(seed=seed,options=dict(point_a=a,point_b=b))
                    else:obs=env.place_eef(b)
                    state=obs['policy'][0].cpu().numpy().copy()
                    error=float(np.linalg.norm(state[:3]-point));assert error<.001
                    robot=env.scene['robot']
                    root_p=robot.data.root_pos_w[0].cpu().numpy();root_r=rotation(robot.data.root_quat_w[0].cpu().numpy())
                    tcp_p=root_p+root_r@state[:3];tcp_r=root_r@rotation(state[3:7])
                    if calibration is None:
                        calibration=dict(root_position=root_p.tolist(),root_rotation=root_r.tolist(),image_size=[256,256])
                        for name in ['third_person','wrist']:
                            c=env.scene[name+'_cam'].data;p,r,_=rendered_pose(name)
                            calibration[name]=dict(intrinsics=c.intrinsic_matrices[0].cpu().numpy().tolist(),
                                position_world=p.tolist(),rotation_ros_world=r.tolist(),
                                tcp_relative_position=(tcp_r.T@(p-tcp_p)).tolist(),tcp_relative_rotation=(tcp_r.T@r).tolist())
                    for name in ['third_person','wrist']:
                        c=env.scene[name+'_cam'].data;p,r,render_data=rendered_pose(name)
                        predicted_p,predicted_r=camera_pose(calibration,state,name)
                        pos_error=float(np.linalg.norm(predicted_p-p));rot_error=float(np.abs(predicted_r-r).max())
                        assert pos_error<2e-5 and rot_error<2e-4,(name,pos_error,rot_error)
                        # Independent sensor-space projection checks the saved transform convention.
                        points=np.stack([state[:3],a,b,state[:3]+[.01,.02,.03]])
                        pixels,depth=project(calibration,state,name,points)
                        world=points@root_r.T+root_p
                        homogeneous=np.concatenate([world,np.ones((len(world),1))],axis=1)
                        clip=homogeneous@np.array(render_data['cameraViewTransform']).reshape(4,4)@np.array(render_data['cameraProjection']).reshape(4,4)
                        ndc=clip[:,:2]/clip[:,3:4]
                        expected=np.stack(((ndc[:,0]+1)*128,(1-ndc[:,1])*128),axis=1)
                        visible=(depth>.02)&(np.max(np.abs(expected),axis=1)<10000)
                        pixel_error=float(np.abs(pixels[visible]-expected[visible]).max()) if visible.any() else 0.
                        assert pixel_error<.1,(name,pixel_error)
                        checks.append(dict(seed=seed,endpoint=label,camera=name,position_error_m=pos_error,
                                           rotation_matrix_error=rot_error,pixel_error=pixel_error,placement_error_m=error))
            except RuntimeError as exc:
                rejected.append(dict(seed=seed,error=str(exc)));continue
            assert np.linalg.norm(a-np.array(cfg.sphere_center))>.15+.049
            assert np.linalg.norm(b-np.array(cfg.sphere_center))>.15+.049
            cases.append(dict(kind='outside_sphere_a_to_b',point_a=a.tolist(),point_b=b.tolist(),start=a.tolist(),
                              reset_seed=seed,sphere_radius=.20))
            print('OUTSIDE_PAIR_VALID '+json.dumps(cases[-1]),flush=True)
            if len(cases)==5:break
        assert len(cases)==5
        (args.root/'camera_calibration.json').write_text(json.dumps(calibration,indent=2))
        (args.root/'outside_cases.json').write_text(json.dumps(cases,indent=2))
        (args.root/'preflight.json').write_text(json.dumps(dict(state='complete',outside_radius_m=.20,
            training_radius_m=.15,checks=checks,rejected=rejected),indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--endpoint-mode',choices=['random_pair','fixed_goal'],default='random_pair')
    args=p.parse_args();args.root.mkdir(exist_ok=True)
    from isaaclab.app import AppLauncher
    class Launcher(AppLauncher):
        def _resolve_device_settings(self,values):
            super()._resolve_device_settings(values);values['active_gpu']=args.gpu;values['multi_gpu']=False
    app=Launcher(dict(headless=True,enable_cameras=True,device='cpu')).app
    try:
        torch.set_num_threads(2)
        with torch.inference_mode():prepare(args)
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close()
