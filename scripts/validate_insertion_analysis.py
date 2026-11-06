"""Check recorded-action replay and expert recovery from restored analysis starts."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from vig_dagger.insertion_analysis import restore
from vig_dagger.insertion_env import make_insertion_env,solve_insertion
from vig_dagger.insertion_training import write_json


def run(root):
    env=make_insertion_env(render_backend='cuda:0')
    cases=json.loads((root/'cases.json').read_text());records=[]
    class Clock:
        @property
        def num_steps(self):return int(env.unwrapped.elapsed_steps[0])
    try:
        first=cases[0];payload=torch.load(root/first['state_file'],weights_only=False,map_location='cpu')
        restore(env,payload)
        with h5py.File(Path(first['source']).with_suffix('.h5')) as h5:
            actions=h5[f'traj_{first["source_episode"]}/actions'][:]
            for action in actions:obs,_,_,_,info=env.step(action)
        d,a=env.unwrapped._compute_distance()
        records.append(dict(test='replay_recorded_actions',case_id=0,success=bool(info['success'][0]),distance_mm=float(d[0])*1000,angle_rad=float(a[0])))
        write_json(root/'environment_validation.json',records)
        for case in cases:
            if case['pair']!=0:continue
            restore(env,torch.load(root/case['state_file'],weights_only=False,map_location='cpu'))
            fallback=False
            try:
                result,_=solve_insertion(env,Clock())
            except RuntimeError:
                # The demo expert uses local screw IK for alignment. For larger
                # evaluation rotations, check reachability with its RRT fallback.
                import sapien
                from dataclasses import replace
                from vig_dagger.insertion_env import make_planner,checked
                from vig_dagger.expert import planner as expert
                base=env.unwrapped;planner=make_planner(env)
                expert._init_speed(planner);expert._set_speed(planner,.5);planner.gripper_state=planner.CLOSED
                target=base.goal_pose.sp*sapien.Pose([-.08,0,0])*(base.charger.pose.sp.inv()*base.agent.tcp.pose.sp)
                checked(expert._transit(planner,target,replace(expert.StrategyParams.sample(0),transit_planner='rrt')),'validation RRT alignment')
                planner.close();fallback=True
                result,_=solve_insertion(env,Clock())
            d,a=env.unwrapped._compute_distance()
            records.append(dict(test='expert_recovery',case_id=case['case_id'],kind=case['kind'],
                success=bool(result[-1]['success'][0]),rrt_fallback=fallback,distance_mm=float(d[0])*1000,angle_rad=float(a[0])))
            write_json(root/'environment_validation.json',records)
            print(records[-1],flush=True)
    finally:env.close()
    write_json(root/'environment_validation.json',records)
    if not records[0]['success']:raise RuntimeError('Recorded demo action replay failed')
    if not all(r['success'] for r in records[1:]):raise RuntimeError('Expert could not solve a pilot start')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
