"""Two-camera dense insertion policies with nested data and epoch checkpoints."""
import argparse
import bisect
import copy
import json
import math
import os
from pathlib import Path
import time

REPO=Path(__file__).resolve().parents[2]
CAMERAS=['hand_camera','side_camera']
CHECKPOINT_EPOCHS=[10,25,50,100,200,300]


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def dataset_view(source,dest,ordered_shards,count):
    task='plug_charger';dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/task/'demo_manifest.json').read_text())
    selected=[copy.deepcopy(manifest['shards'][i]) for i in ordered_shards[:count]]
    for shard in selected:
        shard['videos']={c:shard['videos'][c] for c in CAMERAS}
        paths=[shard['hdf5']]+[p for files in shard['videos'].values() for p in files.values()]
        for relative in paths:
            src=source/task/relative;dst=dest/task/relative;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists(): os.link(src,dst)
    manifest.update(shards=selected,cameras=CAMERAS,source_dataset=str(source),subset_shuffle_seed=20261102)
    write_json(dest/task/'demo_manifest.json',manifest)
    top=json.loads((source/'dataset_manifest.json').read_text());top['num_demos']=count
    write_json(dest/'dataset_manifest.json',top)


def base_config(dataset,out,variant='dense_full'):
    config=json.loads((REPO/'src/vig_dagger/configs/policy.json').read_text());config.pop('_comment',None)
    config['seed']=0
    config['features']['features'][2]['cameras']=CAMERAS
    config['policy']['cameras']=2
    factors=[1,2,4,8,16,32,64] if variant=='moe_7level' else [1]
    if variant=='dense_full':config['policy']['head'].pop('moe',None)
    config['policy']['conditioning']={'type':'indexed','vocab_sizes':[len(factors)]}
    config['processing']['resolution']={'factors':factors}
    config['dataset'].update(root=str(dataset),selection={'val_fraction':0.0})
    config['dataset']['sampling'].update(k_passes=1,seed=42,window_shards=20)
    config['loader'].update(batch_size=64,num_workers=2,persistent_workers=True)
    config['training'].update(device='cuda:0',max_updates=1,learning_rate=1e-4,weight_decay=1e-4,
        warmup_updates=100,min_lr_ratio=1e-4,log_every=50,validate_every=1000,
        checkpoint_every=1000,keep_checkpoints=2)
    config['logging']={'out_dir':str(out),'wandb':{'enabled':False}}
    return config


def prepare(root):
    import numpy as np
    from robopolicy import RunConfig,build_dataset
    from robopolicy.datasets.planning import plan_windows
    from vig_dagger.dataset import combine
    if (root/'experiment.json').exists(): return
    sources=json.loads((root/'collection_sources.json').read_text())
    all_data=root/'datasets/all_500'
    inputs=[(Path(sources['preview'])/'dataset',None)]+[(Path(p['dataset']),None) for p in sources['parts']]
    if not (all_data/'dataset_manifest.json').exists():
        temp=root/'datasets'/f'all_500_build_{int(time.time())}'
        assert combine(inputs,temp)==500
        temp.rename(all_data)
    manifest=json.loads((all_data/'plug_charger/demo_manifest.json').read_text())
    assert len(manifest['shards'])==500
    # Record physical episode identities before any training. All subsets use
    # prefixes of one permutation, and retain the same shard IDs and demo keys.
    identities=[]
    for source in [Path(sources['preview'])]+[Path(p['out']) for p in sources['parts']]:
        files=list((source/'raw').glob('*.json'));assert len(files)==1
        episodes=json.loads(files[0].read_text())['episodes']
        identities.extend(dict(source=str(files[0]),episode_id=e['episode_id'],seed=e['seed'],
            initialization=e['initialization']) for e in episodes)
    assert len(identities)==500 and len({e['seed'] for e in identities})==500
    order=np.random.default_rng(20261102).permutation(500).tolist()
    write_json(root/'demo_order.json',[dict(rank=i,source_index=j,**identities[j]) for i,j in enumerate(order)])
    runs=[]
    for count in [50,200,500]:
        data_root=root/'datasets'/f'train_{count}';dataset_view(all_data,data_root,order,count)
        out=root/'policies'/f'demos{count}_dense_full';out.mkdir(parents=True,exist_ok=True)
        values=base_config(data_root,out);config=RunConfig.model_validate(values)
        data=build_dataset(config.dataset,config.features,'train')
        assert len(data.selection)==count
        # Shards shuffle before worker partitioning. Count each worker's partial
        # batch for every epoch, so checkpoints fall on actual pass boundaries.
        epoch_updates=[]
        for epoch in range(300):
            updates=0
            for worker in range(config.loader.num_workers):
                windows=plan_windows(data.episodes,config.dataset.sampling,epoch,worker,config.loader.num_workers,False)
                samples=sum(max(0,e.length-config.features.max_future) for w in windows for e in w.episodes)
                updates+=math.ceil(samples/config.loader.batch_size)
            epoch_updates.append(updates)
        cumulative=np.cumsum(epoch_updates).tolist()
        values['training']['max_updates']=cumulative[-1]
        spec=dict(name=out.name,num_demos=count,epochs=300,cameras=CAMERAS,training_samples=len(data),
            total_updates=cumulative[-1],epoch_updates=epoch_updates,cumulative_updates=cumulative,
            checkpoints={str(cumulative[e-1]):f'epoch-{e:03d}.pt' for e in CHECKPOINT_EPOCHS},
            selection=[list(v) for v in data.selection],dataset_fingerprint=data.fingerprint,
            run_dir=str(out),training_seed=0,validation='No demonstration holdout; evaluate separate closed-loop ID scenes')
        data.close()
        write_json(out/'run_config.json',RunConfig.model_validate(values).model_dump(mode='json'))
        write_json(out/'experiment.json',spec);runs.append(spec)
    assert set(map(tuple,runs[0]['selection']))<=set(map(tuple,runs[1]['selection']))<=set(map(tuple,runs[2]['selection']))
    write_json(root/'experiment.json',dict(task='charger_insertion',created_at=time.time(),runs=runs,
        cameras=CAMERAS,checkpoint_epochs=CHECKPOINT_EPOCHS,preview_demos_retained=5,
        main_commit='5d2ab39603bbb1269b5129ce6e9705df52ae7add'))


def preflight(root,dataset=None,variant='dense_full'):
    import h5py
    import torch
    from robopolicy import RunConfig,Trainer
    from vig_dagger.policy import LibraryPolicy
    torch.set_num_threads(2)
    out=root/'preflight';out.mkdir(parents=True,exist_ok=True)
    dataset=dataset or root/'datasets/train_50'
    expected=json.loads((dataset/'dataset_manifest.json').read_text())['num_demos']
    values=base_config(dataset,out,variant)
    values['training'].update(max_updates=8,warmup_updates=2,log_every=1)
    config=RunConfig.model_validate(values)
    trainer=Trainer(config)
    assert len(trainer.dataset.selection)==expected and len(trainer.validation_dataset)==0
    assert trainer.policy.config.cameras==2 and list(trainer.packer.schema['images'].cameras)==CAMERAS
    trainer.fit(until_step=4)
    resumed=Trainer.resume(out/'latest.pt');assert resumed.completed_updates==4
    resumed.fit();assert resumed.completed_updates==8
    del trainer,resumed
    policy=LibraryPolicy(out/'latest.pt','cuda:0')
    raw=Path(json.loads((root/'collection_sources.json').read_text())['preview'])/'raw/preview.h5'
    with h5py.File(raw) as h5:
        g=h5['traj_0'];state=torch.from_numpy(g['obs/agent/qpos'][0:1]).float()
        images=torch.stack([torch.from_numpy(g[f'obs/sensor_data/{c}/rgb'][0:1]).permute(0,3,1,2) for c in CAMERAS],dim=1)
    expected_factors=(1,2,4,8,16,32,64) if variant=='moe_7level' else (1,)
    assert policy.factors==expected_factors
    for factor in expected_factors:
        actions=policy.act(state,images,factor=factor)
        assert tuple(actions.shape)==(1,16,8) and torch.isfinite(actions).all()
    flow,_=policy.flow_score(state,images,num_loss_samples=512)
    assert math.isfinite(flow)
    torch.manual_seed(61000000)
    samples=policy.sample_normalized(state,images,[1],128,torch.randn(128,16,8,device=policy.device))
    assert tuple(samples.shape)==(1,128,16,8) and torch.isfinite(samples).all()
    write_json(root/'training_preflight.json',dict(passed=True,steps=8,resume_verified=True,demos=expected,dataset=str(dataset),
        variant=variant,resolution_factors_tested=list(expected_factors),loss_draws=512,distribution_samples=128,
        cameras=policy.cameras,action_shape=list(actions.shape),flow_smoke=flow,finished_at=time.time()))


def train(out,gpu):
    import torch
    from robopolicy import RunConfig,Trainer
    from vig_dagger.jobs import exit_on_sigterm
    exit_on_sigterm();torch.set_num_threads(2)
    spec=json.loads((out/'experiment.json').read_text())
    config=RunConfig.model_validate_json((out/'run_config.json').read_text())
    class InsertionTrainer(Trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert self.policy.config.cameras==2 and list(self.packer.schema['images'].cameras)==CAMERAS
            assert len(self.dataset.selection)==spec['num_demos'] and len(self.validation_dataset)==0
            if spec.get('variant')=='moe_7level':
                assert self.policy.config.head.moe is not None
                assert tuple(self.config.processing.resolution.factors)==(1,2,4,8,16,32,64)
            if spec.get('normalization_reference'):
                reference=json.loads(Path(spec['normalization_reference']).read_text())
                assert self.packer.normalizer.statistics.model_dump(mode='json')==reference
            self.started=time.time();self.initial_step=self.completed_updates
            self.report('running')
        def report(self,state,loss=None):
            elapsed=time.time()-self.started
            rate=(self.completed_updates-self.initial_step)/max(elapsed,1e-6)
            boundaries=spec['cumulative_updates'];epoch=bisect.bisect_right(boundaries,self.completed_updates)
            previous=boundaries[epoch-1] if epoch else 0
            fraction=(self.completed_updates-previous)/spec['epoch_updates'][epoch] if epoch<300 else 0
            row=dict(state=state,pid=os.getpid(),gpu=gpu,step=self.completed_updates,total_updates=spec['total_updates'],
                epoch=epoch+fraction,updated_at=time.time(),loss=loss,updates_per_second=rate,
                eta_seconds=(spec['total_updates']-self.completed_updates)/rate if rate else None)
            write_json(out/'status.json',row);print(json.dumps(row),flush=True)
        def update(self,batch):
            loss,updated=super().update(batch)
            if not math.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            if updated:
                name=spec['checkpoints'].get(str(self.completed_updates))
                if name:self.save(out/name)
                if self.completed_updates%50==0:self.report('running',loss)
            return loss,updated
    try:
        write_json(out/'status.json',dict(state='starting',pid=os.getpid(),gpu=gpu,updated_at=time.time()))
        candidates=[(int(p.stem.split('-')[1]),p) for p in out.glob('step-*.pt')]
        candidates += [(int(step),out/name) for step,name in spec['checkpoints'].items() if (out/name).exists()]
        trainer=InsertionTrainer.resume(max(candidates)[1]) if candidates else InsertionTrainer(config)
        if trainer.completed_updates<spec['total_updates']:trainer.fit()
        else:trainer.save(out/'latest.pt');trainer.close()
        trainer.report('complete')
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',pid=os.getpid(),gpu=gpu,updated_at=time.time(),error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','train'])
    p.add_argument('--root',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--gpu',type=int)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--variant',choices=['dense_full','moe_7level'],default='dense_full')
    args=p.parse_args()
    if args.command=='prepare':prepare(args.root)
    elif args.command=='preflight':preflight(args.root,args.dataset,args.variant)
    else:train(args.run_dir,args.gpu)
