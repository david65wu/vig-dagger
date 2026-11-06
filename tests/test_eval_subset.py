import json
from vig_dagger.evaluation import load


def test_existing_extra_eval_scenes_and_deferred_training_seed_are_excluded(tmp_path):
    for training_seed in [0,1,2]:
        d=tmp_path/f'retrain/base/seed{training_seed}/level1';d.mkdir(parents=True)
        for start in [60000,60025,60050,60075,60100,60125,60150,60175]:
            (d/f'seeds_{start}.jsonl').write_text('\n'.join(json.dumps({'seed':s,'success':True}) for s in range(start,start+25)))
    cfg={'retrain':{'seeds':[0,1]},'evaluation':{'seed_start':60000,'num_episodes':100}}
    data=load(tmp_path,cfg)
    assert len(data)==2
    assert all([r['seed'] for r in rows]==list(range(60000,60100)) for rows in data.values())
    assert len(load(tmp_path))==3  # Archived data still exists.


def test_deferred_driver_evaluations_remain_on_disk_but_out_of_summary(tmp_path):
    for label in ['driver/moe','driver/per_resolution/res2','retrain/base/seed0']:
        d=tmp_path/label/'level1';d.mkdir(parents=True)
        (d/'seeds_60000.jsonl').write_text(json.dumps({'seed':60000,'success':True})+'\n')
    cfg={'retrain':{'seeds':[0,1]},'evaluation':{'seed_start':60000,'num_episodes':100,'evaluate_drivers':False}}
    assert list(load(tmp_path,cfg))==[('retrain/base/seed0',1)]
    assert len(load(tmp_path))==3


def test_only_requested_driver_family_is_included(tmp_path):
    for label in ['driver/moe', 'driver/per_resolution/res2', 'retrain/base/seed0']:
        d = tmp_path / label / 'level1'
        d.mkdir(parents=True)
        (d / 'seeds_60000.jsonl').write_text(json.dumps({'seed': 60000, 'success': True}) + '\n')
    cfg = {'retrain': {'seeds': [0, 1]}, 'evaluation': {
        'seed_start': 60000, 'num_episodes': 100, 'evaluate_drivers': True, 'driver_rounds': ['moe']}}
    assert set(load(tmp_path, cfg)) == {('driver/moe', 1), ('retrain/base/seed0', 1)}
