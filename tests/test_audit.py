import numpy as np
from vig_dagger.audit import wilson
from vig_dagger.analysis import segments


def test_wilson_edge_cases():
    assert wilson(0,0) is None
    lo, hi = wilson(0,10)
    assert abs(lo)<1e-12 and .27<hi<.29
    lo, hi = wilson(10,10)
    assert .71<lo<.73 and abs(hi-1)<1e-12


def test_terminal_expert_segment_is_retained():
    assert segments(np.array([0,1,1,0,1])) == [(1,2),(4,1)]


def test_live_inventory_labels_failures_and_missing_recordings(tmp_path):
    import csv
    import json
    from vig_dagger.audit import video_inventory
    out=tmp_path/'dagger/moe/flow_p95';(out/'episodes').mkdir(parents=True)
    (out/'settings.json').write_text(json.dumps({'target':1}))
    rows=[dict(seed=1,success=False,reason='expert_failed',expert_failure='charger grasp missed',file=None,finished_at=1),
          dict(seed=2,success=True,reason='success_expert',expert_failure=None,file='trajectories/w.hdf5',finished_at=2)]
    (out/'episodes/worker_00.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    dest=tmp_path/'results/dagger_audit';dest.mkdir(parents=True)
    video_inventory(tmp_path)
    failed=list(csv.DictReader((dest/'failed_attempts.csv').open()))
    assert len(failed)==1 and failed[0]['outcome']=='FAILED'
    assert failed[0]['reason']=='charger grasp missed'
    assert failed[0]['status'].startswith('unavailable:')
    assert 'FAILED' in (dest/'FAILED_ATTEMPTS.md').read_text()


def test_status_handles_round_and_scorer_underscores(monkeypatch):
    from types import SimpleNamespace
    from vig_dagger import ops
    from vig_dagger.workers import dagger
    seen=[]
    monkeypatch.setattr(ops.Paths,'of',lambda cfg:SimpleNamespace(dagger=lambda r,s:seen.append((r,s)) or 'path'))
    monkeypatch.setattr(dagger,'count_successes',lambda p:(97,110))
    value=ops._progress({'rounds':['moe','per_resolution'],'dagger':{'num_demos':100}},'dagger_per_resolution_vig_p90_03')
    assert seen==[('per_resolution','vig_p90')]
    assert '97/100' in value


def test_event_clips_resume_without_reencoding_completed_clip(tmp_path):
    import shutil,subprocess
    import pytest
    from vig_dagger.audit import ensure_clips
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg unavailable')
    target=tmp_path/'1.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=96x64:r=60:d=1','-c:v','libx264','-threads','1',str(target)],check=True)
    data={'take_steps':[15],'return_steps':[30]}
    ensure_clips(target,data,60)
    clip=tmp_path/'1_000_takeover.mp4'; stamp=clip.stat().st_mtime_ns
    (tmp_path/'1_001_return.mp4').unlink()
    ensure_clips(target,data,60)
    assert clip.stat().st_mtime_ns==stamp
    assert (tmp_path/'1_001_return.mp4').stat().st_size>0
    assert target.with_suffix('.complete.json').exists()


def test_render_order_alternates_successes_and_failures():
    from vig_dagger.audit import interleave_outcomes, video_path
    from pathlib import Path
    rows=[{'seed':i,'success':i>=3} for i in range(7)]
    ordered=interleave_outcomes(rows)
    assert [e['success'] for e in ordered]==[True,False,True,False,True,False,True]
    assert sorted(e['seed'] for e in ordered)==list(range(7))
    assert video_path(Path('/report'),Path('/dagger/moe/vig_p99'),rows[0])==Path('/report/videos/moe/vig_p99/failed/0.mp4')
    assert video_path(Path('/report'),Path('/dagger/moe/vig_p99'),rows[3])==Path('/report/videos/moe/vig_p99/success/3.mp4')


def test_inventory_completes_for_two_or_three_selected_variants(tmp_path):
    import json
    from vig_dagger.audit import selected_variants, video_inventory
    for n in [2,3]:
        root=tmp_path/str(n)
        dest=root/'results/dagger_audit';dest.mkdir(parents=True)
        ids=[f'custom/variant_{i}' for i in range(n)]
        (root/'config.json').write_text(json.dumps({'analysis':{'variants':ids}}))
        for name in ids:
            out=root/'dagger'/name;(out/'episodes').mkdir(parents=True)
            (out/'settings.json').write_text(json.dumps({'target':1}))
            (out/'episodes/worker_00.jsonl').write_text(json.dumps({'seed':1,'success':True,'reason':'success_policy','file':None,'finished_at':1})+'\n')
        assert selected_variants(root)==set(ids)
        assert video_inventory(root)  # Historical missing RGB is disclosed, not endlessly pending.
        (root/'config.json').write_text(json.dumps({'analysis':{'variants':ids+['custom/not_started']}}))
        assert not video_inventory(root)
        assert 'No completed attempts' in (dest/'README.md').read_text()


def test_variant_scope_discovery_and_configuration(tmp_path):
    import json
    from vig_dagger.audit import selected_variants
    out=tmp_path/'dagger/other/new_score';out.mkdir(parents=True)
    (out/'settings.json').write_text('{}')
    assert selected_variants(tmp_path)=={'other/new_score'}
    (tmp_path/'config.json').write_text(json.dumps({'rounds':['moe'],'dagger':{'scorers':['a','b','c']}}))
    assert selected_variants(tmp_path)=={'moe/a','moe/b','moe/c'}
