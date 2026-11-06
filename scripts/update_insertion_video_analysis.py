"""Build analysis from exactly the rollouts represented in the expanded gallery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl

from run_insertion_experiment import write_json,alive,REPO

COUNTS=[50,200,500]
EPOCHS=[10,25,50,100,200,300]


def link_file(source,destination):
    source=source.resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert destination.resolve()==source
    else:destination.symlink_to(source)


def build(root):
    import numpy as np
    gallery=root/'full_rollout_videos';out=gallery/'analysis';out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    protocol.update(policy_label='Dense full-resolution policy | expanded video rollouts',
        videos='All 50 physical starts at every checkpoint; each chart uses exactly the rollout shown in the video gallery',
        rollout_selection='Retain 141 original recordings; use 759 freshly scored policy rollouts for cases without original recordings. One rollout per case and checkpoint.',
        calibration='Unchanged checkpoint-specific training reference; fresh rollouts use newly measured flow',
        original_analysis=str(root))
    write_json(out/'protocol.json',protocol);shutil.copy2(root/'cases.json',out/'cases.json')
    cases={r['case_id']:r for r in json.loads((root/'cases.json').read_text())}
    selection=[];fresh_total=0
    for count in COUNTS:
        for epoch in EPOCHS:
            name=f'demos{count}_epoch{epoch:03d}';original=root/'runs'/name;expanded=gallery/'runs'/name;dest=out/'runs'/name
            dest.mkdir(parents=True,exist_ok=True)
            manifest=json.loads((expanded/'videos.json').read_text())
            assert len(manifest)==50 and sorted(r['case_id'] for r in manifest)==list(range(50))
            original_rows={r['case_id']:r for r in json.loads((original/'rollouts.json').read_text())}
            reference=np.sort(np.load(original/'calibration.npz')['flow']);rows=[]
            for video in sorted(manifest,key=lambda r:r['case_id']):
                case_id=video['case_id'];kind=video['kind'];stem=f'{case_id:02d}_{kind}'
                if video['source']=='original recording':
                    record=dict(original_rows[case_id]);source=original;record_file=original/'rollouts.json'
                else:
                    assert video['source']=='fresh policy rollout',video['source']
                    record_file=expanded/'fresh_rollouts'/f'{stem}.json'
                    record=json.loads(record_file.read_text());source=expanded;fresh_total+=1
                assert record['case_id']==case_id and record['kind']==kind==cases[case_id]['kind']
                assert record['success']==video['success']
                assert record['seed']==cases[case_id]['seed']
                trace_path=source/record['trace'];movie=(gallery/video['video']).resolve()
                assert movie.is_file() and movie==(source/record['video']).resolve()
                with np.load(trace_path) as trace:
                    assert np.isfinite(trace['flow']).all()
                    expected=100*np.searchsorted(reference,trace['flow'],side='right')/len(reference)
                    assert np.allclose(expected,trace['percentile'],rtol=0,atol=1e-10)
                    assert len(trace['actions'])==record['steps']
                    assert trace['commanded_paths'].shape[1:]==(17,3)
                relative_trace=f'traces/{stem}.npz';link_file(trace_path,dest/relative_trace)
                record.update(trace=relative_trace,video=os.path.relpath(movie,dest),video_source=video['source'])
                rows.append(record)
                selection.append(dict(demos=count,epoch=epoch,case_id=case_id,kind=kind,source=video['source'],
                    record_file=str(record_file),trace_file=str(trace_path),video_file=str(movie),
                    trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest()))
            for kind in {c['kind'] for c in cases.values()}:assert sum(r['kind']==kind for r in rows)==10
            for filename in ['calibration.npz','calibration.json']:link_file(original/filename,dest/filename)
            write_json(dest/'rollouts.json',rows)
            write_json(dest/'complete.json',dict(rollouts=50,successes=sum(r['success'] for r in rows),finished_at=time.time()))
    assert len(selection)==900 and fresh_total==759
    write_json(out/'rollout_selection.json',selection)
    write_json(out/'validation.json',dict(passed=True,checkpoints=18,rollouts=900,original_recordings=141,fresh_rollouts=fresh_total,
        unique_case_per_checkpoint=True,video_outcomes_match=True,percentiles_match_training_references=True,
        all_16_action_targets=True,validated_at=time.time()))
    subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(out)],cwd=REPO,check=True)
    old=json.loads((root/'summary.json').read_text());new=json.loads((out/'summary.json').read_text())
    old={(r['demos'],r['epoch'],r['case']):r for r in old}
    changes=[dict(demos=r['demos'],epoch=r['epoch'],case=r['case'],
        original_successes=old[(r['demos'],r['epoch'],r['case'])]['successes'],updated_successes=r['successes']) for r in new
        if r['successes']!=old[(r['demos'],r['epoch'],r['case'])]['successes']]
    write_json(out/'changes_from_original.json',changes)
    note='<p>These charts use exactly the 900 episodes in the expanded gallery: 141 original recordings and 759 fresh rollouts. Each starting case is counted once per checkpoint. The original evaluation is preserved separately.</p>'
    page=out/'index.html';content=page.read_text().replace('<h2>Representative videos</h2>','<h2>All rollout videos</h2>')
    page.write_text(content.replace('</body>',note+'<p><a href="../index.html">Expanded video gallery</a> | <a href="rollout_selection.json">Exact rollout selection</a></p></body>'))
    # Preserve the original landing page and figures; direct the main index to
    # the revised figures without mixing old trace files with new summaries.
    archive=root/'original_index_before_video_plot_update.html'
    if not archive.exists():shutil.copy2(root/'index.html',archive)
    prefix='full_rollout_videos/analysis/'
    images=['success_rates.png','flow_percentiles.png','flow_raw.png']+[f'action_{m}_demos{n}.png' for n in COUNTS for m in ['translation','rotation']]
    landing=['<html><meta charset="utf-8"><title>Updated dense insertion analysis</title><body><h1>Dense insertion analysis: expanded video rollouts</h1>',note,
        f'<p><a href="{prefix}index.html">Updated analysis and all video links</a> | <a href="full_rollout_videos/index.html">Video gallery</a> | <a href="original_index_before_video_plot_update.html">Original analysis (archived)</a></p>',
        f'<p><a href="{prefix}success_rates.csv">Updated success and flow CSV</a> | <a href="{prefix}success_rates.pdf">Success plot PDF</a></p>']
    landing += [f'<p><a href="{prefix}{name}"><img src="{prefix}{name}" style="max-width:100%"></a></p>' for name in images]
    (root/'index.html').write_text('\n'.join(landing+['</body></html>']))
    for page in [gallery/'index.html',*sorted((gallery/'runs').glob('*/index.html'))]:
        content=page.read_text().replace('Original summary plots still describe the original 900 evaluations.',
            'Updated summary plots use exactly the episodes shown in this gallery.')
        url=os.path.relpath(out/'index.html',page.parent)
        page.write_text(content.replace('</body>',f'<p><a href="{url}">Updated success, flow, and action-distribution plots</a></p></body>'))
    write_json(out/'complete.json',dict(rollouts=900,plots=9,finished_at=time.time()))
    write_json(root/'analysis_selection.json',dict(analysis_root=str(out),reason='Plots matched to expanded video gallery',complete=True,updated_at=time.time()))
    return dict(analysis_root=str(out),rollouts=900,plots=9,changed_success_groups=len(changes))


def run(root):
    gallery=root/'full_rollout_videos';status=gallery/'plot_refresh.json'
    while True:
        pipeline=json.loads((gallery/'pipeline.json').read_text())
        if pipeline['phase']=='complete':break
        if pipeline['phase']=='failed':raise RuntimeError(f'Video generation failed: {pipeline}')
        process=json.loads((gallery/'coordinator_process.json').read_text())
        if not alive(process['pid'],'expand_insertion_videos.py'):raise RuntimeError('Video coordinator exited before completion')
        count=sum(len(json.loads(p.read_text())) for p in (gallery/'runs').glob('*/videos.json'))
        write_json(status,dict(phase='waiting_for_videos',videos=count,total=900,pid=os.getpid(),updated_at=time.time()))
        time.sleep(15)
    write_json(status,dict(phase='validating_and_plotting',pid=os.getpid(),updated_at=time.time()))
    result=build(root)
    write_json(status,dict(phase='complete',**result,finished_at=time.time()))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'full_rollout_videos/plot_refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a.root)
        except BaseException as exc:
            write_json(a.root/'full_rollout_videos/plot_refresh.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
