"""Read-only, resumable DAgger audit; see docs/DAGGER_ANALYSIS.md."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
import h5py
import numpy as np
from vig_dagger import analysis, stages
from vig_dagger.dataset import FPS


def selected_variants(root):
    """Explicit scope, configured grid, or discovered settings; never a fixed count."""
    path = root/'config.json'
    cfg = json.loads(path.read_text()) if path.exists() else {}
    explicit = cfg.get('analysis', {}).get('variants')
    if explicit is not None:
        if not explicit or len(set(explicit)) != len(explicit):
            raise ValueError('analysis.variants must contain unique variant IDs')
        for name in explicit:
            parts = name.split('/')
            if len(parts) != 2 or any(p in ('', '.', '..') for p in parts):
                raise ValueError(f'Invalid variant ID: {name}')
        return set(explicit)
    if cfg.get('rounds') and cfg.get('dagger', {}).get('scorers'):
        return {f'{group}/{scorer}' for group in cfg['rounds'] for scorer in cfg['dagger']['scorers']}
    return {f'{p.parent.parent.name}/{p.parent.name}' for p in (root/'dagger').glob('*/*/settings.json')}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def wilson(k, n):
    if not n:
        return None
    z = 1.95996398454
    p = k / n
    c = (p + z*z/(2*n))/(1+z*z/n)
    d = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(c-d), float(c+d)]


def enriched(out):
    eps, segs, scores, steps = analysis.load_episodes(out)
    summaries = {}
    for p in sorted((out/'episodes').glob('*.jsonl')):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                summaries[r['seed']] = r
    by_seed = {e['seed']: e for e in eps}
    stage_rows = []
    for i, seed in enumerate(steps['seed']):
        e, s = by_seed[seed], summaries[seed]
        c, st = steps['controller'][i], steps['stage'][i]
        own = [t for t in segs if t['seed'] == seed]
        e.update(expert_failure=s.get('expert_failure'), seconds=s.get('seconds'), finished_at=s.get('finished_at'),
                 file=s.get('file'), traj=s.get('traj'),
                 takeovers_per_100_policy_steps=100*e['takeovers']/(e['steps']-e['expert_steps']) if e['steps']>e['expert_steps'] else None,
                 return_fraction=e['returns']/e['takeovers'] if e['takeovers'] else None,
                 takeover_duration_distribution=json.dumps(analysis.distribution(t['length']/FPS for t in own)),
                 mean_takeover_seconds=float(np.mean([t['length']/FPS for t in own])) if own else None,
                 repeated_takeovers=sum(any(0 <= t['start']-r['end'] <= 60 for r in own if r['ended']=='returned') for t in own))
        for t in own:
            t['seconds'] = t['length']/FPS
            t['outcome'] = 'returned' if t['ended']=='returned' else ('success_under_expert' if e['success'] else e['reason'])
        for sid, name in enumerate(stages.STAGES):
            mask = st == sid
            n, ex = int(mask.sum()), int(((c == 1)&mask).sum())
            count = sum(t['stage']==name for t in own)
            stage_rows.append(dict(seed=seed, success=e['success'], stage=name, steps=n, expert_steps=ex,
                                   expert_share=ex/n if n else None, takeovers=count,
                                   takeovers_per_100_policy_steps=100*count/(n-ex) if n>ex else None))
    return eps, segs, scores, steps, stage_rows


def box(ax, groups, labels, title):
    data = [[float(x) for x in g if x is not None and np.isfinite(x)] for g in groups]
    if any(data):
        ax.boxplot([g or [np.nan] for g in data], tick_labels=labels, showfliers=True)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30, labelsize=7)


def plots(dest, eps, segs, rows, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'figure.dpi': 130, 'savefig.dpi': 160})
    def save(fig, name):
        fig.savefig(dest/f'{name}.png', bbox_inches='tight'); plt.close(fig)
    def horizontal(ax, values, labels, title, xlabel, percent=False):
        clean=[[float(x) for x in v if x is not None and np.isfinite(x)] for v in values]
        if any(clean):
            ax.boxplot([v or [np.nan] for v in clean], orientation='horizontal', tick_labels=labels,
                       showmeans=True, meanprops={'marker':'D','markersize':4,'markerfacecolor':'#d95f02','markeredgecolor':'#d95f02'},
                       flierprops={'markersize':2,'alpha':.35}, medianprops={'color':'#204a87','linewidth':2})
        else:
            ax.set_yticks(range(1,len(labels)+1), labels)
            ax.text(.5,.5,'No observations',transform=ax.transAxes,ha='center')
        ax.invert_yaxis(); ax.set(title=title,xlabel=xlabel)
        ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
        if percent: ax.xaxis.set_major_formatter(PercentFormatter(1))
    groups=[[e for e in eps if name=='all' or e['success']==(name=='success')] for name in analysis.GROUPS]
    names=[f'{name} (n={len(g)})' for name,g in zip(analysis.GROUPS,groups)]
    fields=[('expert_share','Expert-controlled portion per attempt','Expert steps / attempt steps',True),
            ('takeovers','Takeovers per attempt','Number of takeovers',False),
            ('returns','Returns per attempt','Number of returns',False),
            ('takeovers_per_100_policy_steps','Takeover frequency','Takeovers per 100 policy steps',False),
            ('return_fraction','Fraction of takeovers returning to policy','Returns / takeovers',True),
            ('mean_takeover_seconds','Mean intervention duration per attempt','Simulated seconds',False)]
    for key,title,xlabel,percent in fields:
        fig,ax=plt.subplots(figsize=(10,3.6),layout='constrained')
        horizontal(ax,[[e[key] for e in g] for g in groups],names,title+'\nDiamond = mean; line = median',xlabel,percent)
        save(fig,'episode_'+key)
    # Separate stage panels avoid tiny labels and ambiguous missing-stage values.
    for group in analysis.GROUPS:
        rr=[r for r in rows if group=='all' or r['success']==(group=='success')]
        for key,title,xlabel,pct in [('takeovers','Takeover starts per attempt','Count',False),
                                     ('expert_share','Expert portion within stage','Expert steps / stage steps',True),
                                     ('takeovers_per_100_policy_steps','Takeover frequency within stage','Takeovers per 100 policy steps',False)]:
            fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
            horizontal(ax,[[r[key] for r in rr if r['stage']==stage] for stage in stages.STAGES],
                       stages.STAGES,f'{group.capitalize()}: {title}',xlabel,pct)
            save(fig,f'stage_{group}_{key}')
    for key,title in [('takeovers','Pooled takeover starts by stage'),('expert_share','Pooled expert portion by stage')]:
        vals=[]
        for stage in stages.STAGES:
            rr=[r for r in rows if r['stage']==stage]; n=sum(r['steps'] for r in rr)
            vals.append(sum(r['takeovers'] for r in rr) if key=='takeovers' else sum(r['expert_steps'] for r in rr)/n if n else np.nan)
        fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
        ax.barh(stages.STAGES,vals,color='#4477aa'); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x',alpha=.2)
        if key=='expert_share': ax.xaxis.set_major_formatter(PercentFormatter(1))
        save(fig,'pooled_stage_'+key)
    # Paginate per-attempt distributions; keep real seed labels visible.
    seeds=[e['seed'] for e in eps]
    for start in range(0,len(seeds),20):
        batch=seeds[start:start+20]; fig,ax=plt.subplots(figsize=(11,9),layout='constrained')
        labels=[f"{s} ({'success' if next(e for e in eps if e['seed']==s)['success'] else 'FAILED'})" for s in batch]
        horizontal(ax,[[t['seconds'] for t in segs if t['seed']==s] for s in batch],labels,
                   f'Intervention durations within each attempt — page {start//20+1}', 'Simulated seconds')
        save(fig,f'durations_page_{start//20+1:02d}')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for outcome in sorted({t['outcome'] for t in segs}):
        v=np.sort([t['seconds'] for t in segs if t['outcome']==outcome])
        if len(v): ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{outcome} (n={len(v)})')
    if segs: ax.legend()
    ax.set(title='Pooled intervention duration by outcome',xlabel='Simulated seconds',ylabel='Cumulative fraction'); ax.grid(alpha=.2)
    save(fig,'duration_pooled')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for who,arrays in scores.items():
        v=np.sort(np.concatenate(arrays)) if arrays else []
        if len(v): ax.plot(v,np.arange(1,len(v)+1)/len(v),label=f'{who} (n={len(v)})')
    ax.set(title='Decision-score distribution by controller',xlabel='Actual decision score',ylabel='Cumulative fraction'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'scores')
    failures=[e for e in eps if not e['success']]
    reasons=sorted({e.get('expert_failure') or e['reason'] for e in failures})
    fig,ax=plt.subplots(figsize=(12,max(3,len(reasons)*.5+1.5)),layout='constrained')
    if reasons:
        ax.barh(reasons,[sum((e.get('expert_failure') or e['reason'])==r for e in failures) for r in reasons],color='#cc6677'); ax.invert_yaxis()
    else: ax.text(.5,.5,'No failed attempts',transform=ax.transAxes,ha='center')
    ax.set(title='Failed attempts: recorded reasons',xlabel='Number of attempts'); save(fig,'failure_reasons')
    explicit=[e for e in failures if e.get('expert_failure')]
    reasons=sorted({e['expert_failure'] for e in explicit})
    fig,ax=plt.subplots(figsize=(13,max(4,len(reasons)*.55+2)),layout='constrained')
    if reasons:
        a=np.array([[sum(e['final_stage']==s and e['expert_failure']==r for e in explicit) for s in stages.STAGES] for r in reasons])
        im=ax.imshow(a,aspect='auto',cmap='Blues'); fig.colorbar(im,ax=ax,label='Attempts',shrink=.7)
        ax.set_yticks(range(len(reasons)),reasons); ax.set_xticks(range(15),[s.replace(' ','\n') for s in stages.STAGES],fontsize=9)
        for y,x in zip(*np.nonzero(a)): ax.text(x,y,str(a[y,x]),ha='center',va='center',color='black')
    else: ax.text(.5,.5,'No explicit expert failures',transform=ax.transAxes,ha='center')
    ax.set_title('Expert failures: final stage × planner reason'); save(fig,'failure_stages')
    lines=['# Detailed collection figures','','Each chart has one metric. Diamond = mean; line = median. Absent stages have no portion/frequency estimate.','']
    for key,_,_,_ in fields: lines += [f'![{key}](episode_{key}.png)','']
    for image in ['pooled_stage_takeovers','pooled_stage_expert_share','duration_pooled','scores','failure_reasons','failure_stages']:
        lines += [f'![{image}]({image}.png)','']
    for group in analysis.GROUPS:
        lines += [f'## {group.capitalize()} attempts by stage','']
        for key in ['takeovers','expert_share','takeovers_per_100_policy_steps']:
            lines += [f'![{group} {key}](stage_{group}_{key}.png)','']
    lines += ['## Intervention duration within individual attempts','']
    for start in range(0,len(seeds),20): lines += [f'![Duration page {start//20+1}](durations_page_{start//20+1:02d}.png)','']
    (dest/'FIGURES.md').write_text('\n'.join(lines))

def video_path(dest, out, episode):
    outcome = 'success' if episode['success'] else 'failed'
    return dest/'videos'/out.parent.name/out.name/outcome/f"{episode['seed']}.mp4"


def interleave_outcomes(episodes):
    """Give successes and failures alternating turns within each setting."""
    success = sorted((e for e in episodes if e['success']), key=lambda e:e['seed'])
    failed = sorted((e for e in episodes if not e['success']), key=lambda e:e['seed'])
    return [group[i] for i in range(max(len(success),len(failed))) for group in [success,failed] if i<len(group)]


def ensure_clips(target, data, count):
    """Resume missing event clips independently of the full MP4."""
    events = [(label, int(t)) for label,key in [('takeover','take_steps'),('return','return_steps')] for t in data[key]]
    for i,(label,t) in enumerate(events):
        clip = target.with_name(f'{target.stem}_{i:03d}_{label}.mp4')
        start = max(0,t/FPS-1)
        duration = min(count/FPS,t/FPS+1)-start
        if duration <= 0 or (clip.exists() and clip.stat().st_size > 0):
            continue
        partial = clip.with_suffix('.partial.mp4')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(target),'-t',str(duration),'-an','-c:v','libx264','-threads','1','-preset','veryfast',str(partial)],check=True)
        partial.replace(clip)
    target.with_suffix('.complete.json').write_text(json.dumps({'full_video':target.name,'events':len(events)}))


def render(out, e, data, target, settings):
    """Read cameras from completed recorder files; reject locks instead of bypassing HDF5 safety."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not e['file']:
        return 'unavailable: recording discarded before retention change'
    if target.exists():
        ensure_clips(target, data, e['steps'])
        return 'ready'
    try:
        f = h5py.File(out/e['file'], 'r')
    except (OSError, BlockingIOError):
        return 'pending: recording file still open or unavailable'
    with f:
        key = f"data/demo_{e['traj']}"
        if key not in f:
            return 'pending: trajectory not flushed'
        g = f[key]
        if 'obs/rgb_camera' not in g:
            return 'unavailable: no recorded RGB'
        cams = [(k, v) for k,v in g['obs/rgb_camera'].items() if isinstance(v,h5py.Dataset)]
        if not cams: return 'unavailable: no recorded RGB'
        count = e['steps']
        if any(len(v)<count for _,v in cams):
            return 'pending: incomplete camera recording'
        width = max(960, 320*len(cams)); height = 640
        fig, ax = plt.subplots(figsize=(width/100, 2.8), dpi=100)
        canvas = FigureCanvasAgg(fig)
        score = data['score']; c = data['controller']; st = data['stage']
        ax.plot(np.arange(len(score))/FPS, score, lw=1)
        ax.axhline(settings['take_threshold'], color='red', ls='--', label='take threshold')
        ax.axhline(settings['return_threshold'], color='green', ls='--', label='return threshold')
        events = []
        for start,length in analysis.segments(c):
            ax.axvspan(start/FPS,(start+length)/FPS,color='orange',alpha=.2)
        # Original event arrays distinguish a return at the terminal observation.
        for label, indices, color in [('takeover',data['take_steps'],'red'),('return',data['return_steps'],'green')]:
            for t in indices:
                ax.axvline(t/FPS,color=color,alpha=.5,lw=.7,label=label if t==indices[0] else None)
                events.append((label,int(t)))
        for t in np.flatnonzero(np.diff(st)!=0)+1:
            ax.axvline(t/FPS,color='gray',alpha=.25,lw=.5)
        ax.set(xlabel='Simulated time (seconds); orange = expert', ylabel=settings.get('scorer_type','score'))
        ax.legend(loc='upper right',fontsize=7); fig.tight_layout()
        canvas.draw()
        chart = np.asarray(canvas.buffer_rgba())[...,:3].copy()
        target.parent.mkdir(parents=True,exist_ok=True)
        temp = target.with_suffix('.partial.mp4')
        cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
               '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            cumulative = np.cumsum(c==1)
            for t in range(count):
                frame = np.full((height,width,3),245,np.uint8)
                title = f"{out.parent.name}/{out.name} seed={e['seed']} {'SUCCESS' if e['success'] else 'FAILED'}: {e.get('expert_failure') or e['reason']}"
                cv2.putText(frame,title,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
                cv2.putText(frame,f"{t/FPS:.2f}s | {stages.STAGES[int(st[t])]} | {'EXPERT' if c[t] else 'POLICY'} | expert {cumulative[t]/(t+1):.1%}",(12,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
                cell = width//len(cams)
                for j,(name,ds) in enumerate(cams):
                    rgb = np.asarray(ds[t])[...,:3]
                    scale = min(cell/rgb.shape[1],285/rgb.shape[0])
                    image = cv2.resize(rgb,(int(rgb.shape[1]*scale),int(rgb.shape[0]*scale)))
                    x = j*cell+(cell-image.shape[1])//2
                    frame[75:75+image.shape[0],x:x+image.shape[1]] = image
                    cv2.putText(frame,name,(j*cell+8,72),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
                frame[-280:] = chart
                px = int(ax.transData.transform((t/FPS,0))[0])
                cv2.line(frame,(px,height-280+int(280-ax.bbox.y1)),(px,height-int(ax.bbox.y0)),(0,0,0),2)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode()
            if proc.wait(): raise RuntimeError(err)
            temp.replace(target)
        finally:
            plt.close(fig)
            if proc.poll() is None: proc.kill(); proc.wait()
        ensure_clips(target, data, count)
        return 'ready'


def snapshot(root, video_limit=None):
    dest = root/'results/dagger_audit'; dest.mkdir(parents=True,exist_ok=True)
    report = ['# DAgger collection audit', '', 'Provisional snapshot while collection continues. All completed attempts are included; failures are excluded from retraining.', '', '[Live video index](VIDEOS.md) · [Failed attempts and exact reasons](FAILED_ATTEMPTS.md)', '',
              f'[Metric definitions and stage rules]({os.path.relpath(Path(__file__).resolve().parents[2]/"docs/DAGGER_ANALYSIS.md",dest)})', '',
              'Stages: '+ ' → '.join(stages.STAGES), '',
              'Final stage and furthest stage are distinct. Expert-failure plots require an explicit recorded expert failure. Video unavailable means historical RGB was discarded; pending means the writer may still own the file.', '']
    index, comparison = [], []
    complete = True
    expected = selected_variants(root)
    for out in sorted((root/'dagger').glob('*/*')):
        if f'{out.parent.name}/{out.name}' not in expected:
            continue
        rendered = 0  # Give every variant its own rendering budget each pass.
        if not (out/'settings.json').exists(): continue
        name = out.parent.name+'/'+out.name
        settings = json.loads((out/'settings.json').read_text())
        try:
            eps,segs,scores,steps,rows = enriched(out)
        except (OSError,BlockingIOError) as exc:
            print(f'[audit] {name}: retry next snapshot: {exc}',flush=True); complete=False; continue
        if not eps: complete=False; continue
        local = dest/out.parent.name/out.name; local.mkdir(parents=True,exist_ok=True)
        m = analysis.metrics(eps,segs); n=len(eps); k=sum(e['success'] for e in eps)
        complete &= k>=settings['target']
        ci=wilson(k,n)
        m['collection']['success_rate_95ci']=ci
        m['collection']['summed_attempt_seconds']=sum(e['seconds'] or 0 for e in eps)
        m['collection']['elapsed_span_seconds']=max(e['finished_at'] for e in eps)-min(e['finished_at']-(e['seconds'] or 0) for e in eps)
        m['collection']['attempts_to_quota']=next((i+1 for i in range(n) if sum(x['success'] for x in sorted(eps,key=lambda x:x['finished_at'])[:i+1])>=settings['target']),None)
        m['decision_event_scores'] = {event: analysis.distribution(t['trigger_score'] for t in segs) if event=='takeover' else analysis.distribution(steps['page'][i]['score'][t['end']] for t in segs if t['ended']=='returned' for i,seed in enumerate(steps['seed']) if seed==t['seed'] and t['end']<len(steps['page'][i]['score'])) for event in ['takeover','return']}
        m['additional_distributions']={group:{key:analysis.distribution(e[key] for e in eps if group=='all' or e['success']==(group=='success')) for key in ['takeovers_per_100_policy_steps','return_fraction','mean_takeover_seconds','repeated_takeovers','seconds']} for group in analysis.GROUPS}
        (local/'metrics.json').write_text(json.dumps(m,indent=2))
        write_csv(local/'episodes.csv',[{key:v for key,v in e.items() if not isinstance(v,list)} for e in eps])
        write_csv(local/'takeovers.csv',segs); write_csv(local/'stages.csv',rows)
        plots(local,eps,segs,rows,scores)
        (local/'snapshot.json').write_text(json.dumps({'generated_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'attempts':len(eps),'successes':k}))
        prefix=f'{out.parent.name}/{out.name}'
        report += [f'## {name}', '', f"Success: **{k}/{n} ({k/n:.1%})**, 95% Wilson interval {ci[0]:.1%}–{ci[1]:.1%}. Training selection: {sum(e['used_for_training'] for e in eps)} successes. Takeover threshold {settings['take_threshold']:.6g}; return {settings['return_threshold']:.6g}.", '',
                   '| Group | Attempts | Pooled expert portion | Mean / median expert portion | Mean / median takeovers | Mean / median returns |', '|---|---:|---:|---:|---:|---:|']
        for group in analysis.GROUPS:
            mm=m[group]
            if not mm['episodes']: continue
            def pair(key):
                d=mm['per_episode'][key]; return f"{d['mean']:.3f} / {d['median']:.3f}"
            report += [f"|{group}|{mm['episodes']}|{mm['pooled']['expert_share']:.1%}|{pair('expert_share')}|{pair('takeovers')}|{pair('returns')}|"]
        report += ['',f"[All metrics]({prefix}/metrics.json) · [Attempts]({prefix}/episodes.csv) · [Interventions]({prefix}/takeovers.csv) · [Per-attempt stage metrics]({prefix}/stages.csv)",'']
        report += [f'[Detailed figures: all outcomes, stages and individual attempts]({prefix}/FIGURES.md)', '']
        for image in ['episode_expert_share','episode_takeovers','pooled_stage_expert_share','failure_reasons']:
            report += [f'![{image}]({prefix}/{image}.png)', '']
        comparison.append(dict(variant=name, attempts=n, successes=k, success_rate=k/n, expert_share=m['all']['pooled']['expert_share']))
        for i,e in enumerate(eps):
            target=video_path(dest,out,e)
            status='pending'
            if not e['file']: status='unavailable: recording discarded before retention change'
            elif target.exists(): status='ready'
            elif video_limit is None or rendered<video_limit:
                # load_episodes preserves recorder order; eps is sorted by seed.
                idx=steps['seed'].index(e['seed'])
                data={key:steps[key][idx] for key in ['score','controller','stage']}
                with h5py.File(out/'attempts'/f"worker_{e['worker']:02d}.h5",'r') as f:
                    g=f[f"seed_{e['seed']}"]; data.update(take_steps=g['take_steps'][:],return_steps=g['return_steps'][:])
                status=render(out,e,data,target,settings)
                if status=='ready':
                    video_inventory(root)
                    rendered+=1; print(f'[audit] video {name}/{e["seed"]}',flush=True)
            if status=='pending' or status.startswith('pending:'): complete=False
            index.append(dict(variant=name,seed=e['seed'],success=e['success'],used_for_training=e['used_for_training'],status=status,video=str(target.relative_to(dest)) if status=='ready' else ''))
        print(f'[audit] {name}: {k}/{n} successes',flush=True)
    compact_comparisons(dest)
    report[2:2] = ['[Compact plots comparing all DAgger settings](COMPARISONS.md)', '']
    if comparison:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,1,figsize=(12,11),layout='constrained')
        for ax,key in zip(axes,['success_rate','expert_share']):
            ax.barh([r['variant'] for r in comparison],[r[key] for r in comparison]); ax.invert_yaxis(); ax.set_title(key); ax.grid(axis='x',alpha=.2)
            from matplotlib.ticker import PercentFormatter
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        fig.tight_layout(); fig.savefig(dest/'comparison.png'); plt.close(fig)
        report[2:2] = ['![Variant comparisons](comparison.png)', '']
    write_csv(dest/'comparison.csv',comparison)  # Live video inventory has its own stable schema.
    report += ['## Every-attempt videos', '', '| Variant | Seed | Outcome | Training | Video and handoff clips |', '|---|---:|---|---|---|']
    for r in index:
        link=f"[MP4]({r['video']})" if r['video'] else r['status']
        if r['video']:
            clips=sorted((dest/r['video']).parent.glob(f"{r['seed']}_*.mp4"))
            link += ' ' + ' '.join(f'[clip {i+1}]({p.relative_to(dest)})' for i,p in enumerate(clips))
        report += [f"|{r['variant']}|{r['seed']}|{'success' if r['success'] else 'failed'}|{r['used_for_training']}|{link}|"]
    temp=dest/'REPORT.tmp'; temp.write_text('\n'.join(report)+'\n'); temp.replace(dest/'REPORT.md')
    return complete and bool(expected) and {r['variant'] for r in comparison} == expected


def video_inventory(root):
    """Cheap live inventory independent of plot/report generation."""
    dest=root/'results/dagger_audit'
    rows=[]; complete=True; variants=0
    expected = selected_variants(root)
    for out in sorted((root/'dagger').glob('*/*')):
        if f'{out.parent.name}/{out.name}' not in expected:
            continue
        if not (out/'settings.json').exists(): continue
        variants+=1
        summaries={}
        for path in (out/'episodes').glob('*.jsonl'):
            for line in path.read_text().splitlines():
                if line.strip():
                    e=json.loads(line); summaries[e['seed']]=e
        target_count=json.loads((out/'settings.json').read_text())['target']
        complete &= sum(e['success'] for e in summaries.values())>=target_count
        selected=set([e['seed'] for e in sorted(summaries.values(),key=lambda e:e['finished_at']) if e['success']][:target_count])
        for e in sorted(summaries.values(),key=lambda e:e['seed']):
            path=video_path(dest,out,e)
            ready=path.exists()
            status='ready' if ready else ('pending' if e.get('file') else 'unavailable: historical RGB discarded')
            if e.get('file') and not ready: complete=False
            rows.append(dict(variant=out.parent.name+'/'+out.name,seed=e['seed'],outcome='SUCCESS' if e['success'] else 'FAILED',
                             reason=e.get('expert_failure') or e['reason'],used_for_training=e['seed'] in selected,
                             status=status,video=str(path.relative_to(dest)) if ready else ''))
    write_csv(dest/'video_index.csv',rows)
    failed=[r for r in rows if r['outcome']=='FAILED'];write_csv(dest/'failed_attempts.csv',failed)
    def document(title, items):
        lines=[f'# {title}','','Updated live during rendering. FAILED means the completed attempt did not succeed; the reason distinguishes expert failures and time limits.', '',
               '| Setting | Seed | Outcome | Reason | Video |','|---|---:|---|---|---|']
        for r in items:
            link=f"[MP4]({r['video']})" if r['video'] else r['status']
            if r['video']:
                for i,p in enumerate(sorted((dest/r['video']).parent.glob(f"{r['seed']}_*.mp4"))):
                    link+=f' · [clip {i+1}]({p.relative_to(dest)})'
            lines.append(f"|{r['variant']}|{r['seed']}|**{r['outcome']}**|{r['reason']}|{link}|")
        return '\n'.join(lines)+'\n'
    for filename,title,items in [('VIDEOS.md','All DAgger attempt videos',rows),('FAILED_ATTEMPTS.md','Failed DAgger attempts',failed),('SUCCESSFUL_ATTEMPTS.md','Successful DAgger attempts',[r for r in rows if r['outcome']=='SUCCESS'])]:
        tmp=dest/(filename+'.tmp');tmp.write_text(document(title,items));tmp.replace(dest/filename)
    from vig_dagger.audit_index import write_overview
    write_overview(root, rows)
    return complete and bool(expected) and variants == len(expected)


def render_available(root, limit=None):
    """Round-robin across variants, alternating successful and failed attempts."""
    dest=root/'results/dagger_audit'; pending=[]
    expected = selected_variants(root)
    for out in sorted((root/'dagger').glob('*/*')):
        if f'{out.parent.name}/{out.name}' not in expected:
            continue
        if not (out/'settings.json').exists(): continue
        settings=json.loads((out/'settings.json').read_text())
        summaries={}
        for p in (out/'episodes').glob('*.jsonl'):
            for line in p.read_text().splitlines():
                if line.strip():
                    e=json.loads(line);summaries[e['seed']]=e
        candidates=[e for e in summaries.values() if e.get('file') and not video_path(dest,out,e).with_suffix('.complete.json').exists()]
        candidates=interleave_outcomes(candidates)
        if limit is not None: candidates=candidates[:limit]
        pending.append((out,settings,candidates))
    for index in range(max((len(es) for _,_,es in pending),default=0)):
        for out,settings,es in pending:
            if index>=len(es): continue
            e=es[index];target=video_path(dest,out,e)
            try:
                with h5py.File(out/'attempts'/f"worker_{e['worker']:02d}.h5",'r') as f:
                    g=f[f"seed_{e['seed']}"]
                    data={k:g[k][:] for k in ['score','controller','take_steps','return_steps']}
                    state={k:g[k][:] for k in stages.STATE_KEYS}
                    data['stage']=stages.label(state,e['success'])[0][:e['steps']]
                status=render(out,e,data,target,settings)
                print(f"[audit] {out.parent.name}/{out.name}/{e['seed']} {'SUCCESS' if e['success'] else 'FAILED'}: {status}",flush=True)
                video_inventory(root)
            except (OSError,BlockingIOError) as exc:
                print(f"[audit] recording pending {target}: {exc}",flush=True)
    done = video_inventory(root)
    return done and all(video_path(dest,out,e).with_suffix('.complete.json').exists() for out,_,es in pending for e in es)


def compact_comparisons(dest):
    """Portrait comparison plots: every setting on one shared horizontal scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    files=sorted(dest.glob('*/*/episodes.csv'))
    tables={}
    for f in files:
        with f.open() as stream:
            tables[f.parent.parent.name+'/'+f.parent.name]=list(csv.DictReader(stream))
    expected = selected_variants(dest.parents[1])
    order = sorted(k for k in tables if not expected or k in expected)
    if not order:
        (dest/'COMPARISONS.md').write_text('# Variant comparisons\n\nNo completed attempts available.\n')
        return
    metrics=[('expert_share','Expert-controlled portion','Expert-controlled steps (% of attempt)',True),
             ('takeovers','Takeovers per attempt','Count',False),
             ('returns','Returns per attempt','Count',False),
             ('takeovers_per_100_policy_steps','Takeover frequency','Takeovers per 100 policy steps',False),
             ('return_fraction','Fraction returning to policy','Returns / takeovers',True),
             ('mean_takeover_seconds','Mean intervention duration per attempt','Simulated seconds',False)]
    lines=['# Compact comparisons across all DAgger settings','','Every row is one setting. Boxes show the middle 50% of attempts; blue line = median; orange diamond = mean. Each attempt receives equal weight.','']
    for group in ['all','success','failed']:
        lines += [f'## {group.capitalize()} attempts','']
        for key,title,xlabel,percent in metrics:
            values=[];labels=[]
            for name in order:
                rows=[r for r in tables[name] if group=='all' or (r['success']=='True')==(group=='success')]
                x=[float(r[key]) for r in rows if r.get(key) not in [None,'','None'] and np.isfinite(float(r[key]))]
                values.append(x or [np.nan])
                family,variant=name.split('/')
                labels.append(f"{ {'moe': 'MoE', 'per_resolution': 'Per-res'}.get(family, family) } · {variant.replace('_',' ').upper()}  (n={len(x)})")
            fig,ax=plt.subplots(figsize=(7.2,max(3.8,1.8+0.5*len(order))),layout='constrained')
            ax.boxplot(values,orientation='horizontal',tick_labels=labels,showmeans=True,widths=.55,
                       medianprops={'color':'#204a87','linewidth':1.8},
                       meanprops={'marker':'D','markersize':3.5,'markerfacecolor':'#d95f02','markeredgecolor':'#d95f02'},
                       flierprops={'markersize':2,'alpha':.3})
            ax.invert_yaxis();ax.tick_params(labelsize=9)
            ax.set_title(f'{title}\n{group.capitalize()} attempts — all settings',fontsize=12,pad=12)
            ax.set_xlabel(xlabel + '\nThresholds: see variant settings · n = valid attempts',fontsize=9)
            ax.set_ylabel('DAgger setting\n(driver · score · takeover threshold)',fontsize=9)
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            boundary=sum(k.startswith('moe/') for k in order)
            if 0<boundary<len(order):ax.axhline(boundary+.5,color='#aaaaaa',ls='--',lw=.8)
            if percent:
                ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlim(-.02,1.02)
            stem=f'compare_{key}_{group}'
            fig.savefig(dest/f'{stem}.png',dpi=180,bbox_inches='tight')
            fig.savefig(dest/f'{stem}.pdf',bbox_inches='tight');plt.close(fig)
            lines += [f'![{title}]({stem}.png)',f'[PDF]({stem}.pdf)','']
    (dest/'COMPARISONS.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--watch',action='store_true'); p.add_argument('--video-limit',type=int,help='Maximum new videos per variant per pass')
    args=p.parse_args()
    import fcntl
    dest=args.work_dir/'results/dagger_audit'; dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.audit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done=snapshot(args.work_dir,0)
                if args.video_limit != 0:
                    done=render_available(args.work_dir,args.video_limit)
            except (OSError,BlockingIOError) as exc:
                if not args.watch: raise
                print(f'[audit] transient I/O; will retry: {exc}',flush=True); done=False
            if done:
                snapshot(args.work_dir,0)  # Refresh metrics after the final recordings arrive.
                video_inventory(args.work_dir)
            if done or not args.watch: break
            time.sleep(120)

if __name__=='__main__': main()
