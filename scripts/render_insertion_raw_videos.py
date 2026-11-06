"""Render raw score axes over the identical recorded rollout footage."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw
from run_insertion_experiment import write_json
from vig_dagger.insertion_analysis import write_checkpoint_gallery

SCORES=[('flow','Flow','#63c4ff'),('vig_original','VIG','#a9d879'),('vig','VIG (chunk mean)','#e5a3ff')]


def limits_for(run,rows):
    from matplotlib.ticker import MaxNLocator
    maxima={key:0. for key,_,_ in SCORES}
    for row in rows:
        with np.load(run/row['trace']) as trace:
            for key in maxima:
                assert np.isfinite(trace[key]).all() and np.min(trace[key])>=0
                maxima[key]=max(maxima[key],float(np.max(trace[key])))
    return {key:MaxNLocator(nbins=5).tick_values(0,(value*1.05 if value>0 else 1e-6)).tolist() for key,value in maxima.items()}


def render(source,target,trace,ticks,steps,*,value_label='raw score',axis_label='raw',scores=None):
    scores=SCORES if scores is None else scores
    source_height,width=840,512
    height=300+180*len(scores)
    reader=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(source),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    temp=target.with_suffix('.partial.mp4')
    writer=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-g','10','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    count=0
    try:
        while True:
            data=reader.stdout.read(source_height*width*3)
            if not data:break
            assert len(data)==source_height*width*3
            source_im=Image.fromarray(np.frombuffer(data,dtype=np.uint8).reshape(source_height,width,3).copy())
            im=Image.new('RGB',(width,height),(22,26,33));im.paste(source_im.crop((0,0,width,302)),(0,0));draw=ImageDraw.Draw(im)
            idx=max(0,bisect_right(trace['score_steps'],count)-1)
            for panel,(key,label,color) in enumerate(scores):
                top=320+180*panel;x0=65;y0=top+20;w=435;h=120;upper=ticks[key][-1]
                draw.text((6,top-16),f'{label}: {trace[key][idx]:.6g} | {value_label}',fill='white')
                draw.text((x0,top+2),label+' ('+axis_label+')',fill='white')
                for tick in ticks[key]:
                    y=y0+h*(1-tick/upper)
                    draw.line((x0,y,500,y),fill=(58,64,74))
                    text=f'{tick:.3g}';box=draw.textbbox((0,0),text)
                    draw.text((x0-8-(box[2]-box[0]),y-6),text,fill='white')
                draw.line((x0,y0,x0,y0+h,500,y0+h),fill=(150,155,164))
                points=[(x0+w*trace['score_steps'][j]/max(1,steps-1),y0+h*(1-trace[key][j]/upper)) for j in range(idx+1)]
                if len(points)>1:draw.line(points,fill=color,width=2)
                if points:
                    x,y=points[-1];draw.ellipse((x-2,y-2,x+2,y+2),fill='white')
            writer.stdin.write(np.asarray(im).tobytes());count+=1
    finally:
        reader.stdout.close();decode_error=reader.stderr.read().decode();decode_code=reader.wait()
        writer.stdin.close();encode_error=writer.stderr.read().decode();encode_code=writer.wait()
    assert decode_code==0,decode_error
    assert encode_code==0,encode_error
    assert count==steps,(count,steps)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(temp),'-f','null','-'],capture_output=True,check=True)
    temp.replace(target)


def main(run,preview=None,scale='rollout',case_id=None):
    rows=json.loads((run/'rollouts.json').read_text())
    shared_ticks=limits_for(run,rows) if scale=='checkpoint' else None
    if preview:
        row=next(r for r in rows if r['case_id']==case_id) if case_id is not None else rows[0]
        ticks=shared_ticks if shared_ticks is not None else limits_for(run,[row])
        with np.load(run/row['trace']) as trace:render(run/row['video'],preview,trace,ticks,row['steps'])
        write_json(preview.with_suffix('.json'),dict(case_id=row['case_id'],source=str(run/row['video']),ticks=ticks,scope=scale))
        return
    assert len(rows)==50 and (run/'complete.json').exists()
    marker=run/('raw_rollout_scale_complete.json' if scale=='rollout' else 'raw_complete.json')
    if marker.exists():return
    out=run/('videos_raw_rollout' if scale=='rollout' else 'videos_raw');out.mkdir(exist_ok=True)
    limits={}
    for row in rows:
        ticks=shared_ticks if shared_ticks is not None else limits_for(run,[row])
        limits[str(row['case_id'])]=ticks
        path=out/Path(row['video']).name
        if not path.exists():
            with np.load(run/row['trace']) as trace:render(run/row['video'],path,trace,ticks,row['steps'])
        if scale=='rollout' and 'video_raw' in row and 'video_raw_checkpoint' not in row:row['video_raw_checkpoint']=row['video_raw']
        row['video_raw']=str(path.relative_to(run));row['raw_axis_scope']=scale
        print('RAW VIDEO',run.name,path.name,scale,flush=True)
    write_json(run/('raw_axis_limits_rollout.json' if scale=='rollout' else 'raw_axis_limits.json'),dict(per_case_ticks=limits,scope=scale,zero_based=True,margin_fraction=.05))
    write_json(run/'rollouts.json',rows);write_checkpoint_gallery(run,rows,True)
    write_json(marker,dict(videos=50,percentile_and_raw_share_footage=True,axis_scope=scale))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--preview',type=Path)
    p.add_argument('--scale',choices=['rollout','checkpoint'],default='rollout');p.add_argument('--case-id',type=int)
    a=p.parse_args();main(a.run_dir,a.preview,a.scale,a.case_id)
