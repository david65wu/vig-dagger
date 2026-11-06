"""Redraw saved flow traces with a readable scale; no simulation or rescoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from vig_dagger.insertion_video_plot import BACKGROUND,VIDEO_HEIGHT,draw_percentile_plot


def update(root,path):
    relative=path.relative_to(root)
    backup=root/'video_originals'/relative
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():os.link(path,backup)
    trace_path=path.parent.parent/'traces'/f'{path.stem}.npz'
    trace=np.load(trace_path)
    score_steps=trace['score_steps'].tolist();percentiles=trace['percentile'].tolist()
    expected=len(trace['actions'])
    reader=cv2.VideoCapture(str(backup))
    assert reader.isOpened(),backup
    fps=reader.get(cv2.CAP_PROP_FPS)
    temporary=path.with_name(path.stem+'.scale_update.mp4')
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'512x{VIDEO_HEIGHT}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2',
        '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    writer=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    frames=0
    try:
        while True:
            ok,frame=reader.read()
            if not ok:break
            assert frame.shape[1]==512
            canvas=Image.new('RGB',(512,VIDEO_HEIGHT),BACKGROUND)
            canvas.paste(Image.fromarray(cv2.cvtColor(frame[:320],cv2.COLOR_BGR2RGB)),(0,0))
            draw_percentile_plot(canvas,frames,score_steps,percentiles,expected)
            writer.stdin.write(np.asarray(canvas).tobytes())
            if frames==min(80,expected-1) and path.stem=='00_familiar_id' and path.parent.parent.name=='demos500_epoch300':
                canvas.save(root/'video_scale_preview.png')
            frames+=1
    finally:
        reader.release();writer.stdin.close()
        error=writer.stderr.read().decode(errors='replace');returncode=writer.wait()
    if returncode:raise RuntimeError(error)
    assert frames==expected,(path,frames,expected)
    # Decode the result and verify the exact frame count before replacing it.
    count=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,width,height','-of','json',str(temporary)],text=True))['streams'][0]
    assert int(count['nb_read_frames'])==expected and count['height']==VIDEO_HEIGHT
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(temporary),'-f','null','-'],check=True,capture_output=True)
    temporary.replace(path)
    return dict(path=str(relative),frames=frames,fps=fps)


def main(root):
    cv2.setNumThreads(1)
    index=json.loads((root/'video_index.json').read_text())
    paths=[root/r['path'] for r in index]
    paths+=sorted((root/'pilot').glob('*/videos/*.mp4'))
    paths=list(dict.fromkeys(paths));started=time.time();results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(update,root,p) for p in paths]
        for f in as_completed(futures):
            results.append(f.result())
            print(f'UPDATED {len(results)}/{len(paths)} {results[-1]["path"]}',flush=True)
    report=dict(completed=len(results),main_videos=len(index),pilot_videos=len(paths)-len(index),
        y_ticks=[0,20,40,60,80,100],height=VIDEO_HEIGHT,seconds=time.time()-started,
        original_videos='video_originals/',scores='Original saved percentile traces; no rescoring',videos=results)
    (root/'video_scale_update.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
