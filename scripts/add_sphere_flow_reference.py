"""Add each policy's training p95 to the full 16-command rollout videos."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from run_sphere_rollouts import NAMES,write


def panel(frame,flows,steps,index,p95):
    frame[576:]=248
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,f'Flow: {flows[index]:.4f}  |  training p95: {p95:.4f}',(15,609),font,.69,(25,25,25),2,cv2.LINE_AA)
    cv2.putText(frame,'Dashed line: training reference',(650,606),font,.55,(145,65,145),1,cv2.LINE_AA)
    ymax=max(.01,float(np.ceil(max(float(np.max(flows)),p95)*110)/100))
    left,right,top,bottom=65,995,645,758
    for x in [0,5,10,15,20]:
        px=int(left+x/20*(right-left));cv2.line(frame,(px,top),(px,bottom),(225,225,225),1)
        cv2.putText(frame,str(x),(px-8,bottom+22),font,.48,(50,50,50),1,cv2.LINE_AA)
    for y in [0,ymax/2,ymax]:
        py=int(bottom-y/ymax*(bottom-top));cv2.line(frame,(left,py),(right,py),(225,225,225),1)
        cv2.putText(frame,f'{y:.3f}',(7,py+5),font,.42,(50,50,50),1,cv2.LINE_AA)
    baseline=int(bottom-p95/ymax*(bottom-top))
    for x in range(left,right,18):cv2.line(frame,(x,baseline),(min(x+10,right),baseline),(160,70,160),2)
    xs=left+np.array(steps[:index+1])/60/20*(right-left)
    ys=bottom-flows[:index+1]/ymax*(bottom-top)
    curve=np.stack((xs,ys),axis=1).round().astype(np.int32)
    if len(curve)>1:cv2.polylines(frame,[curve],False,(190,90,30),2,cv2.LINE_AA)
    cv2.circle(frame,tuple(curve[-1]),4,(50,50,210),-1,cv2.LINE_AA)
    cv2.putText(frame,'Time (s)',(475,793),font,.48,(50,50,50),1,cv2.LINE_AA)


def update(root,name,row):
    out=root/name;stem=f"{row['case_id']:02d}_{row['kind']}"
    source=out/'chunk_videos'/f'{stem}.mp4';dest=out/'calibrated_chunk_videos';dest.mkdir(exist_ok=True)
    target=dest/source.name
    if target.exists() and target.with_suffix('.json').exists():return str(target)
    calibration=json.loads((root/'flow_calibration'/f'{name}.json').read_text())
    p95=calibration['p95'];arrays=np.load(source.with_name(stem+'_chunks.npz'))
    flows=arrays['flow'];steps=arrays['steps'];assert np.isfinite(flows).all()
    cap=cv2.VideoCapture(str(source));temporary=target.with_name(stem+'.partial.mp4')
    proc=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1024x800','-r','7.5',
        '-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    frame_index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            panel(frame,flows,steps,min(frame_index,len(steps)-1),p95)
            if frame_index==0:cv2.imwrite(str(target.with_suffix('.png')),frame)
            proc.stdin.write(frame.tobytes());frame_index+=1
    finally:
        cap.release();proc.stdin.close()
        if proc.wait():raise RuntimeError(f'Encoding failed: {target}')
    assert frame_index==len(steps)+8,(target,frame_index,len(steps))
    temporary.replace(target)
    # Decode every final frame, checking the artifact actually delivered.
    check=subprocess.run(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames,width,height','-of','json',str(target)],capture_output=True,text=True,check=True)
    assert not check.stderr.strip(),check.stderr
    stream=json.loads(check.stdout)['streams'][0]
    assert int(stream['nb_read_frames'])==frame_index and stream['width']==1024 and stream['height']==800
    data_link=dest/f'{stem}_chunks.npz'
    if not data_link.exists():data_link.symlink_to(source.with_name(stem+'_chunks.npz'))
    meta=json.loads(source.with_suffix('.json').read_text())
    meta.update(training_flow_median=calibration['median'],training_flow_p95=p95,
                training_calibration_frames=calibration['frames'],decoded_frames=frame_index,
                calibration=str(root/'flow_calibration'/f'{name}.json'))
    write(target.with_suffix('.json'),meta)
    return str(target)


def run(root):
    cv2.setNumThreads(1);start=time.time()
    for path in [root/'flow_calibration/status.json',root/'chunk_pipeline.json']:
        while True:
            if path.exists():
                state=json.loads(path.read_text())['state']
                if state=='complete':break
                if state=='failed':raise RuntimeError(f'Prerequisite failed: {path}')
            write(root/'calibrated_chunk_pipeline.json',dict(state='waiting',prerequisite=str(path)))
            if time.time()-start>3600:raise TimeoutError('Prerequisites exceeded one hour')
            time.sleep(5)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(update,root,name,row) for name in NAMES for row in json.loads((root/name/'rollouts.json').read_text())]
        for f in as_completed(jobs):
            results.append(f.result());write(root/'calibrated_chunk_pipeline.json',dict(state='rendering',completed=len(results),total=180))
    assert len(results)==180
    write(root/'calibrated_chunk_validation.json',dict(videos=180,all_frames_decoded=True,horizon=16,
        per_policy_training_reference=True,flow_values_unchanged=True,paths=sorted(results)))
    write(root/'calibrated_chunk_pipeline.json',dict(state='complete',completed=180,total=180,elapsed_seconds=time.time()-start))
    for rel in ['scripts/calibrate_sphere_flow.py','scripts/add_sphere_flow_reference.py']:
        shutil.copy2(Path(__file__).resolve().parents[1]/rel,root/'source_snapshot'/Path(rel).name)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:run(args.root)
    except BaseException as exc:
        write(args.root/'calibrated_chunk_pipeline.json',dict(state='failed',error=repr(exc)));raise
