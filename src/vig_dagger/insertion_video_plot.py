"""Shared percentile plot for insertion rollout videos."""
from bisect import bisect_right

BACKGROUND=(22,26,33)
VIDEO_HEIGHT=480
VIDEO_HEIGHT_VIG=660


def draw_percentile_plot(image,step,score_steps,percentiles,total_frames,*,top=320,label=None,color='#63c4ff'):
    from PIL import ImageDraw
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,top,511,top+159),fill=BACKGROUND)
    x0,y0,width,height=38,top+20,462,120
    if label:draw.text((x0,top+2),label,fill='white')
    for tick in range(0,101,20):
        y=y0+height*(1-tick/100)
        draw.line((x0,y,x0+width,y),fill=(58,64,74),width=1)
        draw.line((x0-4,y,x0,y),fill=(170,175,184),width=1)
        label=str(tick)
        box=draw.textbbox((0,0),label)
        draw.text((x0-8-(box[2]-box[0]),y-6),label,fill='white')
    draw.line((x0,y0,x0,y0+height,x0+width,y0+height),fill=(150,155,164),width=1)
    index=max(0,bisect_right(score_steps,step)-1)
    trace=[(x0+width*score_steps[j]/max(1,total_frames-1),
            y0+height*(1-percentiles[j]/100)) for j in range(index+1)]
    if len(trace)>1:draw.line(trace,fill=color,width=2)
    if trace:
        x,y=trace[-1];draw.ellipse((x-2,y-2,x+2,y+2),fill='white')
