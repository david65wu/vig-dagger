"""Synchronized camera-row / raw-flow-row MP4s for resolution diagnostics."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

WIDTH, CAMERA_HEIGHT, CHART_HEIGHT = 1280, 400, 360


class VideoWriter:
    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix('.partial.mp4')
        self.frames = 0
        self.proc = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.tmp)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def finish(self):
        self.proc.stdin.close()
        error = self.proc.stderr.read().decode()
        code = self.proc.wait()
        self.proc.stderr.close()
        if code or not self.frames:
            raise RuntimeError(f'Video encoding failed ({code}): {error}')
        self.tmp.replace(self.path)

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for stream in [self.proc.stdin, self.proc.stderr]:
            if stream and not stream.closed:
                try:
                    stream.close()
                except BrokenPipeError:
                    pass


def camera_frame(obs):
    """Original RGB views (not resolution-degraded); policy input is unchanged."""
    import cv2
    frame = np.full((CAMERA_HEIGHT, WIDTH, 3), 245, dtype=np.uint8)
    for i, name in enumerate(('left', 'wrist')):
        rgb = obs['rgb_camera'][f'{name}_cam'][0].detach().cpu().numpy()[..., :3]
        rgb = np.asarray(rgb, dtype=np.uint8)
        scale = min((WIDTH // 2) / rgb.shape[1], (CAMERA_HEIGHT - 40) / rgb.shape[0])
        view = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
        x = i * (WIDTH // 2) + (WIDTH // 2 - view.shape[1]) // 2
        y = 40 + (CAMERA_HEIGHT - 40 - view.shape[0]) // 2
        frame[y:y + view.shape[0], x:x + view.shape[1]] = view
        cv2.putText(frame, f'{name.title()} camera (original RGB)', (i * (WIDTH // 2) + 15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 30, 30), 1, cv2.LINE_AA)
    return frame


def render_episode(cfg, row, path):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vig_dagger.resolution_flow import TITLES, write_json

    outcome = 'success' if row['success'] else 'failed'
    target = (Path(cfg['work_dir']) / 'results/videos' / row['group'] /
              f"acting_{row['acting_factor']}" / outcome / f"{row['seed']}.mp4")
    marker = target.with_suffix('.json')
    if target.is_file() and marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get('protocol_id') == cfg['protocol_id']:
            return target
    with np.load(path / 'trace.npz') as data:
        seconds, flow, steps = data['query_seconds'], data['flow'], data['video_steps']
    fps = row['video_fps']
    fig, ax = plt.subplots(figsize=(WIDTH / 100, CHART_HEIGHT / 100), dpi=100)
    colors = plt.get_cmap('tab10').colors
    for i, factor in enumerate(cfg['factors']):
        ax.plot(seconds, flow[:, i], color=colors[i], lw=1.8 if factor == row['acting_factor'] else 1,
                alpha=.9, label=f"{factor}" + (' (acting)' if factor == row['acting_factor'] else ''))
    ax.set(xlabel='Simulated time (seconds)', ylabel='Raw flow-matching loss',
           title=f"{TITLES[row['group']]} | acting factor {row['acting_factor']} | seed {row['seed']} | {outcome.upper()}")
    ax.set_xlim(0, max(row['steps'] * row['dt'], row['dt']))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(title='Scored resolution factor', ncols=7, fontsize=8, title_fontsize=8,
              loc='upper center', bbox_to_anchor=(.5, -.22), frameon=False)
    fig.subplots_adjust(left=.07, right=.99, top=.87, bottom=.32)
    fig.canvas.draw()
    chart = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    reader = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1',
                               '-i', str(path / 'cameras.mp4'), '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-threads', '1', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    writer = VideoWriter(target, WIDTH, CAMERA_HEIGHT + CHART_HEIGHT, fps)
    try:
        size = WIDTH * CAMERA_HEIGHT * 3
        for step in steps:
            raw = reader.stdout.read(size)
            if len(raw) != size:
                raise RuntimeError(f'Camera video shorter than its timestamps: {path}')
            t = float(step * row['dt'])
            bottom = chart.copy()
            x = round(ax.transData.transform((t, 0))[0])
            cv2.line(bottom, (x, CHART_HEIGHT - round(ax.bbox.y1)),
                     (x, CHART_HEIGHT - round(ax.bbox.y0)), (25, 25, 25), 2)
            query = int(np.clip(np.searchsorted(seconds, t, side='right') - 1, 0, len(seconds) - 1))
            for i in range(len(cfg['factors'])):
                px, py = ax.transData.transform((seconds[query], flow[query, i]))
                cv2.circle(bottom, (round(px), CHART_HEIGHT - round(py)), 3,
                           tuple(round(c * 255) for c in colors[i]), -1, cv2.LINE_AA)
            top = np.frombuffer(raw, dtype=np.uint8).reshape(CAMERA_HEIGHT, WIDTH, 3).copy()
            cv2.putText(top, f'{t:.2f} s', (WIDTH // 2 - 80, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (30, 30, 30), 1, cv2.LINE_AA)
            writer.write(np.concatenate([top, bottom]))
        if reader.stdout.read(1):
            raise RuntimeError(f'Camera video longer than its timestamps: {path}')
        error = reader.stderr.read().decode()
        if reader.wait():
            raise RuntimeError(error)
        writer.finish()
        write_json(marker, dict(protocol_id=cfg['protocol_id'], frames=len(steps), fps=fps,
                                first_step=int(steps[0]), last_step=int(steps[-1]),
                                source=str(path), success=row['success']))
        return target
    finally:
        plt.close(fig)
        writer.abort()
        if reader.poll() is None:
            reader.kill()
        reader.wait()
        reader.stdout.close()
        reader.stderr.close()
