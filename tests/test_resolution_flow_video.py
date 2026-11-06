import json
import subprocess

import cv2
import numpy as np
import pytest
import torch

from vig_dagger.resolution_flow_video import VideoWriter, camera_frame, render_episode


def make_record(tmp_path, frames=3):
    path = tmp_path / 'episode'
    path.mkdir()
    writer = VideoWriter(path / 'cameras.mp4', 1280, 400, 30)
    for i in range(frames):
        obs = {'rgb_camera': {
            'left_cam': torch.tensor([255, 0, 0], dtype=torch.uint8).repeat(1, 20, 30, 1),
            'wrist_cam': torch.tensor([0, 255, 0], dtype=torch.uint8).repeat(1, 20, 30, 1)}}
        writer.write(camera_frame(obs))
    writer.finish()
    np.savez(path / 'trace.npz', query_seconds=[0., .033333, .066667],
             flow=[[.1, .2], [.3, .1], [.2, .25]], video_steps=[0, 2, 4])
    cfg = dict(work_dir=str(tmp_path), protocol_id='video-test', factors=[1, 2])
    row = dict(group='moe', acting_factor=1, seed=42, success=False, dt=1/60,
               steps=5, video_fps=30)
    return cfg, row, path


def test_video_has_synchronized_frames_both_cameras_and_flow_panel(tmp_path):
    cfg, row, path = make_record(tmp_path)
    target = render_episode(cfg, row, path)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(target)]))['streams'][0]
    assert (info['width'], info['height'], info['nb_frames'], info['r_frame_rate']) == (1280, 760, '3', '30/1')
    capture = cv2.VideoCapture(str(target))
    ok, image = capture.read()
    capture.release()
    assert ok
    assert image[200, 300, 2] > 240 and image[200, 300, 0] < 10  # Left red, decoded BGR.
    assert image[200, 900, 1] > 240  # Wrist green.
    assert image[400:].std() > 10  # Plot panel isn't blank.
    marker = json.loads(target.with_suffix('.json').read_text())
    assert marker['last_step'] == 4 and marker['frames'] == 3
    assert '/failed/' in str(target)


def test_missing_camera_frame_refuses_to_publish_video(tmp_path):
    cfg, row, path = make_record(tmp_path, frames=2)
    with pytest.raises(RuntimeError, match='shorter than its timestamps'):
        render_episode(cfg, row, path)
    assert not list((tmp_path / 'results/videos').rglob('42.mp4'))
    assert not list((tmp_path / 'results/videos').rglob('42.json'))
