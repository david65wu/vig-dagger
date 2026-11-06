"""Shared toy policy evaluation: action samples, flow self-consistency, image sensitivity."""
from functools import partial
from pathlib import Path
import json
import subprocess

import numpy as np
import torch

from vig_dagger.policy import LibraryPolicy
from robopolicy.processing.resolution import degrade_resolution

FACTORS = (1, 2, 4, 8, 16, 32, 64)


def observation(obs, device):
    state = obs["policy"].to(device)
    images = torch.stack([obs["rgb_camera"][f"{c}_cam"][0].permute(2, 0, 1)
                          for c in ("third_person", "wrist")])[None].to(device)
    return state, images


def video(path, frames, fps=30):
    from PIL import Image, ImageDraw
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", "512x284", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                             "-threads", "2", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    try:
        for left, wrist, label in frames:
            panel = Image.new("RGB", (512, 284))
            panel.paste(Image.fromarray(left), (0, 28))
            panel.paste(Image.fromarray(wrist), (256, 28))
            ImageDraw.Draw(panel).text((5, 7), label, fill="white")
            proc.stdin.write(panel.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"Video encoding failed: {path}")


@torch.inference_mode()
def measure(policy: LibraryPolicy, state, images, seed, samples=128, loss_draws=512):
    """Use 128 shared noises across resolutions and eight independently scored actions."""
    torch.manual_seed(seed)
    noise = torch.randn(samples, policy.horizon, policy.action_dim, device=policy.device)
    if len(policy.factors) == 7:
        blocks = policy.sample_normalized(state, images, FACTORS, samples, noise.repeat(7, 1, 1))
    else:
        # The full-res policy only has category 0. Degrade externally, preserving that category.
        blocks = []
        for factor in FACTORS:
            degraded = degrade_resolution(images[:, None], torch.zeros(1, device=policy.device, dtype=torch.long), (factor,))
            # A one-factor tuple whose index is 0 is supported by degrade_resolution.
            blocks.append(policy.sample_normalized(state, degraded[:, 0], [1], samples, noise)[0])
        blocks = torch.stack(blocks)
    sorted_samples = blocks.flatten(2).sort(dim=1).values
    gaps = (sorted_samples[1:] - sorted_samples[:1]).abs().mean(dim=(1, 2))
    raw = policy.denormalize_actions(blocks)
    # Physical-unit translation-only counterpart avoids interpreting tiny rotation scales as millimeters.
    raw_sorted = raw[:, :, :, :3].sort(dim=1).values
    translation_gaps = (raw_sorted[1:] - raw_sorted[:1]).abs().mean(dim=(1, 2, 3))
    inputs = policy.packer(policy._batch(state, images, [1]), device=str(policy.device))
    context, _ = policy.policy.encode(inputs)
    field = policy._field(inputs, loss_draws)
    losses = []
    for action in blocks[0, :8]:
        loss = policy.policy.algorithm.compute_loss(field, action[None].repeat(loss_draws, 1, 1),
                                                    context.repeat_interleave(loss_draws, dim=0))
        losses.append(float(loss))
    chunks = raw[0].cpu().numpy()
    first = chunks[:, 0, :3]
    spread = np.sqrt(np.var(first, axis=0, ddof=1).sum())
    return dict(flow_mean=float(np.mean(losses)), flow_std=float(np.std(losses)), flow_samples=losses,
                vig=float(gaps.max()), vig_by_factor=gaps.cpu().tolist(),
                vig_translation_m=float(translation_gaps.max()),
                action_spread_m=float(spread), mean_action=first.mean(axis=0).tolist()), chunks, raw.cpu().numpy()
