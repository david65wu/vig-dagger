"""Per-sample flow self-consistency for the toy failure search (no VIG)."""
import numpy as np
import torch


@torch.inference_mode()
def chunk_losses(policy, inputs, context, chunks, draws=128, batch_rows=1024):
    """Same loss as compute_loss, retaining each action's Monte Carlo mean/SE."""
    algorithm = policy.policy.algorithm
    if algorithm.config.path_type != 'linear':
        raise ValueError('This experiment requires the existing linear flow path')
    means, errors = [], []
    per_batch = max(1, batch_rows // draws)
    for start in range(0, len(chunks), per_batch):
        actions = chunks[start:start + per_batch]
        target = actions.repeat_interleave(draws, dim=0)
        noise = torch.randn_like(target)
        time = algorithm.scheduler.sample(len(target), target.device, target.dtype)
        if algorithm.config.t_max < 1:
            time = time.clamp(max=algorithm.config.t_max)
        mixed = algorithm.path.interpolate(noise, target, time)
        velocity = algorithm.path.velocity(noise, target, time)
        predicted = policy._field(inputs, len(target))(
            mixed, time, global_cond=context.repeat_interleave(len(target), dim=0))
        if algorithm.config.velocity_clamp is not None:
            predicted = predicted.clamp(-algorithm.config.velocity_clamp, algorithm.config.velocity_clamp)
        loss = (predicted - velocity).square().mean(dim=(1, 2)).reshape(len(actions), draws)
        means.append(loss.mean(dim=1))
        errors.append(loss.std(dim=1, unbiased=True) / draws ** .5)
    return torch.cat(means).cpu().numpy(), torch.cat(errors).cpu().numpy()


@torch.inference_mode()
def sample_and_score(policy, state, images, seed, samples=128, draws=128):
    torch.manual_seed(seed)
    inputs = policy.packer(policy._batch(state, images, [1]), device=str(policy.device))
    context, _ = policy.policy.encode(inputs)
    noise = torch.randn(samples, policy.horizon, policy.action_dim, device=policy.device,
                        dtype=context.dtype)
    chunks = policy.policy.algorithm.generate_samples(
        policy._field(inputs, samples), noise, context.repeat_interleave(samples, dim=0))
    losses, errors = chunk_losses(policy, inputs, context, chunks, draws)
    raw = policy.denormalize_actions(chunks).cpu().numpy()
    first = raw[:, 0, :3]
    return dict(flow_mean=float(losses.mean()), flow_p10=float(np.percentile(losses, 10)),
                flow_p90=float(np.percentile(losses, 90)),
                spread_m=float(np.sqrt(first.var(axis=0, ddof=1).sum())) if samples > 1 else None,
                mean_action=first.mean(axis=0).tolist()), raw, losses, errors


def wilson(failures, trials):
    """95% Wilson interval for an exploratory empirical failure rate."""
    z = 1.959963984540054
    rate = failures / trials
    denominator = 1 + z*z/trials
    center = (rate + z*z/(2*trials)) / denominator
    width = z * ((rate*(1-rate)/trials + z*z/(4*trials*trials)) ** .5) / denominator
    return [max(0., center-width), min(1., center+width)]
