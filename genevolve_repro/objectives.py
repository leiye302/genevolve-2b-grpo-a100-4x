from statistics import fmean, stdev

import torch


def advantages(rewards, epsilon=1e-6):
    if len(rewards) < 2:
        raise ValueError("GRPO requires multiple samples for the same prompt.")
    average = fmean(rewards)
    scale = stdev(rewards) + epsilon
    return [(reward - average) / scale for reward in rewards]


def k3(log_ratio):
    # Outside this range the cap on k3 has already saturated.
    log_ratio = log_ratio.clamp(-20, 20)
    return (torch.expm1(-log_ratio) + log_ratio).clamp(0, 10)


def kl_reward(reward, turns, coef):
    penalty = sum(k3(turn.old - turn.reference).sum().item() for turn in turns)
    return reward - coef * penalty


def grpo_loss(new, turn, advantage, cfg):
    """Return the same per-turn GRPO sum as the matched reproduction."""
    old = turn.old.to(new.device)
    log_ratio = (new - old).clamp(-20, 20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - cfg["clip_low"], 1 + cfg["clip_high"])
    return -torch.minimum(advantage * ratio, advantage * clipped).sum()
