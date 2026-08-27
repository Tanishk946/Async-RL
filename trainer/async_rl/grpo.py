"""Group Relative Policy Optimization (DeepSeekMath). No critic."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from async_rl.client import Trajectory


def group_by_prompt(trajs: Sequence[Trajectory]) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        groups[t.prompt_id].append(t)
    return dict(groups)


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var**0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]


def advantages_for_batch(trajs: Sequence[Trajectory]) -> list[float]:
    """Per-trajectory advantages; normalized inside each prompt group."""
    out = [0.0] * len(trajs)
    groups = group_by_prompt(trajs)
    index = {id(t): i for i, t in enumerate(trajs)}
    for members in groups.values():
        adv = group_advantages([m.reward for m in members])
        for m, a in zip(members, adv):
            out[index[id(m)]] = a
    return out


def mean_logprob(logprobs: Sequence[float]) -> float:
    if not logprobs:
        return 0.0
    return sum(logprobs) / len(logprobs)


def grpo_loss(
    new_logprobs: Sequence[Sequence[float]],
    old_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[float],
    clip_eps: float = 0.2,
) -> float:
    """Token-average clipped GRPO surrogate (KL off). Pure Python for tests."""
    total = 0.0
    count = 0
    for new_lp, old_lp, adv in zip(new_logprobs, old_logprobs, advantages):
        t = min(len(new_lp), len(old_lp))
        if t == 0:
            continue
        for k in range(t):
            ratio = _exp(new_lp[k] - old_lp[k])
            unclipped = ratio * adv
            clipped_ratio = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
            clipped = clipped_ratio * adv
            total += -min(unclipped, clipped)
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _exp(x: float) -> float:
    import math

    x = max(min(x, 20.0), -20.0)
    return math.exp(x)
