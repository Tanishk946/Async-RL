from async_rl.grpo import advantages_for_batch, group_advantages, grpo_loss
from async_rl.client import Trajectory
from async_rl.prompts import addition_prompt, render_chat


def _t(pid: str, reward: float) -> Trajectory:
    return Trajectory(
        prompt_id=pid,
        policy_version="v0",
        prompt_text=render_chat(addition_prompt(1, 2)),
        completion="#### 3",
        token_ids=[1, 2],
        logprobs=[-0.1, -0.2],
        reward=reward,
        finish_reason="stop",
    )


def test_group_advantages_zero_when_constant():
    assert group_advantages([1, 1, 1, 1]) == [0.0, 0.0, 0.0, 0.0]


def test_group_advantages_relative():
    adv = group_advantages([0.0, 1.0])
    assert adv[0] < 0 < adv[1]


def test_advantages_per_prompt_group():
    trajs = [_t("a", 0), _t("a", 1), _t("b", 1), _t("b", 1)]
    adv = advantages_for_batch(trajs)
    assert adv[0] < 0 < adv[1]
    assert adv[2] == 0 and adv[3] == 0


def test_on_policy_grpo_zero_when_advantages_zero():
    lp = [[-0.1, -0.2]] * 4
    loss = grpo_loss(lp, lp, [0, 0, 0, 0])
    assert loss == 0.0


def test_prompt_contains_compute():
    text = addition_prompt(17, 25)
    assert "Compute 17 + 25" in text
    assert "####" in text
