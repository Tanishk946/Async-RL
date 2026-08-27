from __future__ import annotations

import argparse
import json
from pathlib import Path

from async_rl.client import RolloutClient
from async_rl.config import load_config
from async_rl.grpo import advantages_for_batch, grpo_loss, group_by_prompt
from async_rl.prompts import addition_prompt, render_chat, sample_addends


def run_dry(cfg, artifacts: Path) -> dict:
    client = RolloutClient(cfg.rollout.manager_url)
    health = client.health()
    pairs = sample_addends(
        cfg.rollout.prompts_per_step,
        cfg.env.addend_min,
        cfg.env.addend_max,
        seed=0,
    )
    prompts = []
    for i, (a, b) in enumerate(pairs):
        prompts.append({"id": f"p{i}", "text": render_chat(addition_prompt(a, b))})

    trajs = client.rollout(
        prompts=prompts,
        group_size=cfg.rollout.group_size,
        policy_version=cfg.train.policy_version,
        max_tokens=cfg.rollout.max_tokens,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
        stop=cfg.rollout.stop,
    )
    adv = advantages_for_batch(trajs)
    old = [t.logprobs for t in trajs]
    # On-policy first step: ratio ~ 1 if new == old.
    loss = grpo_loss(old, old, adv, clip_eps=cfg.train.clip_eps)
    rewards = [t.reward for t in trajs]
    mean_r = sum(rewards) / len(rewards)

    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "health": health,
        "n_trajectories": len(trajs),
        "mean_reward": mean_r,
        "advantages": adv,
        "loss_on_policy": loss,
        "groups": {k: len(v) for k, v in group_by_prompt(trajs).items()},
        "policy_version": cfg.train.policy_version,
        "trajectories": [
            {
                "prompt_id": t.prompt_id,
                "completion": t.completion,
                "reward": t.reward,
                "policy_version": t.policy_version,
            }
            for t in trajs
        ],
    }
    (artifacts / "last_rollout.json").write_text(json.dumps(payload, indent=2))

    next_version = _bump(cfg.train.policy_version)
    client.set_policy(next_version, checkpoint_path=cfg.train.checkpoint_dir)
    payload["next_policy_version"] = next_version
    print(
        f"trajectories={len(trajs)} mean_reward={mean_r:.3f} "
        f"loss={loss:.4f} policy {cfg.train.policy_version} -> {next_version}"
    )
    return payload


def run_train(cfg, artifacts: Path) -> dict:
    import torch
    from torch.nn.utils import clip_grad_norm_

    from async_rl.policy import completion_logprobs, load_policy, torch_grpo_loss

    tok, model, device = load_policy(
        cfg.model.name,
        cfg.model.lora_r,
        cfg.model.lora_alpha,
        cfg.model.lora_dropout,
        cfg.model.target_modules,
    )
    client = RolloutClient(cfg.rollout.manager_url)
    pairs = sample_addends(
        cfg.rollout.prompts_per_step,
        cfg.env.addend_min,
        cfg.env.addend_max,
        seed=0,
    )
    prompts = []
    for i, (a, b) in enumerate(pairs):
        user = addition_prompt(a, b)
        prompts.append({"id": f"p{i}", "text": render_chat(user, tokenizer=tok)})

    trajs = client.rollout(
        prompts=prompts,
        group_size=cfg.rollout.group_size,
        policy_version=cfg.train.policy_version,
        max_tokens=cfg.rollout.max_tokens,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
        stop=cfg.rollout.stop,
    )
    adv = advantages_for_batch(trajs)
    new_lps = [
        completion_logprobs(
            model, tok, t.prompt_text, t.completion, device, cfg.model.max_seq_len
        )
        for t in trajs
    ]
    loss = torch_grpo_loss(new_lps, [t.logprobs for t in trajs], adv, cfg.train.clip_eps)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.train.lr)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
    opt.step()

    ckpt = Path(cfg.train.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    next_version = _bump(cfg.train.policy_version)
    client.set_policy(next_version, checkpoint_path=str(ckpt))
    mean_r = sum(t.reward for t in trajs) / len(trajs)
    print(f"trained step=1 mean_reward={mean_r:.3f} loss={float(loss):.4f} saved {ckpt}")
    artifacts.mkdir(parents=True, exist_ok=True)
    summary = {
        "mean_reward": mean_r,
        "loss": float(loss.detach().cpu()),
        "next_policy_version": next_version,
        "checkpoint": str(ckpt),
    }
    (artifacts / "last_train.json").write_text(json.dumps(summary, indent=2))
    return summary


def _bump(version: str) -> str:
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return version + ".next"


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 GRPO trainer")
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--dry-run", action="store_true", help="rollout + GRPO math, no model weights")
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()
    cfg = load_config(args.config)
    artifacts = Path(args.artifacts)
    if args.dry_run:
        run_dry(cfg, artifacts)
    else:
        run_train(cfg, artifacts)


if __name__ == "__main__":
    main()
