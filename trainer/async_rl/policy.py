from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_policy(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: list[str]):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    return tok, model, device


def completion_logprobs(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    max_seq_len: int,
) -> torch.Tensor:
    """Logprobs of completion tokens under the current policy."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if len(full) > max_seq_len:
        full = full[:max_seq_len]
    comp_len = max(0, len(full) - len(prompt_ids))
    if comp_len == 0:
        return torch.zeros(0, device=device)

    ids = torch.tensor([full], device=device)
    out = model(ids)
    logp = torch.log_softmax(out.logits, dim=-1)
    # token t is predicted from position t-1
    start = len(prompt_ids)
    token_logp = []
    for i in range(comp_len):
        pos = start + i - 1
        if pos < 0 or pos >= logp.size(1):
            break
        tid = full[start + i]
        token_logp.append(logp[0, pos, tid])
    if not token_logp:
        return torch.zeros(0, device=device)
    return torch.stack(token_logp)


def torch_grpo_loss(
    new_logprobs: list[torch.Tensor],
    old_logprobs: list[list[float]],
    advantages: list[float],
    clip_eps: float,
) -> torch.Tensor:
    losses = []
    for new_lp, old_lp, adv in zip(new_logprobs, old_logprobs, advantages):
        t = min(new_lp.numel(), len(old_lp))
        if t == 0:
            continue
        new = new_lp[:t]
        old = torch.tensor(old_lp[:t], device=new.device, dtype=new.dtype)
        ratio = torch.exp((new - old).clamp(-20, 20))
        adv_t = torch.tensor(adv, device=new.device, dtype=new.dtype)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        losses.append(-torch.min(unclipped, clipped).mean())
    if not losses:
        return torch.zeros((), device=new_logprobs[0].device if new_logprobs else "cpu")
    return torch.stack(losses).mean()
