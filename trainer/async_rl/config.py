from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_seq_len: int = 512


@dataclass
class RolloutConfig:
    manager_url: str = "http://127.0.0.1:8080"
    vllm_url: str = "http://127.0.0.1:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    group_size: int = 4
    prompts_per_step: int = 4
    max_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])


@dataclass
class EnvConfig:
    name: str = "addition"
    addend_min: int = 0
    addend_max: int = 50


@dataclass
class OptimConfig:
    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coeff: float = 0.0
    max_grad_norm: float = 1.0
    policy_version: str = "v0"
    checkpoint_dir: str = "checkpoints/phase1"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: OptimConfig = field(default_factory=OptimConfig)


def _merge(dc, raw: dict[str, Any]):
    for k, v in raw.items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: str | Path) -> TrainConfig:
    cfg = TrainConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    _merge(cfg, data)
    return cfg
