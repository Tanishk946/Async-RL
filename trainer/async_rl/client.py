from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Trajectory:
    prompt_id: str
    policy_version: str
    prompt_text: str
    completion: str
    token_ids: list[int]
    logprobs: list[float]
    reward: float
    finish_reason: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            prompt_id=d["prompt_id"],
            policy_version=d["policy_version"],
            prompt_text=d["prompt_text"],
            completion=d["completion"],
            token_ids=list(d.get("token_ids") or []),
            logprobs=[float(x) for x in (d.get("logprobs") or [])],
            reward=float(d["reward"]),
            finish_reason=d.get("finish_reason") or "",
        )


class RolloutClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.base_url}/healthz")
            r.raise_for_status()
            return r.json()

    def set_policy(self, policy_version: str, checkpoint_path: str = "") -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(
                f"{self.base_url}/v1/policy",
                json={"policy_version": policy_version, "checkpoint_path": checkpoint_path},
            )
            r.raise_for_status()
            return r.json()

    def rollout(
        self,
        prompts: list[dict[str, str]],
        group_size: int,
        policy_version: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str],
    ) -> list[Trajectory]:
        payload = {
            "policy_version": policy_version,
            "prompts": prompts,
            "group_size": group_size,
            "sampling": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            },
        }
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/v1/rollout", json=payload)
            r.raise_for_status()
            data = r.json()
        return [Trajectory.from_dict(t) for t in data["trajectories"]]
