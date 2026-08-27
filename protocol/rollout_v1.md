# Rollout protocol v1

JSON over HTTP. The Go manager is the only process that talks to vLLM and the environment.

## `POST /v1/rollout`

Request:

```json
{
  "policy_version": "v0",
  "prompts": [
    {"id": "p0", "text": "<already chat-templated string>"}
  ],
  "group_size": 4,
  "sampling": {
    "max_tokens": 64,
    "temperature": 0.8,
    "top_p": 0.95,
    "stop": ["<|im_end|>"]
  }
}
```

Response:

```json
{
  "policy_version": "v0",
  "trajectories": [
    {
      "prompt_id": "p0",
      "policy_version": "v0",
      "prompt_text": "...",
      "completion": "#### 42",
      "token_ids": [1, 2, 3],
      "logprobs": [-0.1, -0.2, -0.05],
      "reward": 1.0,
      "finish_reason": "stop"
    }
  ]
}
```

Group size `G` means `G` trajectories per prompt id. GRPO advantages are computed in Python over each prompt group.

## `POST /v1/policy`

```json
{"policy_version": "v1", "checkpoint_path": "checkpoints/phase1"}
```

Stamps future rollouts. Reloading vLLM weights is the trainer's/orchestrator's job in phase 1 (restart vLLM from the checkpoint).

## `GET /healthz`
