# Async RL

Phase 1: one complete GRPO iteration over a tiny open-weight model and a trivial math env.

```
Python trainer  ->  Go rollout manager  ->  vLLM  ->  addition env  ->  trajectories  ->  GRPO step
```

Local default: **Qwen2.5-0.5B-Instruct** + synthetic 2-digit addition (`#### N`). The manager is Go for concurrent rollouts; the loss stays in Python. Phase 1 is **synchronous** (one batch in, one optimizer step out). `policy_version` is stamped on every trajectory so async staleness can be added later without changing the protocol.

This GitHub copy is not a substitute for cluster-scale training. It is the plumbing.

## Layout

| Path | Role |
| --- | --- |
| `architecture.md` | End-to-end design and Phase 1 design choices |
| `manager/` | Go rollout manager: fan-out to vLLM, score env, return groups |
| `trainer/async_rl/` | Python client, GRPO, optional LoRA train step |
| `configs/phase1.yaml` | Model, G=4, sampling, LoRA, KL off |
| `protocol/rollout_v1.md` | HTTP JSON contract |
| `scripts/run_mock_iteration.sh` | End-to-end without a GPU |
| `scripts/run_vllm_iteration.sh` | Real vLLM + trainer on one GPU (sequential) |

## Mock iteration (no GPU)

Needs Go 1.22+ and Python 3.10+.

```bash
./scripts/run_mock_iteration.sh
```

This starts the manager with `-mock` (mixed correct/incorrect `####` answers so groups have reward variance), then runs `python -m async_rl.train --dry-run`: rollout, group advantages, on-policy GRPO surrogate, `POST /v1/policy` bump `v0` → `v1`.

## Real iteration

1. Serve the policy:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 --max-model-len 1024
```

2. Manager (other terminal):

```bash
go -C manager run ./cmd/manager -- \
  -vllm-url http://127.0.0.1:8000/v1 \
  -vllm-model Qwen/Qwen2.5-0.5B-Instruct
```

3. Train one GRPO step (install train extras; one GPU cannot host vLLM and the trainer at once — stop vLLM before this, or use two GPUs):

```bash
pip install -e ".[train]"
python -m async_rl.train --config configs/phase1.yaml
```

Weight sync in phase 1 is **checkpoint + restart vLLM** from `checkpoints/phase1`. The manager only records `policy_version`.

## Tests

```bash
go -C manager test ./...
pip install -e ".[dev]"
pytest
```

## What is intentionally out of scope

- Overlapped generate/train and off-policy corrections
- KL to a frozen reference (config `kl_coeff: 0`)
- Countdown / GSM8K / code envs
- Multi-node NCCL weight broadcast
