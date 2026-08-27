# Architecture — Async RL (Phase 1)

End-to-end design for one complete GRPO training iteration, and the choices that keep Phase 1 small enough to debug while still matching the shape of a later multi-node async system.

## Goal

Prove this closed loop on a single machine:

```text
Python trainer → Go Rollout Manager → vLLM → Environment → Reward → Trajectory → Trainer
```

Success for Phase 1 is not “loss goes down.” It is:

1. A group of `G` trajectories returns with rewards in `{0, 1}` and mixed values in the batch.
2. One GRPO optimizer step runs.
3. `policy_version` increments.
4. The next rollout is stamped with the new version (weight reload may still be checkpoint + restart).

True overlap of generate vs train, stale-policy correction, and multi-node weight broadcast are **out of scope** for Phase 1.

---

## System diagram

```text
┌─────────────────────────────┐
│  Python trainer             │
│  - chat template / tokenize │
│  - sample addition prompts  │
│  - GRPO advantages + loss   │
│  - LoRA step (optional)     │
│  - bump policy_version      │
└──────────────┬──────────────┘
               │ HTTP JSON
               │ POST /v1/rollout
               │ POST /v1/policy
               ▼
┌─────────────────────────────┐
│  Go rollout manager         │
│  - fan-out per prompt       │
│  - group_size G completions │
│  - stamp policy_version     │
│  - score addition env       │
└──────┬──────────────┬───────┘
       │              │
       │ OpenAI API   │ local
       ▼              ▼
┌─────────────┐  ┌──────────────┐
│ vLLM        │  │ Addition env │
│ /v1/        │  │ #### N → 0/1 │
│ completions │  └──────────────┘
└─────────────┘
       │
       ▼
  Trajectories {prompt_id, token_ids, logprobs, reward, policy_version}
```

Phase 1 is **synchronous**: one request completes, then one train step. The Go process exists so concurrency (worker fan-out, timeouts, later queues) lives in one place without reimplementing the loss in Go.

---

## Component responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| **Python trainer** | Chat template, tokenization, GRPO math, LoRA optimizer, checkpoint write | Talking to vLLM directly; env scoring |
| **Go rollout manager** | Concurrent generate calls, grouping `G` samples, env reward, HTTP protocol, `policy_version` stamp | Chat templates, gradients, KL |
| **vLLM** | Sampling + completion logprobs | Reward, training |
| **Addition env** | Parse `Compute A + B` and `#### N`; binary reward | Prompt construction |

### Boundary rule: one tokenizer owner

The trainer applies the chat template and sends **already-rendered** prompt strings. The manager forwards them to `/v1/completions` without re-templating. If Go, vLLM chat mode, and Hugging Face each apply a template, GRPO logprob ratios are wrong.

---

## End-to-end Phase 1 iteration

1. Trainer loads `configs/phase1.yaml`, builds `prompts_per_step` addition prompts, applies chat template.
2. Trainer `POST /v1/rollout` with `policy_version`, `group_size=G`, sampling params.
3. Manager fans out: for each prompt, request `G` completions from vLLM (or the mock completer).
4. Manager scores each completion with the addition env → `reward ∈ {0, 1}`.
5. Manager returns trajectories with `token_ids`, `logprobs`, `reward`, `policy_version`.
6. Trainer groups by `prompt_id`, computes group-relative advantages, runs clipped GRPO surrogate.
7. Trainer saves LoRA/checkpoint (real train path) or skips weights (`--dry-run`).
8. Trainer `POST /v1/policy` with bumped `policy_version` (e.g. `v0` → `v1`).
9. Weight visibility to the next generate: **checkpoint + restart vLLM** in Phase 1. The manager only stamps versions; it does not hot-swap weights.

Dry-run path (`scripts/run_mock_iteration.sh`): mock completer produces mixed correct/incorrect `####` answers so groups have reward variance without a GPU.

---

## Protocol (v1)

Contract lives in [`protocol/rollout_v1.md`](protocol/rollout_v1.md). Summary:

- Transport: HTTP + JSON (same schema can move to gRPC later).
- `POST /v1/rollout` → trajectories for GRPO groups.
- `POST /v1/policy` → stamp future rollouts; optional `checkpoint_path`.
- `GET /healthz` → liveness + current `policy_version`.

Every trajectory carries `policy_version` so Phase 2 can detect stale on-policy mixing without a schema rewrite.

---

## Algorithm: GRPO

Phase 1 uses **Group Relative Policy Optimization** (no critic):

- For each prompt, sample `G` completions.
- Advantage for sample `i` in the group:  
  \(A_i = (r_i - \mathrm{mean}(r)) / (\mathrm{std}(r) + \varepsilon)\)
- Clipped surrogate on completion token logprobs (PPO-style ratio clip).
- **KL / reference model: off** (`kl_coeff: 0`) to save VRAM and simplify the first green loop.

`G = 4` by default. Temperature `0.8` (not greedy) so groups are not all identical.

---

## Model, env, and training defaults

| Choice | Phase 1 value | Rationale |
| --- | --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` | Small open weights, vLLM-friendly, same family as later scale-ups |
| Env | Synthetic 2-digit addition, answer format `#### N` | Deterministic, ms eval, mixed 0/1 rewards on tiny models |
| Reward | Exact match → `1.0`, else `0.0` | No reward model; no format bonus yet |
| Adapter | LoRA r=8 on `q_proj`, `v_proj` | Fits one-GPU sequential generate/train; easier hot-swap later |
| Sampling | max 64 tokens, temp 0.8, top_p 0.95 | Short completions; variance for GRPO |
| Prompt API | Completions + pre-templated strings | Avoids chat-template double application |

Larger local GPU (≥16GB): prefer `Qwen2.5-1.5B-Instruct` with the same protocol. After the plumbing works, swap env to Countdown/TinyZero without changing the manager contract.

---

## Design choices (locked for Phase 1)

### 1. Synchronous first, async-shaped interfaces

Phase 1 does not overlap generate and train. Interfaces already look async-ready (`policy_version`, HTTP manager, trajectory schema) so concurrency can be added without redesigning the trainer.

### 2. Go for rollouts, Python for learning

Go owns connection pools, fan-out, timeouts, and later backpressure. Python owns autograd, LoRA, and GRPO. The loss is never reimplemented in Go.

### 3. Do not use TRL’s built-in generate loop

`trl.GRPOTrainer` wants to generate internally, which fights an external Go manager. Phase 1 uses a small custom train loop and treats TRL as a formula reference only.

### 4. On-policy within one step

All trajectories in a batch share one `policy_version`. No mixing of old and new groups in the same GRPO step. Staleness / truncated IS / V-trace are Phase 2+.

### 5. Weight sync = checkpoint + restart

Phase 1 closed loop for weights: save checkpoint → restart vLLM from it → bump `policy_version`. NCCL/LoRA hot-swap is deferred. The manager records version; the orchestrator reloads weights.

### 6. One GPU means sequential occupancy

vLLM and the HF trainer usually cannot share one GPU. Pattern: generate → stop/unload vLLM → train → reload. Two GPUs: pin vLLM to one device and the trainer to the other from day one.

### 7. Completions API, not Chat

Trainer owns templating; manager calls `/v1/completions` with raw prompt text and requests completion logprobs. Chat endpoints hide tokenization details that GRPO needs to match.

### 8. Mock completer for CI / no-GPU

Deterministic-ish mixed rewards so `./scripts/run_mock_iteration.sh` validates protocol + GRPO math without downloading a model.

### 9. Env stays rule-based and tiny

Addition is not the research target; it is a plumbing probe. Code / SWE / web envs hide manager bugs behind sandbox noise.

### 10. Cursor rules under `.cursor/`

Repo-local guidance for Go concurrency, rollout correctness, and Python training conventions so later async work stays consistent with Phase 1 boundaries.

---

## Repository layout

```text
manager/                 Go rollout manager (cmd + internal/{api,env,rollout,vllm,types})
trainer/async_rl/        Python client, prompts, GRPO, optional LoRA train
configs/phase1.yaml      Defaults for model / G / sampling / LoRA
protocol/rollout_v1.md   Wire format
scripts/                 Mock e2e and vLLM iteration helpers
architecture.md          This document
```

---

## Explicitly deferred (Phase 2+)

- Overlapped generate vs train and trajectory queues with backpressure
- Off-policy / stale `policy_version` correction
- KL to a frozen reference policy
- Countdown, GSM8K, code, or agent environments
- Multi-node NCCL or shared-memory weight broadcast
- Ray / distributed orchestrators (HTTP first; same JSON schema later)

---

## Failure and correctness notes

- A failed vLLM call fails the whole `/v1/rollout` for that request (no silent partial groups in Phase 1).
- Constant rewards in a group → advantages are zero → no GRPO update signal (mock and prompt difficulty are chosen to avoid all-zeros when possible).
- Cancellation / deadline propagation across manager → vLLM → env is required before production async; Phase 1 uses HTTP client timeouts only.
- Do not treat truncated completions as successes unless the algorithm explicitly allows it (Phase 1: still scored; usually reward 0 if `####` missing).

---

## Related docs

- [`README.md`](README.md) — run instructions
- [`protocol/rollout_v1.md`](protocol/rollout_v1.md) — HTTP schema
- [`configs/phase1.yaml`](configs/phase1.yaml) — concrete hyperparameters
- [`.cursor/rules/rl-rollouts.mdc`](.cursor/rules/rl-rollouts.mdc) — rollout engineering invariants
