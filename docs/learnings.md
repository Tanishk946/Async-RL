# Learnings — Phase 1 Async RL (personal runbook)

Notes from building and running Phase 1 on a small WSL2 box (~8 GB RAM, no GPU).  
Narrative order: design choices first, then the CPU vLLM bring-up failures and how we got past them.

---

## Phase 1 shape (why the system looks like this)

### One loop, not async yet

We deliberately shipped a **synchronous** iteration: rollout batch → GRPO step → bump `policy_version`.  
True overlap of generate vs train, stale-policy correction, and trajectory queues are deferred. Interfaces already carry `policy_version` and an HTTP manager so those can land without rewriting the trainer.

**Takeaway:** prove plumbing on a tiny env before adding concurrency.

### Go for rollouts, Python for learning

vLLM and env scoring sit behind a Go manager (fan-out, timeouts, later backpressure).  
GRPO + LoRA stay in Python. We do not reimplement the loss in Go, and we do not use TRL’s built-in generate loop (it fights an external manager).

**Takeaway:** keep the language boundary at “trajectories in, gradients out.”

### One tokenizer owner

The trainer applies the chat template and sends **already-rendered** prompt strings.  
The manager calls OpenAI-style `/v1/completions` and never re-templates. If Go, vLLM chat mode, and Hugging Face each apply a template, logprob ratios for GRPO are wrong.

**Takeaway:** completions API + pre-templated strings; chat endpoints hide tokenization we need.

### Addition env as a plumbing probe

Binary reward on `#### N` is boring on purpose: deterministic, millisecond eval, mixed 0/1 rewards on a 0.5B model.  
Research envs (Countdown, GSM8K, code) hide manager bugs behind sandbox noise.

**Takeaway:** validate protocol + GRPO math with a rule-based env first.

### Mock completer for no-GPU CI

`Completer` is a one-method interface (`Complete(ctx, prompt, n, sampling)`).  
`MockCompleter` returns mixed correct/incorrect `####` answers so groups have reward variance without downloading weights. `./scripts/run_mock_iteration.sh` exercises the full HTTP + GRPO path.

**Takeaway:** swap generate backends without touching `Manager.Rollout`.

### Weight sync is checkpoint + restart

Phase 1 does not hot-swap LoRA into vLLM. Trainer saves a checkpoint and `POST /v1/policy` only **stamps** the version string. Reloading weights is an orchestrator concern (restart vLLM from the checkpoint).

**Takeaway:** version stamps now; NCCL / LoRA hot-swap later.

### Failure semantics for groups

Any failed vLLM call fails the whole `/v1/rollout` for that request — no silent partial groups.  
Constant rewards in a group → zero advantages → zero GRPO signal (we saw this on the successful CPU dry-run when mean reward was 1.0).

**Takeaway:** distinguish “loop works” from “there was a learning signal.”

---

## Running on CPU / WSL (what actually broke)

Goal: `./scripts/run_vllm_cpu_iteration.sh` — real CPU vLLM generate + dry-run GRPO (no second HF model copy on ~8 GB).

### Error 1 — CUDA wheel vs CPU wheel

**What happened:** Plain `pip install vllm` is the GPU build. On CPU you need the `+cpu` wheel (we used `vllm==0.28.0+cpu`) and CPU torch from the PyTorch CPU index.

**Turnaround:** Install torch from `https://download.pytorch.org/whl/cpu`, then the GitHub `vllm-…+cpu-…manylinux….whl`. Keep a dedicated venv; do not let a later `pip install` pull CUDA torch over it.

### Error 2 — Missing `libnuma.so.1`

**What happened:** vLLM’s AVX512 CPU extension failed to import (`libnuma.so.1: cannot open shared object file`). Without that extension, the worker crashed later with `torch.ops._C` missing `init_cpu_memory_env`.

**Turnaround:** We could not `sudo apt-get install libnuma1` in the agent environment. Instead:

```bash
apt-get download libnuma1
dpkg-deb -x libnuma1_*.deb .deps/numa_extract
```

Stage `libnuma.so.1` (plus `libiomp5.so` / tcmalloc) into a **space-free** dir such as `/tmp/async-rl-libs` and set `LD_LIBRARY_PATH` / `LD_PRELOAD`. The CPU script now does this automatically when `.deps/numa_extract` exists.

**Design choice:** Prefer system `libnuma1` when you have sudo; keep the extracted fallback for locked-down WSL.

### Error 3 — Spaces in the repo path (`Async RL`)

**What happened:** Two different breakages:

1. `LD_PRELOAD=/mnt/g/Async RL/...` split on the space → preload ignored / wrong paths.
2. TorchInductor’s linker received `-L/mnt/g/Async RL/.venv/.../torch/lib`, which became two args (`…/Async` and `RL/…`). Result: `cannot find -ltorch` / `CppCompileError`, then `EngineDeadError` on `/v1/completions`.

**Turnaround:** Move the venv to a path **without spaces**:

```bash
mv .venv ~/.venvs/async-rl
ln -s ~/.venvs/async-rl .venv
source ~/.venvs/async-rl/bin/activate   # use the real path, not only the symlink
```

The CPU script prefers `$HOME/.venvs/async-rl/bin/python` so `torch.__file__` has no spaces even if the repo lives under `Async RL`. Also set `TORCHDYNAMO_DISABLE=1` as belt-and-suspenders with `--enforce-eager`.

**Design choice:** Repo name with a space is fine for git; **never** put the Python env or native link paths under a spaced directory on this stack.

### Error 4 — KV cache larger than free RAM

**What happened:** After loading Qwen2.5-0.5B, vLLM tried to reserve `VLLM_CPU_KVCACHE_SPACE=2` GiB and failed: only ~0.6 GiB free on the NUMA node. A later attempt with `VLLM_CPU_KVCACHE_SPACE=0.4` failed validation — that env var expects an **integer** GiB.

**Turnaround:** Pass an explicit byte budget and shorten context:

```bash
--kv-cache-memory-bytes 400000000   # ~400 MiB
--max-model-len 256
```

Use `configs/phase1_cpu.yaml` (G=2, 2 prompts, short completions) so peak memory and latency stay small.

**Design choice:** On ~8 GB WSL, dry-run against real vLLM is the default; `FULL_TRAIN=1` (HF LoRA while vLLM may still be up) is optional and likely to OOM.

### Error 5 — Timeouts too short for CPU generate

**What happened:** A single `/v1/completions` with `n=2` on CPU took on the order of **many minutes**. Default 120 s timeouts on the Go vLLM client and Python `httpx` client were not enough once generation actually worked.

**Turnaround:** Raise both sides to **900 s** for Phase 1 CPU runs (`manager/internal/vllm/client.go`, `trainer/async_rl/client.py`).

**Design choice:** Timeouts should be distinguishable from ordinary failures; lengthen them for CPU, do not disable them.

### Error 6 — Rollout `400` / Engine dead (symptom, not root cause)

**What happened:** Trainer saw `httpx.HTTPStatusError: 400` from `POST /v1/rollout` while the manager was only forwarding a dead/failed vLLM engine. Logs also showed shared-memory broadcast waits while the engine was stuck compiling or restarting.

**Turnaround:** Always read **vLLM worker / EngineCore** logs for the root cause (Inductor link error, KV OOM, etc.). The manager correctly fails the whole rollout; the HTTP status alone is not diagnostic enough.

**Design choice:** Keep “fail the whole group” for Phase 1; improve error passthrough later so clients see the vLLM message body.

### Error 7 — Cleanup unbound `MGR_PID`

**What happened:** With `set -u`, if vLLM never became ready, the script’s `trap` referenced `MGR_PID` before it was set.

**Turnaround:** Initialize `VLLM_PID=""` / `MGR_PID=""` and only `kill` when set.

---

## What finally worked (CPU dry-run)

```bash
cd "/mnt/g/Async RL"   # or your clone
source "$HOME/.venvs/async-rl/bin/activate"
./scripts/run_vllm_cpu_iteration.sh
```

Successful print looked like:

```text
trajectories=4 mean_reward=1.000 loss=0.0000 policy v0 -> v1
phase1 CPU vLLM iteration ok
```

`loss=0` with `mean_reward=1.0` means every sample in each group got reward 1 → **group advantages are zero** → no GRPO update signal. The loop still proved: templating → manager → CPU vLLM → env → advantages → policy bump.

For learning signal on CPU, prefer the mock completer (forced reward mix) or a harder prompt distribution so groups are not all-correct.

---

## Checklist for the next machine

1. CPU vLLM wheel + CPU torch (not CUDA).
2. `libnuma1` installed or extracted into `.deps/` / `/tmp/async-rl-libs`.
3. Venv on a **space-free** path; activate that path explicitly.
4. Small KV budget (`--kv-cache-memory-bytes`) and `phase1_cpu.yaml`.
5. Long HTTP timeouts if generating on CPU.
6. Prefer `--dry-run` on ≤8 GB RAM; full LoRA train needs more memory or sequential unload of vLLM.
7. Read EngineCore logs when `/v1/rollout` returns 4xx/5xx.

---

## Still open / watch next

- `run_train` still loads LoRA **before** rollout, so one GPU / low-RAM sequential occupancy is not fully reflected in code order.
- Manager returns generic 400 on rollout failure; surfacing vLLM’s error string would save debug time.
- NUMA MEMBIND on WSL can report less “available” memory than `free -h`; size KV cache from what vLLM prints, not from host free alone.
- Path with spaces will keep biting any native toolchain (Inductor, CUDA builds, some `LD_*` vars) — rename or isolate the env early.
