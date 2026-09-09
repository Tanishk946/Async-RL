#!/usr/bin/env bash
# CPU vLLM loop: generate with vLLM CPU backend, then GRPO.
# Default: --dry-run (real completions, no second HF model copy).
# Full train: FULL_TRAIN=1 ./scripts/run_vllm_cpu_iteration.sh
#   (needs more RAM; may OOM on ~8GB if vLLM stays up during train)
#
# Needs libnuma.so.1 for the AVX512 CPU extension. If missing system-wide:
#   apt-get download libnuma1 && dpkg-deb -x libnuma1_*.deb .deps/numa_extract
#   (script will pick it up from .deps/ or /tmp/async-rl-libs)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
MGR_ADDR="${MGR_ADDR:-127.0.0.1:8080}"
CONFIG="${CONFIG:-configs/phase1_cpu.yaml}"
# KV cache budget in GB for the CPU backend
# KV cache: integer GiB via env, or precise bytes via flag (0.4 GiB env is invalid).
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-1}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-400000000}"
# TorchInductor linker breaks on spaces in -L paths (repo name "Async RL").
# Always prefer the real venv under ~/.venvs so torch.__file__ has no spaces.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
SPACEFREE_PY="${SPACEFREE_PY:-$HOME/.venvs/async-rl/bin/python}"
if [[ -x "$SPACEFREE_PY" ]]; then
  export PATH="$(dirname "$SPACEFREE_PY"):$PATH"
  hash -r 2>/dev/null || true
  PY="$SPACEFREE_PY"
else
  PY="$(command -v python3)"
fi
_torch_file="$("$PY" -c 'import torch,os; print(torch.__file__)' 2>/dev/null || true)"
if [[ -z "$_torch_file" || "$_torch_file" == *" "* ]]; then
  echo "error: torch path has spaces (Inductor cannot link): $_torch_file"
  echo "  Expected space-free venv at: $HOME/.venvs/async-rl"
  echo "  mv .venv \"\$HOME/.venvs/async-rl\" && ln -s \"\$HOME/.venvs/async-rl\" .venv"
  echo "  then: source \"\$HOME/.venvs/async-rl/bin/activate\""
  exit 1
fi
TORCH_LIB="$("$PY" -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LIBRARY_PATH="${TORCH_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use $PY for all Python child processes (never a spaced symlink path).

# Space-free lib dir so LD_PRELOAD/LD_LIBRARY_PATH survive paths like "Async RL".
LIBDIR="${LIBDIR:-/tmp/async-rl-libs}"
mkdir -p "$LIBDIR"
_stage_lib() {
  local src="$1"
  local name
  name="$(basename "$src")"
  if [[ -f "$src" && ! -f "$LIBDIR/$name" ]]; then
    cp -a "$src" "$LIBDIR/$name"
  fi
}
# Prefer user-extracted libnuma (no sudo), then staged copy.
if [[ ! -f "$LIBDIR/libnuma.so.1" ]]; then
  if [[ -f "$ROOT/.deps/numa_extract/usr/lib/x86_64-linux-gnu/libnuma.so.1" ]]; then
    _stage_lib "$ROOT/.deps/numa_extract/usr/lib/x86_64-linux-gnu/libnuma.so.1"
  elif [[ -f /usr/lib/x86_64-linux-gnu/libnuma.so.1 ]]; then
    _stage_lib /usr/lib/x86_64-linux-gnu/libnuma.so.1
  fi
fi
IOMP="$(find "$(dirname "$PY")/.." -name 'libiomp5.so' 2>/dev/null | head -1 || true)"
[[ -n "$IOMP" ]] && _stage_lib "$IOMP"
TCM="$(find "$(dirname "$PY")/.." -name 'libtcmalloc_minimal.so.4' 2>/dev/null | head -1 || true)"
[[ -n "$TCM" ]] && _stage_lib "$TCM"

export LD_LIBRARY_PATH="${LIBDIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PRELOADS=()
[[ -f "$LIBDIR/libnuma.so.1" ]] && PRELOADS+=("$LIBDIR/libnuma.so.1")
[[ -f "$LIBDIR/libiomp5.so" ]] && PRELOADS+=("$LIBDIR/libiomp5.so")
[[ -f "$LIBDIR/libtcmalloc_minimal.so.4" ]] && PRELOADS+=("$LIBDIR/libtcmalloc_minimal.so.4")
if ((${#PRELOADS[@]})); then
  # Join with ':' — paths have no spaces when staged under /tmp/async-rl-libs.
  export LD_PRELOAD="$(IFS=:; echo "${PRELOADS[*]}")${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [[ ! -f "$LIBDIR/libnuma.so.1" ]]; then
  echo "warning: libnuma.so.1 not found; vLLM CPU AVX512 ops will fail."
  echo "  sudo apt-get install -y libnuma1"
  echo "  # or: apt-get download libnuma1 && dpkg-deb -x libnuma1_*.deb .deps/numa_extract"
fi

if ! "$PY" -c "import vllm" 2>/dev/null; then
  echo "vLLM not installed. For CPU (not the CUDA wheel), run:"
  echo "  python3 -m venv \$HOME/.venvs/async-rl && source \$HOME/.venvs/async-rl/bin/activate"
  echo "  pip install -U pip"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
  echo "  pip install -e \".[dev]\""
  echo "  VLLM_VERSION=0.28.0"
  echo "  pip install \"https://github.com/vllm-project/vllm/releases/download/v\${VLLM_VERSION}/vllm-\${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl\" \\"
  echo "    --extra-index-url https://download.pytorch.org/whl/cpu"
  exit 1
fi

echo "start vLLM (CPU) serving $MODEL  KV_BYTES=${KV_CACHE_BYTES}  LIBDIR=$LIBDIR  PY=$PY"
VLLM_PID=""
MGR_PID=""
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$VLLM_PORT" \
  --max-model-len 256 \
  --dtype bfloat16 \
  --enforce-eager \
  --kv-cache-memory-bytes "$KV_CACHE_BYTES" &
VLLM_PID=$!

cleanup() {
  [[ -n "${VLLM_PID}" ]] && kill "$VLLM_PID" 2>/dev/null || true
  [[ -n "${MGR_PID}" ]] && kill "$MGR_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for vLLM on :${VLLM_PORT} (first download can take several minutes)..."
ready=0
for _ in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited early; check logs above"
    exit 1
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "timed out waiting for vLLM"
  exit 1
fi

go -C "$ROOT/manager" run ./cmd/manager -- \
  -addr "$MGR_ADDR" \
  -vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  -vllm-model "$MODEL" &
MGR_PID=$!

for _ in $(seq 1 50); do
  if curl -sf "http://${MGR_ADDR}/healthz" >/dev/null; then
    break
  fi
  sleep 0.1
done

export PYTHONPATH="${ROOT}/trainer${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${FULL_TRAIN:-0}" == "1" ]]; then
  echo "FULL_TRAIN=1: loading LoRA trainer (ensure enough RAM; prefer killing vLLM after generate on small machines)"
  "$PY" -m pip install -e ".[train]" >/dev/null
  "$PY" -m async_rl.train --config "$CONFIG"
else
  echo "dry-run against real CPU vLLM (set FULL_TRAIN=1 for optimizer step)"
  "$PY" -m async_rl.train --config "$CONFIG" --dry-run
fi

echo "phase1 CPU vLLM iteration ok"
