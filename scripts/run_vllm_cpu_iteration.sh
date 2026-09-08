#!/usr/bin/env bash
# CPU vLLM loop: generate with vLLM CPU backend, then GRPO.
# Default: --dry-run (real completions, no second HF model copy).
# Full train: FULL_TRAIN=1 ./scripts/run_vllm_cpu_iteration.sh
#   (needs more RAM; may OOM on ~8GB if vLLM stays up during train)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
MGR_ADDR="${MGR_ADDR:-127.0.0.1:8080}"
CONFIG="${CONFIG:-configs/phase1_cpu.yaml}"
# KV cache budget in GB for the CPU backend
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}"

if ! python3 -c "import vllm" 2>/dev/null; then
  echo "vLLM not installed. For CPU (not the CUDA wheel), run:"
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install -U pip"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
  echo "  pip install -e \".[dev]\""
  echo "  VLLM_VERSION=0.28.0"
  echo "  pip install \"https://github.com/vllm-project/vllm/releases/download/v\${VLLM_VERSION}/vllm-\${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl\" \\"
  echo "    --extra-index-url https://download.pytorch.org/whl/cpu"
  exit 1
fi

echo "start vLLM (CPU) serving $MODEL  KV=${VLLM_CPU_KVCACHE_SPACE}G"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$VLLM_PORT" \
  --max-model-len 512 \
  --dtype bfloat16 \
  --enforce-eager &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" "$MGR_PID" 2>/dev/null || true
  wait "$VLLM_PID" "$MGR_PID" 2>/dev/null || true
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
  pip install -e ".[train]" >/dev/null
  python3 -m async_rl.train --config "$CONFIG"
else
  echo "dry-run against real CPU vLLM (set FULL_TRAIN=1 for optimizer step)"
  python3 -m async_rl.train --config "$CONFIG" --dry-run
fi

echo "phase1 CPU vLLM iteration ok"
