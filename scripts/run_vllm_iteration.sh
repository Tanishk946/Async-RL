#!/usr/bin/env bash
# Sequential one-GPU loop: vLLM generate -> stop -> train -> reload.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "start vLLM serving $MODEL"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$VLLM_PORT" \
  --max-model-len 1024 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" "$MGR_PID" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 || \
     curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

go -C manager run ./cmd/manager -- \
  -addr 127.0.0.1:8080 \
  -vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  -vllm-model "$MODEL" &
MGR_PID=$!
sleep 1

python3 -m async_rl.train --config configs/phase1.yaml
echo "stop vLLM and reload from checkpoints/phase1 on the next run"
