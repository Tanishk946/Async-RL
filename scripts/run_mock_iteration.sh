#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

go -C "$ROOT/manager" test ./...

PYTHONPATH="$ROOT/trainer" python3 -m pytest -q "$ROOT/trainer/tests"

BIN="$(mktemp -d)/async-rl-manager"
go -C "$ROOT/manager" build -o "$BIN" ./cmd/manager
"$BIN" -mock -addr 127.0.0.1:8080 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; rm -f "$BIN"; }
trap cleanup EXIT

for _ in $(seq 1 50); do
  if curl -sf http://127.0.0.1:8080/healthz >/dev/null; then
    break
  fi
  sleep 0.1
done

PYTHONPATH="$ROOT/trainer" python3 -m async_rl.train --config "$ROOT/configs/phase1.yaml" --dry-run
echo "phase1 mock iteration ok"
