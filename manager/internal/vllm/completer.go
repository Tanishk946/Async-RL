package vllm

import (
	"context"

	"github.com/Tanishk946/Async-RL/manager/internal/types"
)

// Completer is implemented by the real vLLM client and the mock.
type Completer interface {
	Complete(ctx context.Context, prompt string, n int, sp types.SamplingParams) ([]types.Completion, error)
}
