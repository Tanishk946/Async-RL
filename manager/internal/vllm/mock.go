package vllm

import (
	"context"
	"fmt"
	"hash/fnv"

	"github.com/Tanishk946/Async-RL/manager/internal/env"
	"github.com/Tanishk946/Async-RL/manager/internal/types"
)

// MockCompleter produces mixed correct/incorrect #### answers so GRPO groups have variance.
type MockCompleter struct{}

func (MockCompleter) Complete(_ context.Context, prompt string, n int, _ types.SamplingParams) ([]types.Completion, error) {
	gold, ok := env.GoldFromPrompt(prompt)
	if !ok {
		gold = 0
	}
	out := make([]types.Completion, n)
	for i := 0; i < n; i++ {
		h := fnv.New32a()
		_, _ = h.Write([]byte(prompt))
		_, _ = fmt.Fprintf(h, "#%d", i)
		correct := h.Sum32()%2 == 0
		val := gold
		if !correct {
			val = gold + 1
		}
		text := fmt.Sprintf("#### %d", val)
		out[i] = types.Completion{
			Text:         text,
			TokenIDs:     []int{1, 2, 3},
			Logprobs:     []float64{-0.2, -0.1, -0.3},
			FinishReason: "stop",
		}
	}
	return out, nil
}
