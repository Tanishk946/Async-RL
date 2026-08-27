package rollout

import (
	"context"
	"testing"

	"github.com/Tanishk946/Async-RL/manager/internal/types"
	"github.com/Tanishk946/Async-RL/manager/internal/vllm"
)

func TestMockRolloutGroup(t *testing.T) {
	m := New(vllm.MockCompleter{}, "v0")
	resp, err := m.Rollout(context.Background(), types.RolloutRequest{
		PolicyVersion: "v0",
		GroupSize:     4,
		Prompts: []types.Prompt{
			{ID: "p0", Text: "Compute 10 + 5.\nPut the final answer on its own line as: #### <number>"},
		},
		Sampling: types.SamplingParams{MaxTokens: 32, Temperature: 0.8, TopP: 0.95},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Trajectories) != 4 {
		t.Fatalf("got %d traj", len(resp.Trajectories))
	}
	var saw0, saw1 bool
	for _, tr := range resp.Trajectories {
		if tr.Reward == 0 {
			saw0 = true
		}
		if tr.Reward == 1 {
			saw1 = true
		}
		if tr.PolicyVersion != "v0" {
			t.Fatalf("version %s", tr.PolicyVersion)
		}
	}
	if !saw0 || !saw1 {
		t.Fatal("mock should produce mixed rewards in a group of 4")
	}
}
