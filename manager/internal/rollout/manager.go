package rollout

import (
	"context"
	"fmt"
	"sync"

	"github.com/Tanishk946/Async-RL/manager/internal/env"
	"github.com/Tanishk946/Async-RL/manager/internal/types"
	"github.com/Tanishk946/Async-RL/manager/internal/vllm"
)

type Manager struct {
	completer     vllm.Completer
	policyVersion string
	mu            sync.RWMutex
}

func New(completer vllm.Completer, policyVersion string) *Manager {
	return &Manager{completer: completer, policyVersion: policyVersion}
}

func (m *Manager) PolicyVersion() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.policyVersion
}

func (m *Manager) SetPolicy(version string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.policyVersion = version
}

func (m *Manager) Rollout(ctx context.Context, req types.RolloutRequest) (*types.RolloutResponse, error) {
	if req.GroupSize < 1 {
		return nil, fmt.Errorf("group_size must be >= 1")
	}
	if len(req.Prompts) == 0 {
		return nil, fmt.Errorf("prompts required")
	}
	version := req.PolicyVersion
	if version == "" {
		version = m.PolicyVersion()
	}

	type result struct {
		traj []types.Trajectory
		err  error
	}
	ch := make(chan result, len(req.Prompts))
	for _, p := range req.Prompts {
		p := p
		go func() {
			comps, err := m.completer.Complete(ctx, p.Text, req.GroupSize, req.Sampling)
			if err != nil {
				ch <- result{err: fmt.Errorf("prompt %s: %w", p.ID, err)}
				return
			}
			traj := make([]types.Trajectory, 0, len(comps))
			for _, c := range comps {
				traj = append(traj, types.Trajectory{
					PromptID:      p.ID,
					PolicyVersion: version,
					PromptText:    p.Text,
					Completion:    c.Text,
					TokenIDs:      c.TokenIDs,
					Logprobs:      c.Logprobs,
					Reward:        env.Reward(p.Text, c.Text),
					FinishReason:  c.FinishReason,
				})
			}
			ch <- result{traj: traj}
		}()
	}

	out := make([]types.Trajectory, 0, len(req.Prompts)*req.GroupSize)
	for i := 0; i < len(req.Prompts); i++ {
		r := <-ch
		if r.err != nil {
			return nil, r.err
		}
		out = append(out, r.traj...)
	}
	return &types.RolloutResponse{PolicyVersion: version, Trajectories: out}, nil
}
