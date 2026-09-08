package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Tanishk946/Async-RL/manager/internal/rollout"
	"github.com/Tanishk946/Async-RL/manager/internal/types"
	"github.com/Tanishk946/Async-RL/manager/internal/vllm"
)

func TestHealthAndPolicy(t *testing.T) {
	s := New(rollout.New(vllm.MockCompleter{}, "v0"))
	srv := httptest.NewServer(s.Mux())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("health %d", resp.StatusCode)
	}

	body, _ := json.Marshal(types.PolicyUpdate{PolicyVersion: "v1"})
	resp, err = http.Post(srv.URL+"/v1/policy", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("policy %d", resp.StatusCode)
	}
}

func TestRolloutHTTP(t *testing.T) {
	s := New(rollout.New(vllm.MockCompleter{}, "v0"))
	srv := httptest.NewServer(s.Mux())
	defer srv.Close()

	req := types.RolloutRequest{
		PolicyVersion: "v0",
		GroupSize:     2,
		Prompts:       []types.Prompt{{ID: "p0", Text: "Compute 3 + 4."}},
		Sampling:      types.SamplingParams{MaxTokens: 16, Temperature: 0.8, TopP: 1},
	}
	raw, _ := json.Marshal(req)
	resp, err := http.Post(srv.URL+"/v1/rollout", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out types.RolloutResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Trajectories) != 2 {
		t.Fatalf("got %d", len(out.Trajectories))
	}
}
