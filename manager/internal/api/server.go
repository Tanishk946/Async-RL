package api

import (
	"encoding/json"
	"net/http"

	"github.com/Tanishk946/Async-RL/manager/internal/rollout"
	"github.com/Tanishk946/Async-RL/manager/internal/types"
)

type Server struct {
	manager *rollout.Manager
}

func New(manager *rollout.Manager) *Server {
	return &Server{manager: manager}
}

func (s *Server) Mux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/rollout", s.rollout)
	mux.HandleFunc("POST /v1/policy", s.policy)
	return mux
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":          "ok",
		"policy_version":  s.manager.PolicyVersion(),
	})
}

func (s *Server) rollout(w http.ResponseWriter, r *http.Request) {
	var req types.RolloutRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp, err := s.manager.Rollout(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) policy(w http.ResponseWriter, r *http.Request) {
	var req types.PolicyUpdate
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.PolicyVersion == "" {
		http.Error(w, "policy_version required", http.StatusBadRequest)
		return
	}
	s.manager.SetPolicy(req.PolicyVersion)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"policy_version":  s.manager.PolicyVersion(),
		"checkpoint_path": req.CheckpointPath,
	})
}
