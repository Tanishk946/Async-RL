package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/Tanishk946/Async-RL/manager/internal/api"
	"github.com/Tanishk946/Async-RL/manager/internal/rollout"
	"github.com/Tanishk946/Async-RL/manager/internal/vllm"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	mock := flag.Bool("mock", false, "use mock completer (no vLLM)")
	vllmURL := flag.String("vllm-url", "http://127.0.0.1:8000/v1", "vLLM OpenAI base URL")
	vllmModel := flag.String("vllm-model", "Qwen/Qwen2.5-0.5B-Instruct", "served model name")
	policy := flag.String("policy-version", "v0", "initial policy_version stamp")
	flag.Parse()

	var completer vllm.Completer
	if *mock {
		log.Printf("rollout manager mock mode on %s", *addr)
		completer = vllm.MockCompleter{}
	} else {
		log.Printf("rollout manager vLLM=%s model=%s", *vllmURL, *vllmModel)
		completer = vllm.New(*vllmURL, *vllmModel)
	}

	mgr := rollout.New(completer, *policy)
	srv := api.New(mgr)
	log.Fatal(http.ListenAndServe(*addr, srv.Mux()))
}
