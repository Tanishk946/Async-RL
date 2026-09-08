package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Tanishk946/Async-RL/manager/internal/types"
)

type Client struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

func New(baseURL, model string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

type completionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	N           int      `json:"n"`
	Stop        []string `json:"stop,omitempty"`
	Logprobs    int      `json:"logprobs"`
}

type completionResponse struct {
	Choices []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
		Logprobs     *struct {
			Tokens        []string   `json:"tokens"`
			TokenLogprobs []float64  `json:"token_logprobs"`
			TokenIDs      []int      `json:"token_ids"`
		} `json:"logprobs"`
	} `json:"choices"`
}

func (c *Client) Complete(ctx context.Context, prompt string, n int, sp types.SamplingParams) ([]types.Completion, error) {
	if n < 1 {
		n = 1
	}
	body, err := json.Marshal(completionRequest{
		Model:       c.model,
		Prompt:      prompt,
		MaxTokens:   sp.MaxTokens,
		Temperature: sp.Temperature,
		TopP:        sp.TopP,
		N:           n,
		Stop:        sp.Stop,
		Logprobs:    1,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("vllm %s: %s", resp.Status, string(raw))
	}
	var parsed completionResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	out := make([]types.Completion, 0, len(parsed.Choices))
	for _, ch := range parsed.Choices {
		comp := types.Completion{
			Text:         ch.Text,
			FinishReason: ch.FinishReason,
		}
		if ch.Logprobs != nil {
			comp.Logprobs = ch.Logprobs.TokenLogprobs
			comp.TokenIDs = ch.Logprobs.TokenIDs
			if len(comp.TokenIDs) == 0 && len(ch.Logprobs.Tokens) > 0 {
				comp.TokenIDs = make([]int, len(ch.Logprobs.Tokens))
			}
		}
		out = append(out, comp)
	}
	return out, nil
}
