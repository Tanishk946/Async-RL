package types

type SamplingParams struct {
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	Stop        []string `json:"stop"`
}

type Prompt struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type RolloutRequest struct {
	PolicyVersion string          `json:"policy_version"`
	Prompts       []Prompt        `json:"prompts"`
	GroupSize     int             `json:"group_size"`
	Sampling      SamplingParams  `json:"sampling"`
}

type Trajectory struct {
	PromptID       string    `json:"prompt_id"`
	PolicyVersion  string    `json:"policy_version"`
	PromptText     string    `json:"prompt_text"`
	Completion     string    `json:"completion"`
	TokenIDs       []int     `json:"token_ids"`
	Logprobs       []float64 `json:"logprobs"`
	Reward         float64   `json:"reward"`
	FinishReason   string    `json:"finish_reason"`
}

type RolloutResponse struct {
	PolicyVersion string        `json:"policy_version"`
	Trajectories  []Trajectory  `json:"trajectories"`
}

type PolicyUpdate struct {
	PolicyVersion  string `json:"policy_version"`
	CheckpointPath string `json:"checkpoint_path,omitempty"`
}

type Completion struct {
	Text         string
	TokenIDs     []int
	Logprobs     []float64
	FinishReason string
}
