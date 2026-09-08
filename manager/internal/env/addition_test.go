package env

import "testing"

func TestGoldFromPrompt(t *testing.T) {
	got, ok := GoldFromPrompt("Compute 17 + 25.\nPut the final answer")
	if !ok || got != 42 {
		t.Fatalf("got %d ok=%v", got, ok)
	}
}

func TestRewardExact(t *testing.T) {
	prompt := "Compute 17 + 25."
	if Reward(prompt, "some reasoning\n#### 42") != 1 {
		t.Fatal("expected 1")
	}
	if Reward(prompt, "#### 41") != 0 {
		t.Fatal("expected 0")
	}
	if Reward(prompt, "the answer is 42") != 0 {
		t.Fatal("format miss should be 0")
	}
}
