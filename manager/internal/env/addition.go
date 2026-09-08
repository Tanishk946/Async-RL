package env

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	computeRe = regexp.MustCompile(`(?i)compute\s+(-?\d+)\s*\+\s*(-?\d+)`)
	answerRe  = regexp.MustCompile(`####\s*(-?\d+)`)
)

// GoldFromPrompt parses "Compute A + B" from trainer-rendered text.
func GoldFromPrompt(prompt string) (int, bool) {
	m := computeRe.FindStringSubmatch(prompt)
	if m == nil {
		return 0, false
	}
	a, errA := strconv.Atoi(m[1])
	b, errB := strconv.Atoi(m[2])
	if errA != nil || errB != nil {
		return 0, false
	}
	return a + b, true
}

// ParseAnswer returns the last #### N in the completion.
func ParseAnswer(completion string) (int, bool) {
	matches := answerRe.FindAllStringSubmatch(completion, -1)
	if len(matches) == 0 {
		return 0, false
	}
	n, err := strconv.Atoi(matches[len(matches)-1][1])
	if err != nil {
		return 0, false
	}
	return n, true
}

// Reward is 1 if the parsed answer matches gold, else 0.
func Reward(prompt, completion string) float64 {
	gold, ok := GoldFromPrompt(prompt)
	if !ok {
		return 0
	}
	got, ok := ParseAnswer(strings.TrimSpace(completion))
	if !ok {
		return 0
	}
	if got == gold {
		return 1
	}
	return 0
}
