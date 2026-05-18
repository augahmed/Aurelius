package gpt2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	sharedmodel "github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/tokenizer"
)

type ParityFixture struct {
	Name              string                 `json:"name"`
	Prompt            string                 `json:"prompt"`
	ExpectedInput     []int                  `json:"expected_input_tokens"`
	ExpectedTopTokens []ExpectedTopToken     `json:"expected_top_tokens"`
	ExpectedLogits    []float64              `json:"expected_logits"`
	LogitTolerance    float64                `json:"logit_tolerance"`
	Metadata          map[string]interface{} `json:"metadata"`
}

type ExpectedTopToken struct {
	Token int     `json:"token"`
	Logit float64 `json:"logit"`
}

type TokenScore struct {
	Token int
	Logit float64
}

func BuildObservation(prompt string, tok tokenizer.Tokenizer, mdl sharedmodel.Model, topK int) (ParityFixture, error) {
	if tok == nil {
		return ParityFixture{}, fmt.Errorf("tokenizer is required")
	}
	if mdl == nil {
		return ParityFixture{}, fmt.Errorf("model is required")
	}
	if prompt == "" {
		return ParityFixture{}, fmt.Errorf("prompt is required")
	}
	if topK <= 0 {
		return ParityFixture{}, fmt.Errorf("topK must be positive")
	}

	input, err := tok.Encode(prompt)
	if err != nil {
		return ParityFixture{}, fmt.Errorf("encode prompt: %w", err)
	}
	logits, err := mdl.Forward(input, nil)
	if err != nil {
		return ParityFixture{}, fmt.Errorf("forward prompt: %w", err)
	}

	top := TopTokenScores(logits, topK)
	expected := make([]ExpectedTopToken, len(top))
	for i, score := range top {
		expected[i] = ExpectedTopToken{
			Token: score.Token,
			Logit: score.Logit,
		}
	}

	return ParityFixture{
		Prompt:            prompt,
		ExpectedInput:     input,
		ExpectedTopTokens: expected,
		LogitTolerance:    1e-5,
		Metadata: map[string]interface{}{
			"source": "aurelius-observation",
			"note":   "Do not use as an external parity reference. Generate the final fixture from a known-good implementation.",
		},
	}, nil
}

func LoadParityFixture(path string) (ParityFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ParityFixture{}, fmt.Errorf("read parity fixture %q: %w", path, err)
	}

	var fixture ParityFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return ParityFixture{}, fmt.Errorf("parse parity fixture %q: %w", path, err)
	}
	if fixture.Prompt == "" {
		return ParityFixture{}, fmt.Errorf("parity fixture %q must include a prompt", path)
	}
	if len(fixture.ExpectedTopTokens) == 0 && len(fixture.ExpectedLogits) == 0 {
		return ParityFixture{}, fmt.Errorf("parity fixture %q must include expected_top_tokens or expected_logits", path)
	}
	return fixture, nil
}

func ValidateParityFixture(fixture ParityFixture, tok tokenizer.Tokenizer, mdl sharedmodel.Model) error {
	if tok == nil {
		return fmt.Errorf("tokenizer is required")
	}
	if mdl == nil {
		return fmt.Errorf("model is required")
	}

	input, err := tok.Encode(fixture.Prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}
	if len(fixture.ExpectedInput) > 0 && !sameTokens(input, fixture.ExpectedInput) {
		return fmt.Errorf("encoded input = %v, want %v", input, fixture.ExpectedInput)
	}

	logits, err := mdl.Forward(input, nil)
	if err != nil {
		return fmt.Errorf("forward prompt: %w", err)
	}

	tolerance := fixture.LogitTolerance
	if tolerance == 0 {
		tolerance = 1e-5
	}

	if len(fixture.ExpectedLogits) > 0 {
		if len(logits) != len(fixture.ExpectedLogits) {
			return fmt.Errorf("logit length = %d, want %d", len(logits), len(fixture.ExpectedLogits))
		}
		for i, want := range fixture.ExpectedLogits {
			if math.Abs(logits[i]-want) > tolerance {
				return fmt.Errorf("logit[%d] = %.8f, want %.8f", i, logits[i], want)
			}
		}
	}

	if len(fixture.ExpectedTopTokens) > 0 {
		got := TopTokenScores(logits, len(fixture.ExpectedTopTokens))
		for i, want := range fixture.ExpectedTopTokens {
			if i >= len(got) {
				return fmt.Errorf("top token count = %d, want at least %d", len(got), len(fixture.ExpectedTopTokens))
			}
			if got[i].Token != want.Token {
				return fmt.Errorf("top token[%d] = %d, want %d", i, got[i].Token, want.Token)
			}
			if math.Abs(got[i].Logit-want.Logit) > tolerance {
				return fmt.Errorf("top token logit[%d] = %.8f, want %.8f", i, got[i].Logit, want.Logit)
			}
		}
	}

	return nil
}

func TopTokenScores(logits []float64, k int) []TokenScore {
	if k <= 0 || len(logits) == 0 {
		return nil
	}
	if k > len(logits) {
		k = len(logits)
	}

	scores := make([]TokenScore, 0, k)
	for token, logit := range logits {
		score := TokenScore{Token: token, Logit: logit}
		inserted := false
		for i := range scores {
			if logit > scores[i].Logit {
				scores = append(scores, TokenScore{})
				copy(scores[i+1:], scores[i:])
				scores[i] = score
				inserted = true
				break
			}
		}
		if inserted {
			if len(scores) > k {
				scores = scores[:k]
			}
			continue
		}
		if len(scores) < k {
			scores = append(scores, score)
		}
	}

	return scores
}

func sameTokens(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
