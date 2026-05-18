package gpt2

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

func TestLoadParityFixture(t *testing.T) {
	path := filepath.Join("testdata", "tiny_parity_fixture.json")

	fixture, err := LoadParityFixture(path)
	if err != nil {
		t.Fatalf("LoadParityFixture error: %v", err)
	}

	if fixture.Prompt != "hello!" {
		t.Fatalf("Prompt = %q, want %q", fixture.Prompt, "hello!")
	}
	if len(fixture.ExpectedTopTokens) != 3 {
		t.Fatalf("ExpectedTopTokens length = %d, want %d", len(fixture.ExpectedTopTokens), 3)
	}
}

func TestValidateParityFixture(t *testing.T) {
	fixture, err := LoadParityFixture(filepath.Join("testdata", "tiny_parity_fixture.json"))
	if err != nil {
		t.Fatalf("LoadParityFixture error: %v", err)
	}

	tok, err := NewFixtureTokenizer()
	if err != nil {
		t.Fatalf("NewFixtureTokenizer error: %v", err)
	}

	model, err := NewModel(tinyConfig(), tinyStateDict())
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}

	if err := ValidateParityFixture(fixture, tok, model); err != nil {
		t.Fatalf("ValidateParityFixture error: %v", err)
	}
}

func TestTopTokenScores(t *testing.T) {
	got := TopTokenScores([]float64{0.1, 0.9, -0.5, 0.8}, 3)

	want := []TokenScore{
		{Token: 1, Logit: 0.9},
		{Token: 3, Logit: 0.8},
		{Token: 0, Logit: 0.1},
	}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestBuildObservation(t *testing.T) {
	tok, err := NewFixtureTokenizer()
	if err != nil {
		t.Fatalf("NewFixtureTokenizer error: %v", err)
	}
	model, err := NewModel(tinyConfig(), tinyStateDict())
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}

	observation, err := BuildObservation("hello!", tok, model, 3)
	if err != nil {
		t.Fatalf("BuildObservation error: %v", err)
	}
	if len(observation.ExpectedInput) != 2 || observation.ExpectedInput[0] != 7 || observation.ExpectedInput[1] != 8 {
		t.Fatalf("ExpectedInput = %v, want [7 8]", observation.ExpectedInput)
	}
	if len(observation.ExpectedTopTokens) != 3 {
		t.Fatalf("ExpectedTopTokens length = %d, want %d", len(observation.ExpectedTopTokens), 3)
	}
	if observation.ExpectedTopTokens[0].Token != 8 {
		t.Fatalf("top token = %d, want %d", observation.ExpectedTopTokens[0].Token, 8)
	}

	if _, err := json.Marshal(observation); err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}
}

func NewFixtureTokenizer() (tokenizer.Tokenizer, error) {
	return tokenizer.NewBPETokenizer(testVocab(), testMerges())
}
