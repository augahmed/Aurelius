package arithmetic

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

func TestGenerateDatasetAndLoadExamples(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		MinOperand: 0,
		MaxOperand: 9,
		Operations: []string{"add", "sub"},
		Seed:       7,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}

	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples(train) error: %v", err)
	}
	val, err := LoadExamples(filepath.Join(dir, "val.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples(val) error: %v", err)
	}
	if len(train) != 4 {
		t.Fatalf("len(train) = %d, want 4", len(train))
	}
	if len(val) != 2 {
		t.Fatalf("len(val) = %d, want 2", len(val))
	}
	for _, example := range train {
		if strings.TrimSpace(example.Prompt) == "" || strings.TrimSpace(example.Completion) == "" {
			t.Fatalf("invalid example: %+v", example)
		}
	}
}

func TestBuildTrainingSequences(t *testing.T) {
	examples := []Example{{
		Prompt:     "2 + 3 = ",
		Completion: "5",
		Operation:  "add",
	}}
	sequences, err := BuildTrainingSequences(examples, tokenizer.NewByteTokenizer(), 4)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}
	if len(sequences) == 0 {
		t.Fatal("expected at least one training sequence")
	}
	last := sequences[len(sequences)-1]
	if last.Target != int('\n') {
		t.Fatalf("last target = %d, want newline token", last.Target)
	}
	if len(last.Context) != 4 {
		t.Fatalf("context length = %d, want 4", len(last.Context))
	}
}
