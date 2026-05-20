package arithmetic

import (
	"fmt"
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
	if len(sequences) != 2 {
		t.Fatalf("len(sequences) = %d, want completion plus newline targets", len(sequences))
	}
	if sequences[0].Target != int('5') {
		t.Fatalf("first target = %d, want completion token", sequences[0].Target)
	}
	last := sequences[len(sequences)-1]
	if last.Target != int('\n') {
		t.Fatalf("last target = %d, want newline token", last.Target)
	}
	if len(last.Context) != 4 {
		t.Fatalf("context length = %d, want 4", len(last.Context))
	}
}

func TestGenerateDatasetIncludesCurriculumMetadata(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 12,
		ValCount:   6,
		Operations: []string{"add", "sub"},
		Levels:     []int{1, 2, 3},
		Seed:       11,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}

	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	seen := map[int]bool{}
	seenPairs := map[string]bool{}
	for _, example := range train {
		seen[example.Level] = true
		seenPairs[fmt.Sprintf("%d/%s", example.Level, example.Operation)] = true
		if example.Template == "" {
			t.Fatalf("example missing template: %+v", example)
		}
		if example.MinOperand > example.MaxOperand {
			t.Fatalf("invalid operand metadata: %+v", example)
		}
		if example.Level == 3 && example.Operation == "add" && !example.RequiresCarry {
			t.Fatalf("level 3 addition should require carry: %+v", example)
		}
		if example.Level == 3 && example.Operation == "sub" && !example.RequiresBorrow {
			t.Fatalf("level 3 subtraction should require borrow: %+v", example)
		}
		if example.Level == 2 && (example.RequiresCarry || example.RequiresBorrow) {
			t.Fatalf("level 2 should avoid carry and borrow: %+v", example)
		}
	}
	for _, level := range []int{1, 2, 3} {
		if !seen[level] {
			t.Fatalf("missing level %d in generated dataset", level)
		}
	}
	for _, pair := range []string{"1/add", "1/sub", "2/add", "2/sub", "3/add", "3/sub"} {
		if !seenPairs[pair] {
			t.Fatalf("missing level/operation pair %s in generated dataset", pair)
		}
	}
}

func TestGenerateDatasetRejectsIncompatibleLevelAndOperation(t *testing.T) {
	err := GenerateDataset(t.TempDir(), GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		Operations: []string{"mul"},
		Levels:     []int{1},
		Seed:       1,
	})
	if err == nil {
		t.Fatal("expected incompatible level and operation error")
	}
}
