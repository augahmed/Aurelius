package textdata

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

func TestBuildPretrainingSequences(t *testing.T) {
	sequences, err := BuildPretrainingSequences("hello", tokenizer.NewByteTokenizer(), BuildConfig{
		ContextSize: 4,
		Stride:      1,
	})
	if err != nil {
		t.Fatalf("BuildPretrainingSequences error: %v", err)
	}
	if len(sequences) != 4 {
		t.Fatalf("len(sequences) = %d, want 4", len(sequences))
	}
	if sequences[0].Target != int('e') {
		t.Fatalf("first target = %d, want %d", sequences[0].Target, int('e'))
	}
	if len(sequences[0].Context) != 4 {
		t.Fatalf("context len = %d, want 4", len(sequences[0].Context))
	}
}

func TestLoadTextExpandsDirectories(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("alpha"), 0o644); err != nil {
		t.Fatalf("WriteFile a error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "b.md"), []byte("bravo"), 0o644); err != nil {
		t.Fatalf("WriteFile b error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "skip.json"), []byte("skip"), 0o644); err != nil {
		t.Fatalf("WriteFile skip error: %v", err)
	}
	text, err := LoadText([]string{root})
	if err != nil {
		t.Fatalf("LoadText error: %v", err)
	}
	if text != "alpha\n\nbravo" {
		t.Fatalf("text = %q, want joined text", text)
	}
}

func TestBuildInstructionSequences(t *testing.T) {
	examples := []InstructionExample{{
		System:      "Be brief.",
		Instruction: "Say hello.",
		Output:      "hello",
	}}
	sequences, err := BuildInstructionSequences(examples, tokenizer.NewByteTokenizer(), 16)
	if err != nil {
		t.Fatalf("BuildInstructionSequences error: %v", err)
	}
	if len(sequences) != len("hello\n") {
		t.Fatalf("len(sequences) = %d, want completion target count", len(sequences))
	}
}

func TestLoadInstructionExamples(t *testing.T) {
	path := filepath.Join(t.TempDir(), "instructions.jsonl")
	raw := `{"instruction":"Add.","input":"2 + 2","output":"4"}` + "\n" +
		`{"prompt":"User: hi\n\nAssistant:","completion":"hello"}` + "\n"
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}
	examples, err := LoadInstructionExamples([]string{path})
	if err != nil {
		t.Fatalf("LoadInstructionExamples error: %v", err)
	}
	if len(examples) != 2 {
		t.Fatalf("len(examples) = %d, want 2", len(examples))
	}
	prompt, completion := examples[0].PromptCompletion()
	if prompt != "User: Add.\n2 + 2\n\nAssistant:" || completion != "4" {
		t.Fatalf("prompt/completion = %q / %q", prompt, completion)
	}
}
