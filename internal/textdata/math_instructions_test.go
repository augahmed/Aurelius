package textdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

func TestArithmeticExamplesToInstructionsInstructionFormat(t *testing.T) {
	examples := []arithmetic.Example{{
		Prompt:     "2 + 2 = ",
		Completion: "4",
		Operation:  "add",
		Level:      1,
		Template:   "equation",
	}}
	instructions := ArithmeticExamplesToInstructions(examples, "Be exact.", MathInstructionFormatInstruction)
	if len(instructions) != 1 {
		t.Fatalf("len(instructions) = %d, want 1", len(instructions))
	}
	if instructions[0].System != "Be exact." {
		t.Fatalf("System = %q, want custom system", instructions[0].System)
	}
	if instructions[0].Instruction != "Solve: 2 + 2 =" {
		t.Fatalf("Instruction = %q, want solve-form instruction", instructions[0].Instruction)
	}
	if instructions[0].Output != "4" {
		t.Fatalf("Output = %q, want 4", instructions[0].Output)
	}
}

func TestArithmeticExamplesToInstructionsChatFormat(t *testing.T) {
	examples := []arithmetic.Example{{
		Prompt:     "What is 7 * 8? ",
		Completion: "56",
		Operation:  "mul",
		Level:      4,
		Template:   "question",
	}}
	instructions := ArithmeticExamplesToInstructions(examples, "", MathInstructionFormatChat)
	if instructions[0].Prompt != "User: What is 7 * 8?\n\nAssistant:" {
		t.Fatalf("Prompt = %q, want chat prompt", instructions[0].Prompt)
	}
	if instructions[0].Completion != "56" {
		t.Fatalf("Completion = %q, want 56", instructions[0].Completion)
	}
}

func TestArithmeticExamplesToInstructionsCompactFormat(t *testing.T) {
	examples := []arithmetic.Example{{
		Prompt:     "What is 7 * 8? ",
		Completion: "56",
		Operation:  "mul",
		Level:      4,
		Template:   "question",
	}}
	instructions := ArithmeticExamplesToInstructions(examples, "", MathInstructionFormatCompact)
	if instructions[0].Prompt != "User: What is 7 * 8?\nAssistant:" {
		t.Fatalf("Prompt = %q, want compact prompt", instructions[0].Prompt)
	}
	if instructions[0].Completion != "56" {
		t.Fatalf("Completion = %q, want 56", instructions[0].Completion)
	}
}

func TestGenerateMathInstructionDataset(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	output := filepath.Join(root, "instructions")
	if err := arithmetic.GenerateDataset(source, arithmetic.GenerateConfig{
		TrainCount: 2,
		ValCount:   1,
		Operations: []string{"add"},
		Levels:     []int{1},
		Templates:  []string{"equation"},
		Seed:       11,
	}); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	report, err := GenerateMathInstructionDataset(MathInstructionConfig{
		DataDir:   source,
		OutputDir: output,
		System:    "Answer exactly.",
		Format:    MathInstructionFormatInstruction,
	})
	if err != nil {
		t.Fatalf("GenerateMathInstructionDataset error: %v", err)
	}
	if report.TrainCount != 2 || report.ValCount != 1 {
		t.Fatalf("report = %+v, want 2 train / 1 val", report)
	}
	examples, err := LoadInstructionExamples([]string{filepath.Join(output, "train.jsonl")})
	if err != nil {
		t.Fatalf("LoadInstructionExamples error: %v", err)
	}
	if len(examples) != 2 {
		t.Fatalf("len(examples) = %d, want 2", len(examples))
	}
	if examples[0].System != "Answer exactly." || !strings.HasPrefix(examples[0].Instruction, "Solve:") {
		t.Fatalf("example = %+v, want exact solve instruction", examples[0])
	}
	if _, err := os.Stat(filepath.Join(output, "meta.json")); err != nil {
		t.Fatalf("meta.json missing: %v", err)
	}
}
