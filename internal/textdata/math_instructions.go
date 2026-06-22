package textdata

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

const (
	MathInstructionFormatInstruction = "instruction"
	MathInstructionFormatChat        = "chat"
	MathInstructionFormatCompact     = "compact"
	DefaultMathInstructionSystem     = "Answer math questions accurately and concisely."
)

type MathInstructionConfig struct {
	DataDir   string
	OutputDir string
	System    string
	Format    string
}

type MathInstructionReport struct {
	TrainCount int    `json:"train_count"`
	ValCount   int    `json:"val_count"`
	Format     string `json:"format"`
	System     string `json:"system,omitempty"`
}

type MathInstructionMetadata struct {
	SourceDataDir string `json:"source_data_dir"`
	TrainCount    int    `json:"train_count"`
	ValCount      int    `json:"val_count"`
	Format        string `json:"format"`
	System        string `json:"system,omitempty"`
}

func GenerateMathInstructionDataset(cfg MathInstructionConfig) (MathInstructionReport, error) {
	if strings.TrimSpace(cfg.DataDir) == "" {
		return MathInstructionReport{}, fmt.Errorf("data dir is required")
	}
	if strings.TrimSpace(cfg.OutputDir) == "" {
		return MathInstructionReport{}, fmt.Errorf("output dir is required")
	}
	format := cfg.Format
	if format == "" {
		format = MathInstructionFormatInstruction
	}
	if format != MathInstructionFormatInstruction && format != MathInstructionFormatChat && format != MathInstructionFormatCompact {
		return MathInstructionReport{}, fmt.Errorf("unsupported instruction format %q", cfg.Format)
	}
	system := cfg.System
	if system == "" {
		system = DefaultMathInstructionSystem
	}
	trainExamples, err := arithmetic.LoadExamples(filepath.Join(cfg.DataDir, "train.jsonl"))
	if err != nil {
		return MathInstructionReport{}, fmt.Errorf("load source train: %w", err)
	}
	valExamples, err := arithmetic.LoadExamples(filepath.Join(cfg.DataDir, "val.jsonl"))
	if err != nil {
		return MathInstructionReport{}, fmt.Errorf("load source val: %w", err)
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return MathInstructionReport{}, fmt.Errorf("create output dir: %w", err)
	}
	trainInstructions := ArithmeticExamplesToInstructions(trainExamples, system, format)
	valInstructions := ArithmeticExamplesToInstructions(valExamples, system, format)
	if err := WriteInstructionExamples(filepath.Join(cfg.OutputDir, "train.jsonl"), trainInstructions); err != nil {
		return MathInstructionReport{}, err
	}
	if err := WriteInstructionExamples(filepath.Join(cfg.OutputDir, "val.jsonl"), valInstructions); err != nil {
		return MathInstructionReport{}, err
	}
	report := MathInstructionReport{
		TrainCount: len(trainInstructions),
		ValCount:   len(valInstructions),
		Format:     format,
		System:     system,
	}
	meta := MathInstructionMetadata{
		SourceDataDir: cfg.DataDir,
		TrainCount:    report.TrainCount,
		ValCount:      report.ValCount,
		Format:        report.Format,
		System:        report.System,
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return MathInstructionReport{}, fmt.Errorf("marshal instruction meta: %w", err)
	}
	if err := os.WriteFile(filepath.Join(cfg.OutputDir, "meta.json"), data, 0o644); err != nil {
		return MathInstructionReport{}, fmt.Errorf("write instruction meta: %w", err)
	}
	return report, nil
}

func ArithmeticExamplesToInstructions(examples []arithmetic.Example, system string, format string) []InstructionExample {
	out := make([]InstructionExample, len(examples))
	for i, example := range examples {
		instruction := arithmeticPromptToInstruction(example.Prompt)
		output := strings.TrimSpace(example.Completion)
		if format == MathInstructionFormatChat {
			out[i] = InstructionExample{
				Prompt:     "User: " + instruction + "\n\nAssistant:",
				Completion: output,
			}
			continue
		}
		if format == MathInstructionFormatCompact {
			out[i] = InstructionExample{
				Prompt:     "User: " + instruction + "\nAssistant:",
				Completion: output,
			}
			continue
		}
		out[i] = InstructionExample{
			System:      system,
			Instruction: instruction,
			Output:      output,
		}
	}
	return out
}

func WriteInstructionExamples(path string, examples []InstructionExample) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create instruction file %q: %w", path, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	encoder := json.NewEncoder(writer)
	for _, example := range examples {
		if err := example.Validate(); err != nil {
			return fmt.Errorf("invalid instruction example for %q: %w", path, err)
		}
		if err := encoder.Encode(example); err != nil {
			return fmt.Errorf("write instruction file %q: %w", path, err)
		}
	}
	if err := writer.Flush(); err != nil {
		return fmt.Errorf("flush instruction file %q: %w", path, err)
	}
	return nil
}

func arithmeticPromptToInstruction(prompt string) string {
	trimmed := strings.TrimSpace(prompt)
	if strings.HasPrefix(trimmed, "What is ") || strings.HasPrefix(trimmed, "Solve:") || strings.HasPrefix(trimmed, "Derrivative:") {
		return trimmed
	}
	if strings.HasSuffix(trimmed, "=") {
		return "Solve: " + trimmed
	}
	return trimmed
}
