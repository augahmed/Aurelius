package arithmetic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

type Example struct {
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
	Operation  string `json:"operation"`
}

type GenerateConfig struct {
	TrainCount int
	ValCount   int
	MinOperand int
	MaxOperand int
	Operations []string
	Seed       int64
}

type Metadata struct {
	TrainCount int      `json:"train_count"`
	ValCount   int      `json:"val_count"`
	MinOperand int      `json:"min_operand"`
	MaxOperand int      `json:"max_operand"`
	Operations []string `json:"operations"`
	Seed       int64    `json:"seed"`
	Tokenizer  string   `json:"tokenizer"`
}

type SequenceExample struct {
	Context []int
	Target  int
}

var supportedOperations = []string{"add", "sub", "mul", "div"}

func (c GenerateConfig) Validate() error {
	if c.TrainCount <= 0 {
		return fmt.Errorf("train count must be positive")
	}
	if c.ValCount <= 0 {
		return fmt.Errorf("val count must be positive")
	}
	if c.MinOperand < 0 {
		return fmt.Errorf("min operand must be non-negative")
	}
	if c.MaxOperand < c.MinOperand {
		return fmt.Errorf("max operand must be greater than or equal to min operand")
	}
	if len(c.Operations) == 0 {
		return fmt.Errorf("at least one operation is required")
	}
	for _, operation := range c.Operations {
		if !slices.Contains(supportedOperations, operation) {
			return fmt.Errorf("unsupported operation %q", operation)
		}
	}
	return nil
}

func GenerateDataset(outputDir string, cfg GenerateConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	train := make([]Example, cfg.TrainCount)
	val := make([]Example, cfg.ValCount)
	for i := range train {
		train[i] = generateExample(cfg, rng)
	}
	for i := range val {
		val[i] = generateExample(cfg, rng)
	}

	if err := writeJSONL(filepath.Join(outputDir, "train.jsonl"), train); err != nil {
		return err
	}
	if err := writeJSONL(filepath.Join(outputDir, "val.jsonl"), val); err != nil {
		return err
	}

	meta := Metadata{
		TrainCount: cfg.TrainCount,
		ValCount:   cfg.ValCount,
		MinOperand: cfg.MinOperand,
		MaxOperand: cfg.MaxOperand,
		Operations: append([]string(nil), cfg.Operations...),
		Seed:       cfg.Seed,
		Tokenizer:  "byte",
	}
	metaPath := filepath.Join(outputDir, "meta.json")
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}
	if err := os.WriteFile(metaPath, data, 0o644); err != nil {
		return fmt.Errorf("write meta: %w", err)
	}
	return nil
}

func LoadExamples(path string) ([]Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset %q: %w", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	examples := make([]Example, 0)
	line := 0
	for scanner.Scan() {
		line++
		raw := strings.TrimSpace(scanner.Text())
		if raw == "" {
			continue
		}
		var example Example
		if err := json.Unmarshal([]byte(raw), &example); err != nil {
			return nil, fmt.Errorf("parse dataset line %d: %w", line, err)
		}
		if strings.TrimSpace(example.Prompt) == "" {
			return nil, fmt.Errorf("dataset line %d has empty prompt", line)
		}
		if strings.TrimSpace(example.Completion) == "" {
			return nil, fmt.Errorf("dataset line %d has empty completion", line)
		}
		examples = append(examples, example)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan dataset %q: %w", path, err)
	}
	return examples, nil
}

func BuildTrainingSequences(examples []Example, tok tokenizer.Tokenizer, contextSize int) ([]SequenceExample, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	if contextSize <= 0 {
		return nil, fmt.Errorf("context size must be positive")
	}
	sequences := make([]SequenceExample, 0)
	for _, example := range examples {
		text := example.Prompt + example.Completion + "\n"
		tokens, err := tok.Encode(text)
		if err != nil {
			return nil, fmt.Errorf("encode training text: %w", err)
		}
		for index, target := range tokens {
			context := make([]int, contextSize)
			start := max(0, index-contextSize)
			window := tokens[start:index]
			copy(context[contextSize-len(window):], window)
			sequences = append(sequences, SequenceExample{
				Context: context,
				Target:  target,
			})
		}
	}
	return sequences, nil
}

func writeJSONL(path string, examples []Example) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create dataset file %q: %w", path, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	encoder := json.NewEncoder(writer)
	for _, example := range examples {
		if err := encoder.Encode(example); err != nil {
			return fmt.Errorf("write dataset file %q: %w", path, err)
		}
	}
	if err := writer.Flush(); err != nil {
		return fmt.Errorf("flush dataset file %q: %w", path, err)
	}
	return nil
}

func generateExample(cfg GenerateConfig, rng *rand.Rand) Example {
	operation := cfg.Operations[rng.Intn(len(cfg.Operations))]
	left, right, answer := operandsForOperation(operation, cfg.MinOperand, cfg.MaxOperand, rng)
	switch rng.Intn(3) {
	case 0:
		return Example{
			Prompt:     fmt.Sprintf("%d %s %d = ", left, symbolForOperation(operation), right),
			Completion: fmt.Sprintf("%d", answer),
			Operation:  operation,
		}
	case 1:
		return Example{
			Prompt:     fmt.Sprintf("What is %d %s %d? ", left, symbolForOperation(operation), right),
			Completion: fmt.Sprintf("%d", answer),
			Operation:  operation,
		}
	default:
		return Example{
			Prompt:     fmt.Sprintf("Solve: %d %s %d = ", left, symbolForOperation(operation), right),
			Completion: fmt.Sprintf("%d", answer),
			Operation:  operation,
		}
	}
}

func operandsForOperation(operation string, minOperand, maxOperand int, rng *rand.Rand) (int, int, int) {
	switch operation {
	case "add":
		left := randomOperand(minOperand, maxOperand, rng)
		right := randomOperand(minOperand, maxOperand, rng)
		return left, right, left + right
	case "sub":
		left := randomOperand(minOperand, maxOperand, rng)
		right := randomOperand(minOperand, maxOperand, rng)
		if right > left {
			left, right = right, left
		}
		return left, right, left - right
	case "mul":
		left := randomOperand(minOperand, maxOperand, rng)
		right := randomOperand(minOperand, maxOperand, rng)
		return left, right, left * right
	case "div":
		divisor := randomOperand(max(1, minOperand), max(1, maxOperand), rng)
		quotient := randomOperand(minOperand, maxOperand, rng)
		dividend := divisor * quotient
		return dividend, divisor, quotient
	default:
		panic("unsupported operation")
	}
}

func randomOperand(minOperand, maxOperand int, rng *rand.Rand) int {
	if maxOperand <= minOperand {
		return minOperand
	}
	return minOperand + rng.Intn(maxOperand-minOperand+1)
}

func symbolForOperation(operation string) string {
	switch operation {
	case "add":
		return "+"
	case "sub":
		return "-"
	case "mul":
		return "*"
	case "div":
		return "/"
	default:
		return "?"
	}
}
