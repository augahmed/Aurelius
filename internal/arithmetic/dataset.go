package arithmetic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

type Example struct {
	Prompt          string `json:"prompt"`
	Completion      string `json:"completion"`
	Operation       string `json:"operation"`
	Level           int    `json:"level"`
	MinOperand      int    `json:"min_operand"`
	MaxOperand      int    `json:"max_operand"`
	AnswerDigits    int    `json:"answer_digits"`
	SmallDifference bool   `json:"small_difference,omitempty"`
	RequiresCarry   bool   `json:"requires_carry,omitempty"`
	RequiresBorrow  bool   `json:"requires_borrow,omitempty"`
	Template        string `json:"template"`
}

type GenerateConfig struct {
	TrainCount          int
	ValCount            int
	MinOperand          int
	MaxOperand          int
	Operations          []string
	Levels              []int
	AnswerDigits        []int
	SmallDifferenceOnly bool
	Seed                int64
}

type MixSource struct {
	Path   string `json:"path"`
	Weight int    `json:"weight"`
}

type MixConfig struct {
	Sources []MixSource
	Seed    int64
}

type Metadata struct {
	TrainCount          int      `json:"train_count"`
	ValCount            int      `json:"val_count"`
	MinOperand          int      `json:"min_operand"`
	MaxOperand          int      `json:"max_operand"`
	Operations          []string `json:"operations"`
	Levels              []int    `json:"levels"`
	AnswerDigits        []int    `json:"answer_digits,omitempty"`
	SmallDifferenceOnly bool     `json:"small_difference_only,omitempty"`
	Seed                int64    `json:"seed"`
	Tokenizer           string   `json:"tokenizer"`
}

type MixMetadata struct {
	TrainCount int         `json:"train_count"`
	ValCount   int         `json:"val_count"`
	Sources    []MixSource `json:"sources"`
	Seed       int64       `json:"seed"`
	Tokenizer  string      `json:"tokenizer"`
}

type SequenceExample struct {
	Context []int
	Target  int
}

type generationTask struct {
	Level     int
	Operation string
}

var supportedOperations = []string{"add", "sub", "mul", "div", "word"}
var supportedLevels = []int{1, 2, 3, 4, 5, 6}

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
	levels := levelsOrDefault(c.Levels, c.Operations)
	for _, operation := range c.Operations {
		if !slices.Contains(supportedOperations, operation) {
			return fmt.Errorf("unsupported operation %q", operation)
		}
	}
	for _, level := range levels {
		if !slices.Contains(supportedLevels, level) {
			return fmt.Errorf("unsupported level %d", level)
		}
		if len(compatibleOperations(level, c.Operations)) == 0 {
			return fmt.Errorf("level %d has no compatible operation in %v", level, c.Operations)
		}
	}
	for _, digits := range c.AnswerDigits {
		if digits <= 0 {
			return fmt.Errorf("answer digits must be positive")
		}
	}
	if c.SmallDifferenceOnly && !slices.Contains(c.Operations, "sub") {
		return fmt.Errorf("small difference filtering requires subtraction examples")
	}
	return nil
}

func GenerateDataset(outputDir string, cfg GenerateConfig) error {
	cfg.Levels = levelsOrDefault(cfg.Levels, cfg.Operations)
	if err := cfg.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	train := generateExamples(cfg, cfg.TrainCount, rng)
	val := generateExamples(cfg, cfg.ValCount, rng)

	if err := writeJSONL(filepath.Join(outputDir, "train.jsonl"), train); err != nil {
		return err
	}
	if err := writeJSONL(filepath.Join(outputDir, "val.jsonl"), val); err != nil {
		return err
	}

	meta := Metadata{
		TrainCount:          cfg.TrainCount,
		ValCount:            cfg.ValCount,
		MinOperand:          cfg.MinOperand,
		MaxOperand:          cfg.MaxOperand,
		Operations:          append([]string(nil), cfg.Operations...),
		Levels:              append([]int(nil), cfg.Levels...),
		AnswerDigits:        append([]int(nil), cfg.AnswerDigits...),
		SmallDifferenceOnly: cfg.SmallDifferenceOnly,
		Seed:                cfg.Seed,
		Tokenizer:           "byte",
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

func MixDatasets(outputDir string, cfg MixConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	train := make([]Example, 0)
	val := make([]Example, 0)
	for _, source := range cfg.Sources {
		sourceTrain, err := LoadExamples(filepath.Join(source.Path, "train.jsonl"))
		if err != nil {
			return fmt.Errorf("load source train %q: %w", source.Path, err)
		}
		sourceVal, err := LoadExamples(filepath.Join(source.Path, "val.jsonl"))
		if err != nil {
			return fmt.Errorf("load source val %q: %w", source.Path, err)
		}
		for i := 0; i < source.Weight; i++ {
			train = append(train, sourceTrain...)
			val = append(val, sourceVal...)
		}
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	rng.Shuffle(len(train), func(i, j int) {
		train[i], train[j] = train[j], train[i]
	})
	rng.Shuffle(len(val), func(i, j int) {
		val[i], val[j] = val[j], val[i]
	})

	if err := writeJSONL(filepath.Join(outputDir, "train.jsonl"), train); err != nil {
		return err
	}
	if err := writeJSONL(filepath.Join(outputDir, "val.jsonl"), val); err != nil {
		return err
	}
	meta := MixMetadata{
		TrainCount: len(train),
		ValCount:   len(val),
		Sources:    append([]MixSource(nil), cfg.Sources...),
		Seed:       cfg.Seed,
		Tokenizer:  "byte",
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal mix meta: %w", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "meta.json"), data, 0o644); err != nil {
		return fmt.Errorf("write mix meta: %w", err)
	}
	return nil
}

func (c MixConfig) Validate() error {
	if len(c.Sources) == 0 {
		return fmt.Errorf("at least one source is required")
	}
	for _, source := range c.Sources {
		if strings.TrimSpace(source.Path) == "" {
			return fmt.Errorf("source path is required")
		}
		if source.Weight <= 0 {
			return fmt.Errorf("source %q weight must be positive", source.Path)
		}
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
		normalizeExampleMetadata(&example)
		examples = append(examples, example)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan dataset %q: %w", path, err)
	}
	return examples, nil
}

func normalizeExampleMetadata(example *Example) {
	answer, err := strconv.Atoi(strings.TrimSpace(example.Completion))
	if err == nil {
		if example.AnswerDigits <= 0 {
			example.AnswerDigits = digitCount(answer)
		}
		if example.Operation == "sub" && answer >= 0 && answer <= 9 {
			example.SmallDifference = true
		}
	}
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
		promptTokens, err := tok.Encode(example.Prompt)
		if err != nil {
			return nil, fmt.Errorf("encode training prompt: %w", err)
		}
		targetTokens, err := tok.Encode(example.Completion + "\n")
		if err != nil {
			return nil, fmt.Errorf("encode training completion: %w", err)
		}
		prefix := append([]int(nil), promptTokens...)
		for _, target := range targetTokens {
			context := make([]int, contextSize)
			start := max(0, len(prefix)-contextSize)
			window := prefix[start:]
			copy(context[contextSize-len(window):], window)
			sequences = append(sequences, SequenceExample{
				Context: context,
				Target:  target,
			})
			prefix = append(prefix, target)
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

func generateExamples(cfg GenerateConfig, count int, rng *rand.Rand) []Example {
	tasks := buildGenerationTasks(cfg)
	examples := make([]Example, count)
	for i := range examples {
		task := tasks[i%len(tasks)]
		examples[i] = generateExampleForTask(cfg, task, rng)
	}
	rng.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})
	return examples
}

func buildGenerationTasks(cfg GenerateConfig) []generationTask {
	tasks := make([]generationTask, 0)
	for _, level := range cfg.Levels {
		for _, operation := range compatibleOperations(level, cfg.Operations) {
			tasks = append(tasks, generationTask{
				Level:     level,
				Operation: operation,
			})
		}
	}
	return tasks
}

func generateExampleForTask(cfg GenerateConfig, task generationTask, rng *rand.Rand) Example {
	if task.Level == 6 {
		return generateWordExample(randomOperand(1, 20, rng), randomOperand(1, 20, rng), rng)
	}
	const maxAttempts = 10000
	for attempt := 0; attempt < maxAttempts; attempt++ {
		example := generateCandidateForTask(cfg, task, rng)
		if matchesAnswerFilters(example, cfg) {
			return example
		}
	}
	panic("could not generate arithmetic example matching filters")
}

func generateCandidateForTask(cfg GenerateConfig, task generationTask, rng *rand.Rand) Example {
	left, right, answer, requiresCarry, requiresBorrow := operandsForLevel(task.Operation, task.Level, cfg.MinOperand, cfg.MaxOperand, rng)
	template := rng.Intn(3)
	example := Example{
		Completion:      fmt.Sprintf("%d", answer),
		Operation:       task.Operation,
		Level:           task.Level,
		MinOperand:      minOperandForLevel(task.Level, cfg.MinOperand),
		MaxOperand:      maxOperandForLevel(task.Level, cfg.MaxOperand),
		AnswerDigits:    digitCount(answer),
		SmallDifference: task.Operation == "sub" && answer >= 0 && answer <= 9,
		RequiresCarry:   requiresCarry,
		RequiresBorrow:  requiresBorrow,
	}
	switch template {
	case 0:
		example.Prompt = fmt.Sprintf("%d %s %d = ", left, symbolForOperation(task.Operation), right)
		example.Template = "equation"
	case 1:
		example.Prompt = fmt.Sprintf("What is %d %s %d? ", left, symbolForOperation(task.Operation), right)
		example.Template = "question"
	default:
		example.Prompt = fmt.Sprintf("Solve: %d %s %d = ", left, symbolForOperation(task.Operation), right)
		example.Template = "solve"
	}
	return example
}

func matchesAnswerFilters(example Example, cfg GenerateConfig) bool {
	if len(cfg.AnswerDigits) > 0 && !slices.Contains(cfg.AnswerDigits, example.AnswerDigits) {
		return false
	}
	if cfg.SmallDifferenceOnly && !example.SmallDifference {
		return false
	}
	return true
}

func operandsForLevel(operation string, level, fallbackMin, fallbackMax int, rng *rand.Rand) (int, int, int, bool, bool) {
	switch level {
	case 1:
		return operandsForOperation(operation, 0, 9, rng, nil, nil)
	case 2:
		return operandsForOperation(operation, 10, 99, rng, boolPtr(false), boolPtr(false))
	case 3:
		return operandsForOperation(operation, 10, 99, rng, boolPtr(operation == "add"), boolPtr(operation == "sub"))
	case 4:
		return operandsForOperation(operation, 0, 12, rng, nil, nil)
	case 5:
		return operandsForOperation(operation, 1, 12, rng, nil, nil)
	default:
		return operandsForOperation(operation, fallbackMin, fallbackMax, rng, nil, nil)
	}
}

func operandsForOperation(operation string, minOperand, maxOperand int, rng *rand.Rand, requireCarry, requireBorrow *bool) (int, int, int, bool, bool) {
	switch operation {
	case "add":
		for {
			left := randomOperand(minOperand, maxOperand, rng)
			right := randomOperand(minOperand, maxOperand, rng)
			carry := (left%10 + right%10) >= 10
			if requireCarry == nil || carry == *requireCarry {
				return left, right, left + right, carry, false
			}
		}
	case "sub":
		for {
			left := randomOperand(minOperand, maxOperand, rng)
			right := randomOperand(minOperand, maxOperand, rng)
			if right > left {
				left, right = right, left
			}
			borrow := (left % 10) < (right % 10)
			if requireBorrow == nil || borrow == *requireBorrow {
				return left, right, left - right, false, borrow
			}
		}
	case "mul":
		left := randomOperand(minOperand, maxOperand, rng)
		right := randomOperand(minOperand, maxOperand, rng)
		return left, right, left * right, false, false
	case "div":
		divisor := randomOperand(max(1, minOperand), max(1, maxOperand), rng)
		quotient := randomOperand(minOperand, maxOperand, rng)
		dividend := divisor * quotient
		return dividend, divisor, quotient, false, false
	default:
		panic("unsupported operation")
	}
}

func boolPtr(value bool) *bool {
	return &value
}

func generateWordExample(left, right int, rng *rand.Rand) Example {
	extra := randomOperand(1, 9, rng)
	answer := left + right - extra
	if answer < 0 {
		answer = left + right + extra
		return Example{
			Prompt:       fmt.Sprintf("Mia has %d marbles, gets %d more, then gets %d more. How many marbles? ", left, right, extra),
			Completion:   fmt.Sprintf("%d", answer),
			Operation:    "word",
			Level:        6,
			MinOperand:   1,
			MaxOperand:   20,
			AnswerDigits: digitCount(answer),
			Template:     "two_step_word",
		}
	}
	return Example{
		Prompt:       fmt.Sprintf("Mia has %d marbles, gets %d more, then gives away %d. How many marbles? ", left, right, extra),
		Completion:   fmt.Sprintf("%d", answer),
		Operation:    "word",
		Level:        6,
		MinOperand:   1,
		MaxOperand:   20,
		AnswerDigits: digitCount(answer),
		Template:     "two_step_word",
	}
}

func compatibleOperations(level int, requested []string) []string {
	allowed := map[int][]string{
		1: []string{"add", "sub"},
		2: []string{"add", "sub"},
		3: []string{"add", "sub"},
		4: []string{"mul"},
		5: []string{"div"},
		6: []string{"word"},
	}
	out := make([]string, 0)
	for _, operation := range requested {
		if slices.Contains(allowed[level], operation) {
			out = append(out, operation)
		}
	}
	return out
}

func levelsOrDefault(levels []int, operations []string) []int {
	if len(levels) == 0 {
		defaults := make([]int, 0, 5)
		for _, level := range []int{1, 2, 3, 4, 5, 6} {
			if len(compatibleOperations(level, operations)) > 0 {
				defaults = append(defaults, level)
			}
		}
		return defaults
	}
	return append([]int(nil), levels...)
}

func minOperandForLevel(level, fallback int) int {
	switch level {
	case 1, 4:
		return 0
	case 2, 3:
		return 10
	case 5, 6:
		return 1
	default:
		return fallback
	}
}

func maxOperandForLevel(level, fallback int) int {
	switch level {
	case 1:
		return 9
	case 2, 3:
		return 99
	case 4, 5:
		return 12
	case 6:
		return 20
	default:
		return fallback
	}
}

func randomOperand(minOperand, maxOperand int, rng *rand.Rand) int {
	if maxOperand <= minOperand {
		return minOperand
	}
	return minOperand + rng.Intn(maxOperand-minOperand+1)
}

func digitCount(value int) int {
	if value < 0 {
		value = -value
	}
	if value == 0 {
		return 1
	}
	digits := 0
	for value > 0 {
		digits++
		value /= 10
	}
	return digits
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
