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
	Answer          string `json:"answer,omitempty"`
	Operation       string `json:"operation"`
	Level           int    `json:"level"`
	MinOperand      int    `json:"min_operand"`
	MaxOperand      int    `json:"max_operand"`
	AnswerDigits    int    `json:"answer_digits"`
	SmallDifference bool   `json:"small_difference,omitempty"`
	RequiresCarry   bool   `json:"requires_carry,omitempty"`
	RequiresBorrow  bool   `json:"requires_borrow,omitempty"`
	Template        string `json:"template"`
	ReasoningStyle  string `json:"reasoning_style,omitempty"`
}

type GenerateConfig struct {
	TrainCount          int
	ValCount            int
	MinOperand          int
	MaxOperand          int
	Operations          []string
	Levels              []int
	Templates           []string
	ReasoningStyle      string
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
	Templates           []string `json:"templates,omitempty"`
	ReasoningStyle      string   `json:"reasoning_style"`
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
	Template  string
}

var supportedOperations = []string{"add", "sub", "mul", "div", "word", "derivative"}
var supportedLevels = []int{1, 2, 3, 4, 5, 6, 7}
var supportedTemplates = []string{"equation", "question", "solve"}
var supportedReasoningStyles = []string{"direct", "worked", "compact", "coefficients"}

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
	templates := templatesOrDefault(c.Templates)
	for _, template := range templates {
		if !slices.Contains(supportedTemplates, template) {
			return fmt.Errorf("unsupported template %q", template)
		}
	}
	reasoningStyle := reasoningStyleOrDefault(c.ReasoningStyle)
	if !slices.Contains(supportedReasoningStyles, reasoningStyle) {
		return fmt.Errorf("unsupported reasoning style %q", c.ReasoningStyle)
	}
	taskCfg := c
	taskCfg.Levels = levels
	taskCfg.Templates = templates
	if len(buildGenerationTasks(taskCfg)) == 0 {
		return fmt.Errorf("no compatible generation tasks for levels %v, operations %v, templates %v", levels, c.Operations, templates)
	}
	if c.SmallDifferenceOnly && !slices.Contains(c.Operations, "sub") {
		return fmt.Errorf("small difference filtering requires subtraction examples")
	}
	return nil
}

func GenerateDataset(outputDir string, cfg GenerateConfig) error {
	cfg.Levels = levelsOrDefault(cfg.Levels, cfg.Operations)
	cfg.Templates = templatesOrDefault(cfg.Templates)
	cfg.ReasoningStyle = reasoningStyleOrDefault(cfg.ReasoningStyle)
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
		Templates:           append([]string(nil), cfg.Templates...),
		ReasoningStyle:      cfg.ReasoningStyle,
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
	if example.ReasoningStyle == "" {
		example.ReasoningStyle = "direct"
	}
	if strings.TrimSpace(example.Answer) == "" {
		example.Answer = FinalAnswer(*example)
	}
	answer, err := strconv.Atoi(strings.TrimSpace(example.Answer))
	if err == nil {
		if example.AnswerDigits <= 0 {
			example.AnswerDigits = digitCount(answer)
		}
		if example.Operation == "sub" && answer >= 0 && answer <= 9 {
			example.SmallDifference = true
		}
	}
}

func FinalAnswer(example Example) string {
	if strings.TrimSpace(example.Answer) != "" {
		return strings.TrimSpace(example.Answer)
	}
	return ExtractFinalAnswer(example.Completion)
}

func ExtractFinalAnswer(text string) string {
	trimmed := strings.TrimSpace(text)
	lower := strings.ToLower(trimmed)
	marker, index := lastAnswerMarker(lower)
	if index >= 0 {
		return firstIntegerOrTrimmed(trimmed[index+len(marker):])
	}
	return trimmed
}

func lastAnswerMarker(lower string) (string, int) {
	bestMarker := ""
	bestIndex := -1
	for _, marker := range []string{"answer:", "ans:"} {
		index := strings.LastIndex(lower, marker)
		if index > bestIndex {
			bestMarker = marker
			bestIndex = index
		}
	}
	return bestMarker, bestIndex
}

func firstIntegerOrTrimmed(text string) string {
	text = strings.TrimSpace(text)
	start := -1
	for i, r := range text {
		if r == '-' || (r >= '0' && r <= '9') {
			start = i
			break
		}
	}
	if start < 0 {
		return text
	}
	end := start
	for end < len(text) {
		ch := text[end]
		if end == start && ch == '-' {
			end++
			continue
		}
		if ch < '0' || ch > '9' {
			break
		}
		end++
	}
	return text[start:end]
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
			for _, template := range compatibleTemplates(operation, cfg.Templates) {
				tasks = append(tasks, generationTask{
					Level:     level,
					Operation: operation,
					Template:  template,
				})
			}
		}
	}
	return tasks
}

func generateExampleForTask(cfg GenerateConfig, task generationTask, rng *rand.Rand) Example {
	if task.Level == 6 {
		return generateWordExample(randomOperand(1, 20, rng), randomOperand(1, 20, rng), cfg.ReasoningStyle, rng)
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
	if task.Level == 7 {
		return generateDerivativeExample(task.Template, cfg.ReasoningStyle, rng)
	}
	left, right, answer, requiresCarry, requiresBorrow := operandsForLevel(task.Operation, task.Level, cfg.MinOperand, cfg.MaxOperand, rng)
	answerText := fmt.Sprintf("%d", answer)
	example := Example{
		Completion:      completionForReasoningStyle(task.Operation, left, right, answer, cfg.ReasoningStyle),
		Answer:          answerText,
		Operation:       task.Operation,
		Level:           task.Level,
		MinOperand:      minOperandForLevel(task.Level, cfg.MinOperand),
		MaxOperand:      maxOperandForLevel(task.Level, cfg.MaxOperand),
		AnswerDigits:    digitCount(answer),
		SmallDifference: task.Operation == "sub" && answer >= 0 && answer <= 9,
		RequiresCarry:   requiresCarry,
		RequiresBorrow:  requiresBorrow,
		ReasoningStyle:  cfg.ReasoningStyle,
	}
	switch task.Template {
	case "equation":
		example.Prompt = fmt.Sprintf("%d %s %d = ", left, symbolForOperation(task.Operation), right)
		example.Template = "equation"
	case "question":
		example.Prompt = fmt.Sprintf("What is %d %s %d? ", left, symbolForOperation(task.Operation), right)
		example.Template = "question"
	case "solve":
		example.Prompt = fmt.Sprintf("Solve: %d %s %d = ", left, symbolForOperation(task.Operation), right)
		example.Template = "solve"
	default:
		panic("unsupported template")
	}
	return example
}

func completionForReasoningStyle(operation string, left, right, answer int, reasoningStyle string) string {
	switch reasoningStyleOrDefault(reasoningStyle) {
	case "direct", "coefficients":
		return fmt.Sprintf("%d", answer)
	case "compact":
		return compactCompletion(operation, left, right, answer)
	}
	switch operation {
	case "add":
		return workedAddition(left, right, answer)
	case "sub":
		return workedSubtraction(left, right, answer)
	case "mul":
		return fmt.Sprintf("multiply: %d*%d=%d; answer: %d", left, right, answer, answer)
	case "div":
		return fmt.Sprintf("check: %d*%d=%d; answer: %d", right, answer, left, answer)
	default:
		return fmt.Sprintf("answer: %d", answer)
	}
}

func compactCompletion(operation string, left, right, answer int) string {
	switch operation {
	case "add":
		return compactAddition(left, right, answer)
	case "sub":
		return compactSubtraction(left, right, answer)
	case "mul":
		return fmt.Sprintf("mul:%d*%d=%d; ans:%d", left, right, answer, answer)
	case "div":
		return fmt.Sprintf("chk:%d*%d=%d; ans:%d", right, answer, left, answer)
	default:
		return fmt.Sprintf("ans:%d", answer)
	}
}

func compactAddition(left, right, answer int) string {
	if left < 10 && right < 10 {
		return fmt.Sprintf("o:%d+%d=%d; ans:%d", left, right, answer, answer)
	}
	leftOnes, rightOnes := left%10, right%10
	leftTens, rightTens := left/10, right/10
	onesSum := leftOnes + rightOnes
	onesDigit := onesSum % 10
	carry := onesSum / 10
	tensSum := leftTens + rightTens + carry
	if carry > 0 {
		return fmt.Sprintf("o:%d+%d=%d w%d c%d; t:%d+%d+%d=%d; ans:%d",
			leftOnes, rightOnes, onesSum, onesDigit, carry, leftTens, rightTens, carry, tensSum, answer)
	}
	return fmt.Sprintf("o:%d+%d=%d; t:%d+%d=%d; ans:%d",
		leftOnes, rightOnes, onesSum, leftTens, rightTens, tensSum, answer)
}

func compactSubtraction(left, right, answer int) string {
	if left < 10 && right < 10 {
		return fmt.Sprintf("o:%d-%d=%d; ans:%d", left, right, answer, answer)
	}
	leftOnes, rightOnes := left%10, right%10
	leftTens, rightTens := left/10, right/10
	if leftOnes < rightOnes {
		onesDigit := leftOnes + 10 - rightOnes
		tensDigit := leftTens - 1 - rightTens
		return fmt.Sprintf("o:%d-%d=%d b1; t:%d-%d=%d; ans:%d",
			leftOnes+10, rightOnes, onesDigit, leftTens-1, rightTens, tensDigit, answer)
	}
	onesDigit := leftOnes - rightOnes
	tensDigit := leftTens - rightTens
	return fmt.Sprintf("o:%d-%d=%d; t:%d-%d=%d; ans:%d",
		leftOnes, rightOnes, onesDigit, leftTens, rightTens, tensDigit, answer)
}

func workedAddition(left, right, answer int) string {
	if left < 10 && right < 10 {
		return fmt.Sprintf("ones: %d+%d=%d; answer: %d", left, right, answer, answer)
	}
	leftOnes, rightOnes := left%10, right%10
	leftTens, rightTens := left/10, right/10
	onesSum := leftOnes + rightOnes
	onesDigit := onesSum % 10
	carry := onesSum / 10
	tensSum := leftTens + rightTens + carry
	if carry > 0 {
		return fmt.Sprintf("ones: %d+%d=%d so write %d carry %d; tens: %d+%d+%d=%d; answer: %d",
			leftOnes, rightOnes, onesSum, onesDigit, carry, leftTens, rightTens, carry, tensSum, answer)
	}
	return fmt.Sprintf("ones: %d+%d=%d; tens: %d+%d=%d; answer: %d",
		leftOnes, rightOnes, onesSum, leftTens, rightTens, tensSum, answer)
}

func workedSubtraction(left, right, answer int) string {
	if left < 10 && right < 10 {
		return fmt.Sprintf("ones: %d-%d=%d; answer: %d", left, right, answer, answer)
	}
	leftOnes, rightOnes := left%10, right%10
	leftTens, rightTens := left/10, right/10
	if leftOnes < rightOnes {
		onesDigit := leftOnes + 10 - rightOnes
		tensDigit := leftTens - 1 - rightTens
		return fmt.Sprintf("ones: borrow 1 ten; %d-%d=%d; tens: %d-%d=%d; answer: %d",
			leftOnes+10, rightOnes, onesDigit, leftTens-1, rightTens, tensDigit, answer)
	}
	onesDigit := leftOnes - rightOnes
	tensDigit := leftTens - rightTens
	return fmt.Sprintf("ones: %d-%d=%d; tens: %d-%d=%d; answer: %d",
		leftOnes, rightOnes, onesDigit, leftTens, rightTens, tensDigit, answer)
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

func generateWordExample(left, right int, reasoningStyle string, rng *rand.Rand) Example {
	extra := randomOperand(1, 9, rng)
	answer := left + right - extra
	answerText := fmt.Sprintf("%d", answer)
	if answer < 0 {
		answer = left + right + extra
		answerText = fmt.Sprintf("%d", answer)
		return Example{
			Prompt:         fmt.Sprintf("Mia has %d marbles, gets %d more, then gets %d more. How many marbles? ", left, right, extra),
			Completion:     wordCompletion(left, right, extra, answer, "add_add", reasoningStyle),
			Answer:         answerText,
			Operation:      "word",
			Level:          6,
			MinOperand:     1,
			MaxOperand:     20,
			AnswerDigits:   digitCount(answer),
			Template:       "two_step_word",
			ReasoningStyle: reasoningStyleOrDefault(reasoningStyle),
		}
	}
	return Example{
		Prompt:         fmt.Sprintf("Mia has %d marbles, gets %d more, then gives away %d. How many marbles? ", left, right, extra),
		Completion:     wordCompletion(left, right, extra, answer, "add_sub", reasoningStyle),
		Answer:         answerText,
		Operation:      "word",
		Level:          6,
		MinOperand:     1,
		MaxOperand:     20,
		AnswerDigits:   digitCount(answer),
		Template:       "two_step_word",
		ReasoningStyle: reasoningStyleOrDefault(reasoningStyle),
	}
}

func wordCompletion(left, right, extra, answer int, pattern, reasoningStyle string) string {
	switch reasoningStyleOrDefault(reasoningStyle) {
	case "direct", "coefficients":
		return fmt.Sprintf("%d", answer)
	case "compact":
		sum := left + right
		if pattern == "add_add" {
			return fmt.Sprintf("s:%d+%d=%d; n:%d+%d=%d; ans:%d", left, right, sum, sum, extra, answer, answer)
		}
		return fmt.Sprintf("s:%d+%d=%d; n:%d-%d=%d; ans:%d", left, right, sum, sum, extra, answer, answer)
	}
	sum := left + right
	if pattern == "add_add" {
		return fmt.Sprintf("first: %d+%d=%d; then: %d+%d=%d; answer: %d", left, right, sum, sum, extra, answer, answer)
	}
	return fmt.Sprintf("first: %d+%d=%d; then: %d-%d=%d; answer: %d", left, right, sum, sum, extra, answer, answer)
}

func generateDerivativeExample(template, reasoningStyle string, rng *rand.Rand) Example {
	expression, polynomialAnswer, coefficientAnswer := derivativeExpression(rng)
	answer := derivativeAnswerForReasoningStyle(polynomialAnswer, coefficientAnswer, reasoningStyle)
	prompt := fmt.Sprintf("Derrivative: %s ", expression)
	if template == "question" {
		prompt = fmt.Sprintf("What is the derrivative of %s? ", expression)
	}
	return Example{
		Prompt:         prompt,
		Completion:     derivativeCompletion(expression, answer, reasoningStyle),
		Answer:         answer,
		Operation:      "derivative",
		Level:          7,
		MinOperand:     1,
		MaxOperand:     9,
		AnswerDigits:   digitCharacters(answer),
		Template:       template,
		ReasoningStyle: reasoningStyleOrDefault(reasoningStyle),
	}
}

func derivativeExpression(rng *rand.Rand) (string, string, string) {
	degree := randomOperand(1, 3, rng)
	coefficients := make([]int, degree+1)
	for power := 0; power <= degree; power++ {
		coefficients[power] = randomOperand(1, 9, rng)
	}
	expressionTerms := make([]string, 0, degree+1)
	derivativeTerms := make([]string, 0, degree)
	derivativeCoefficients := make([]string, 0, degree)
	for power := degree; power >= 0; power-- {
		expressionTerms = append(expressionTerms, formatPolynomialTerm(coefficients[power], power))
		if power > 0 {
			derivativeCoefficient := coefficients[power] * power
			derivativeTerms = append(derivativeTerms, formatPolynomialTerm(derivativeCoefficient, power-1))
			derivativeCoefficients = append(derivativeCoefficients, fmt.Sprintf("%d", derivativeCoefficient))
		}
	}
	return strings.Join(expressionTerms, " + "), strings.Join(derivativeTerms, " + "), strings.Join(derivativeCoefficients, ",")
}

func derivativeAnswerForReasoningStyle(polynomialAnswer, coefficientAnswer, reasoningStyle string) string {
	if reasoningStyleOrDefault(reasoningStyle) == "coefficients" {
		return coefficientAnswer
	}
	return polynomialAnswer
}

func derivativeCompletion(expression, answer, reasoningStyle string) string {
	switch reasoningStyleOrDefault(reasoningStyle) {
	case "direct", "coefficients":
		return answer
	case "compact":
		return fmt.Sprintf("d:%s; ans:%s", expression, answer)
	default:
		return fmt.Sprintf("derivative: d/dx %s = %s; answer: %s", expression, answer, answer)
	}
}

func formatPolynomialTerm(coefficient, power int) string {
	switch power {
	case 0:
		return fmt.Sprintf("%d", coefficient)
	case 1:
		if coefficient == 1 {
			return "x"
		}
		return fmt.Sprintf("%dx", coefficient)
	default:
		if coefficient == 1 {
			return fmt.Sprintf("x^%d", power)
		}
		return fmt.Sprintf("%dx^%d", coefficient, power)
	}
}

func digitCharacters(value string) int {
	count := 0
	for _, char := range value {
		if char >= '0' && char <= '9' {
			count++
		}
	}
	return max(1, count)
}

func compatibleOperations(level int, requested []string) []string {
	allowed := map[int][]string{
		1: []string{"add", "sub"},
		2: []string{"add", "sub"},
		3: []string{"add", "sub"},
		4: []string{"mul"},
		5: []string{"div"},
		6: []string{"word"},
		7: []string{"derivative"},
	}
	out := make([]string, 0)
	for _, operation := range requested {
		if slices.Contains(allowed[level], operation) {
			out = append(out, operation)
		}
	}
	return out
}

func compatibleTemplates(operation string, requested []string) []string {
	if operation != "derivative" {
		return append([]string(nil), requested...)
	}
	out := make([]string, 0, 2)
	for _, template := range requested {
		if template == "equation" || template == "question" {
			out = append(out, template)
		}
	}
	return out
}

func levelsOrDefault(levels []int, operations []string) []int {
	if len(levels) == 0 {
		defaults := make([]int, 0, 7)
		for _, level := range []int{1, 2, 3, 4, 5, 6, 7} {
			if len(compatibleOperations(level, operations)) > 0 {
				defaults = append(defaults, level)
			}
		}
		return defaults
	}
	return append([]int(nil), levels...)
}

func templatesOrDefault(templates []string) []string {
	if len(templates) == 0 {
		return append([]string(nil), supportedTemplates...)
	}
	return append([]string(nil), templates...)
}

func reasoningStyleOrDefault(reasoningStyle string) string {
	reasoningStyle = strings.TrimSpace(reasoningStyle)
	if reasoningStyle == "" {
		return "direct"
	}
	return reasoningStyle
}

func minOperandForLevel(level, fallback int) int {
	switch level {
	case 1, 4:
		return 0
	case 2, 3:
		return 10
	case 5, 6:
		return 1
	case 7:
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
	case 7:
		return 9
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
	case "derivative":
		return "d/dx"
	default:
		return "?"
	}
}
