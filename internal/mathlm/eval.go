package mathlm

import (
	"fmt"
	"strings"

	"github.com/augahmed/aurelius/internal/arithmetic"
	sharedmodel "github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
)

type EvalReport struct {
	Total             int                  `json:"total"`
	Correct           int                  `json:"correct"`
	Accuracy          float64              `json:"accuracy"`
	MaxTokens         int                  `json:"max_tokens"`
	ByOperation       map[string]EvalGroup `json:"by_operation"`
	ByLevel           map[int]EvalGroup    `json:"by_level"`
	ByTemplate        map[string]EvalGroup `json:"by_template,omitempty"`
	ByAnswerDigits    map[int]EvalGroup    `json:"by_answer_digits,omitempty"`
	BySmallDifference map[string]EvalGroup `json:"by_small_difference,omitempty"`
	Errors            []EvalError          `json:"errors,omitempty"`
}

type EvalGroup struct {
	Total    int     `json:"total"`
	Correct  int     `json:"correct"`
	Accuracy float64 `json:"accuracy"`
}

type EvalOptions struct {
	CollectErrors bool
}

type EvalError struct {
	Prompt          string `json:"prompt"`
	Expected        string `json:"expected"`
	Generated       string `json:"generated"`
	Operation       string `json:"operation"`
	Level           int    `json:"level"`
	Template        string `json:"template"`
	AnswerDigits    int    `json:"answer_digits"`
	SmallDifference bool   `json:"small_difference"`
	RequiresCarry   bool   `json:"requires_carry"`
	RequiresBorrow  bool   `json:"requires_borrow"`
	MinOperand      int    `json:"min_operand"`
	MaxOperand      int    `json:"max_operand"`
	ReasoningStyle  string `json:"reasoning_style,omitempty"`
}

func EvaluateExamples(model sharedmodel.Model, examples []arithmetic.Example, maxTokens int) (EvalReport, error) {
	return EvaluateExamplesWithOptions(model, examples, maxTokens, EvalOptions{})
}

func EvaluateExamplesWithOptions(model sharedmodel.Model, examples []arithmetic.Example, maxTokens int, options EvalOptions) (EvalReport, error) {
	if model == nil {
		return EvalReport{}, fmt.Errorf("model is required")
	}
	if maxTokens <= 0 {
		return EvalReport{}, fmt.Errorf("max tokens must be positive")
	}
	tok := tokenizer.NewByteTokenizer()
	engine, err := runtime.NewEngine(tok, model, sampler.NewGreedySampler())
	if err != nil {
		return EvalReport{}, err
	}

	report := EvalReport{
		Total:       len(examples),
		MaxTokens:   maxTokens,
		ByOperation: make(map[string]EvalGroup),
		ByLevel:     make(map[int]EvalGroup),
	}
	if options.CollectErrors {
		report.ByTemplate = make(map[string]EvalGroup)
		report.ByAnswerDigits = make(map[int]EvalGroup)
		report.BySmallDifference = make(map[string]EvalGroup)
	}
	for _, example := range examples {
		output, err := engine.GenerateWithOptions(example.Prompt, runtime.GenerateOptions{
			MaxTokens:  maxTokens,
			StopTokens: []int{int('\n')},
		})
		if err != nil {
			return EvalReport{}, err
		}
		generated := generatedCompletion(output, example.Prompt)
		expected := arithmetic.FinalAnswer(example)
		generatedAnswer := arithmetic.ExtractFinalAnswer(generated)
		correct := generatedAnswer == expected
		if correct {
			report.Correct++
		}
		addGroupResult(report.ByOperation, example.Operation, correct)
		addGroupResult(report.ByLevel, example.Level, correct)
		if options.CollectErrors {
			addGroupResult(report.ByTemplate, example.Template, correct)
			addGroupResult(report.ByAnswerDigits, example.AnswerDigits, correct)
			addGroupResult(report.BySmallDifference, boolGroupKey(example.SmallDifference), correct)
			if !correct {
				report.Errors = append(report.Errors, EvalError{
					Prompt:          example.Prompt,
					Expected:        expected,
					Generated:       generatedAnswer,
					Operation:       example.Operation,
					Level:           example.Level,
					Template:        example.Template,
					AnswerDigits:    example.AnswerDigits,
					SmallDifference: example.SmallDifference,
					RequiresCarry:   example.RequiresCarry,
					RequiresBorrow:  example.RequiresBorrow,
					MinOperand:      example.MinOperand,
					MaxOperand:      example.MaxOperand,
					ReasoningStyle:  example.ReasoningStyle,
				})
			}
		}
	}
	if report.Total > 0 {
		report.Accuracy = float64(report.Correct) / float64(report.Total)
	}
	return report, nil
}

func generatedCompletion(output, prompt string) string {
	return strings.TrimSpace(strings.TrimPrefix(output, prompt))
}

func boolGroupKey(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func addGroupResult[K comparable](groups map[K]EvalGroup, key K, correct bool) {
	group := groups[key]
	group.Total++
	if correct {
		group.Correct++
	}
	group.Accuracy = float64(group.Correct) / float64(group.Total)
	groups[key] = group
}
