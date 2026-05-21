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
	Total       int                  `json:"total"`
	Correct     int                  `json:"correct"`
	Accuracy    float64              `json:"accuracy"`
	MaxTokens   int                  `json:"max_tokens"`
	ByOperation map[string]EvalGroup `json:"by_operation"`
	ByLevel     map[int]EvalGroup    `json:"by_level"`
}

type EvalGroup struct {
	Total    int     `json:"total"`
	Correct  int     `json:"correct"`
	Accuracy float64 `json:"accuracy"`
}

func EvaluateExamples(model sharedmodel.Model, examples []arithmetic.Example, maxTokens int) (EvalReport, error) {
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
	for _, example := range examples {
		output, err := engine.GenerateWithOptions(example.Prompt, runtime.GenerateOptions{
			MaxTokens:  maxTokens,
			StopTokens: []int{int('\n')},
		})
		if err != nil {
			return EvalReport{}, err
		}
		generated := strings.TrimSpace(strings.TrimPrefix(output, example.Prompt))
		expected := strings.TrimSpace(example.Completion)
		correct := generated == expected
		if correct {
			report.Correct++
		}
		addGroupResult(report.ByOperation, example.Operation, correct)
		addGroupResult(report.ByLevel, example.Level, correct)
	}
	if report.Total > 0 {
		report.Accuracy = float64(report.Correct) / float64(report.Total)
	}
	return report, nil
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
