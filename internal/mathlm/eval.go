package mathlm

import (
	"fmt"
	"strings"

	"github.com/augahmed/aurelius/internal/arithmetic"
	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
)

type EvalReport struct {
	Total     int     `json:"total"`
	Correct   int     `json:"correct"`
	Accuracy  float64 `json:"accuracy"`
	MaxTokens int     `json:"max_tokens"`
}

func EvaluateExamples(model *Model, examples []arithmetic.Example, maxTokens int) (EvalReport, error) {
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

	report := EvalReport{Total: len(examples), MaxTokens: maxTokens}
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
		if generated == expected {
			report.Correct++
		}
	}
	if report.Total > 0 {
		report.Accuracy = float64(report.Correct) / float64(report.Total)
	}
	return report, nil
}
