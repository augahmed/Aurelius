package runtime

import (
	"fmt"

	"github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
)

type Engine struct {
	tokenizer tokenizer.Tokenizer
	model     model.Model
	sampler   sampler.Sampler
	cache     model.Cache
}

func NewEngine(tok tokenizer.Tokenizer, mdl model.Model, samp sampler.Sampler) (*Engine, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	if mdl == nil {
		return nil, fmt.Errorf("model is required")
	}
	if samp == nil {
		return nil, fmt.Errorf("sampler is required")
	}
	return &Engine{
		tokenizer: tok,
		model:     mdl,
		sampler:   samp,
		cache:     model.NoopCache{},
	}, nil
}

func (e *Engine) Generate(prompt string, maxTokens int) (string, error) {
	if prompt == "" {
		return "", fmt.Errorf("prompt cannot be empty")
	}
	if maxTokens < 0 {
		return "", fmt.Errorf("maxTokens must be non-negative")
	}
	tokens, err := e.tokenizer.Encode(prompt)
	if err != nil {
		return "", err
	}
	if len(tokens) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}
	for i := 0; i < maxTokens; i++ {
		logits, err := e.model.Forward(tokens, e.cache)
		if err != nil {
			return "", err
		}
		next, err := e.sampler.Sample(logits)
		if err != nil {
			return "", err
		}
		tokens = append(tokens, next)
	}
	return e.tokenizer.Decode(tokens)
}
