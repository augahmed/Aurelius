package runtime

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
)

type Engine struct {
	tokenizer tokenizer.Tokenizer
	model     model.Model
	sampler   sampler.Sampler
}

type GenerateOptions struct {
	MaxTokens   int
	UseCache    bool
	StopTokens  []int
	Temperature float64
	TopK        int
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
	}, nil
}

func (e *Engine) Generate(prompt string, maxTokens int) (string, error) {
	return e.GenerateWithOptions(prompt, GenerateOptions{
		MaxTokens: maxTokens,
	})
}

func (e *Engine) GenerateWithOptions(prompt string, options GenerateOptions) (string, error) {
	if prompt == "" {
		return "", fmt.Errorf("prompt cannot be empty")
	}
	if options.MaxTokens < 0 {
		return "", fmt.Errorf("max tokens must be non-negative")
	}
	tokens, err := e.tokenizer.Encode(prompt)
	if err != nil {
		return "", err
	}
	if len(tokens) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}

	if options.UseCache {
		if cacheModel, ok := e.model.(model.CacheCapableModel); ok {
			return e.generateWithCache(tokens, options, cacheModel)
		}
	}
	return e.generateWithoutCache(tokens, options)
}

func (e *Engine) generateWithoutCache(tokens []int, options GenerateOptions) (string, error) {
	cache := model.NoopCache{}
	nextTokenSampler, err := e.samplerForOptions(options)
	if err != nil {
		return "", err
	}
	for i := 0; i < options.MaxTokens; i++ {
		logits, err := e.model.Forward(tokens, cache)
		if err != nil {
			return "", err
		}
		next, err := nextTokenSampler.Sample(logits)
		if err != nil {
			return "", err
		}
		tokens = append(tokens, next)
		if shouldStop(next, options.StopTokens) {
			break
		}
	}
	return e.tokenizer.Decode(tokens)
}

func (e *Engine) generateWithCache(tokens []int, options GenerateOptions, cacheModel model.CacheCapableModel) (string, error) {
	if options.MaxTokens == 0 {
		return e.tokenizer.Decode(tokens)
	}
	nextTokenSampler, err := e.samplerForOptions(options)
	if err != nil {
		return "", err
	}

	cache := cacheModel.NewCache()
	logits, err := cacheModel.Forward(tokens, cache)
	if err != nil {
		return "", err
	}

	for i := 0; i < options.MaxTokens; i++ {
		next, err := nextTokenSampler.Sample(logits)
		if err != nil {
			return "", err
		}
		tokens = append(tokens, next)
		if shouldStop(next, options.StopTokens) {
			break
		}
		if i == options.MaxTokens-1 {
			break
		}
		logits, err = cacheModel.Forward([]int{next}, cache)
		if err != nil {
			return "", err
		}
	}
	return e.tokenizer.Decode(tokens)
}

func (e *Engine) samplerForOptions(options GenerateOptions) (sampler.Sampler, error) {
	if options.TopK < 0 {
		return nil, fmt.Errorf("top-k must be non-negative")
	}
	if options.Temperature < 0 {
		return nil, fmt.Errorf("temperature must be non-negative")
	}
	if options.TopK > 0 {
		if options.TopK == 1 {
			return sampler.NewGreedySampler(), nil
		}
		temperature := options.Temperature
		if temperature == 0 {
			temperature = 1.0
		}
		return sampler.NewTopKSampler(options.TopK, temperature, rand.New(rand.NewSource(time.Now().UnixNano()))), nil
	}
	if options.Temperature == 0 {
		return e.sampler, nil
	}
	return sampler.NewTemperatureSampler(options.Temperature, rand.New(rand.NewSource(time.Now().UnixNano()))), nil
}

func shouldStop(token int, stopTokens []int) bool {
	for _, stopToken := range stopTokens {
		if token == stopToken {
			return true
		}
	}
	return false
}
