package runtime

import (
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
	"github.com/augahmed/aurelius/internal/transformer"
)

func TestEngineGenerate(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	got, err := engine.Generate("hi", 4)
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if !strings.HasPrefix(got, "hi") {
		t.Fatalf("generated text = %q, want prefix %q", got, "hi")
	}
	if len(got) != len("hi")+4 {
		t.Fatalf("generated text length = %d, want %d", len(got), len("hi")+4)
	}
}

func TestEngineRejectsEmptyPrompt(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	if _, err := engine.Generate("", 1); err == nil {
		t.Fatal("expected empty prompt error")
	}
}

func TestEngineGenerateWithOptionsUseCacheFalse(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	got, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens: 4,
		UseCache:  false,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if !strings.HasPrefix(got, "hi") {
		t.Fatalf("generated text = %q, want prefix %q", got, "hi")
	}
	if len(got) != len("hi")+4 {
		t.Fatalf("generated text length = %d, want %d", len(got), len("hi")+4)
	}
}

func TestEngineGenerateWithOptionsUseCacheTrue(t *testing.T) {
	baseModel, err := newTinyModel()
	if err != nil {
		t.Fatalf("newTinyModel error: %v", err)
	}
	traceModel := &tracingCacheModel{underlying: baseModel}
	engine, err := NewEngine(tokenizer.NewByteTokenizer(), traceModel, sampler.NewGreedySampler())
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}

	got, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens: 2,
		UseCache:  true,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if !strings.HasPrefix(got, "hi") {
		t.Fatalf("generated text = %q, want prefix %q", got, "hi")
	}
	if len(traceModel.callLengths) != 2 {
		t.Fatalf("call count = %d, want 2", len(traceModel.callLengths))
	}
	if traceModel.callLengths[0] != 2 || traceModel.callLengths[1] != 1 {
		t.Fatalf("call lengths = %v, want [2 1]", traceModel.callLengths)
	}
	if !traceModel.cacheUsed {
		t.Fatal("expected cache-backed generation path to be used")
	}
}

func TestEngineCachedGenerationMatchesUncachedGeneration(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}

	uncached, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens: 4,
		UseCache:  false,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions uncached error: %v", err)
	}
	cached, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens: 4,
		UseCache:  true,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions cached error: %v", err)
	}
	if cached != uncached {
		t.Fatalf("cached output = %q, want %q", cached, uncached)
	}
}

func TestEngineFallsBackWhenModelDoesNotSupportCaching(t *testing.T) {
	baseModel, err := newTinyModel()
	if err != nil {
		t.Fatalf("newTinyModel error: %v", err)
	}
	traceModel := &tracingModel{underlying: baseModel}
	engine, err := NewEngine(tokenizer.NewByteTokenizer(), traceModel, sampler.NewGreedySampler())
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}

	got, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens: 2,
		UseCache:  true,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if !strings.HasPrefix(got, "hi") {
		t.Fatalf("generated text = %q, want prefix %q", got, "hi")
	}
	if len(traceModel.callLengths) != 2 {
		t.Fatalf("call count = %d, want 2", len(traceModel.callLengths))
	}
	if traceModel.callLengths[0] != 2 || traceModel.callLengths[1] != 3 {
		t.Fatalf("call lengths = %v, want [2 3]", traceModel.callLengths)
	}
}

func TestEngineRejectsInvalidMaxTokens(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	if _, err := engine.GenerateWithOptions("hi", GenerateOptions{MaxTokens: -1}); err == nil {
		t.Fatal("expected invalid max tokens error")
	}
}

func TestEngineRejectsInvalidTemperature(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	if _, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens:   1,
		Temperature: -0.5,
	}); err == nil {
		t.Fatal("expected invalid temperature error")
	}
}

func TestEngineRejectsInvalidTopK(t *testing.T) {
	engine, err := newTestEngine(t)
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	if _, err := engine.GenerateWithOptions("hi", GenerateOptions{
		MaxTokens: 1,
		TopK:      -1,
	}); err == nil {
		t.Fatal("expected invalid top-k error")
	}
}

func newTestEngine(t *testing.T) (*Engine, error) {
	t.Helper()
	model, err := newTinyModel()
	if err != nil {
		return nil, err
	}
	return NewEngine(tokenizer.NewByteTokenizer(), model, sampler.NewGreedySampler())
}

func newTinyModel() (*transformer.TinyTransformer, error) {
	tok := tokenizer.NewByteTokenizer()
	return transformer.NewTinyTransformer(transformer.DefaultTinyConfig(tok.VocabSize()))
}

type tracingModel struct {
	underlying  model.Model
	callLengths []int
}

func (m *tracingModel) Config() model.Config {
	return m.underlying.Config()
}

func (m *tracingModel) Forward(input []int, cache model.Cache) ([]float64, error) {
	m.callLengths = append(m.callLengths, len(input))
	return m.underlying.Forward(input, cache)
}

type tracingCacheModel struct {
	underlying  model.CacheCapableModel
	callLengths []int
	cacheUsed   bool
}

func (m *tracingCacheModel) Config() model.Config {
	return m.underlying.Config()
}

func (m *tracingCacheModel) Forward(input []int, cache model.Cache) ([]float64, error) {
	m.callLengths = append(m.callLengths, len(input))
	if cache != nil {
		m.cacheUsed = true
	}
	return m.underlying.Forward(input, cache)
}

func (m *tracingCacheModel) NewCache() model.Cache {
	return m.underlying.NewCache()
}
