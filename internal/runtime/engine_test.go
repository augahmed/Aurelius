package runtime

import (
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
	"github.com/augahmed/aurelius/internal/transformer"
)

func TestEngineGenerate(t *testing.T) {
	tok := tokenizer.NewByteTokenizer()
	model, err := transformer.NewTinyTransformer(transformer.DefaultTinyConfig(tok.VocabSize()))
	if err != nil {
		t.Fatalf("NewTinyTransformer error: %v", err)
	}
	engine, err := NewEngine(tok, model, sampler.NewGreedySampler())
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
	tok := tokenizer.NewByteTokenizer()
	model, err := transformer.NewTinyTransformer(transformer.DefaultTinyConfig(tok.VocabSize()))
	if err != nil {
		t.Fatalf("NewTinyTransformer error: %v", err)
	}
	engine, err := NewEngine(tok, model, sampler.NewGreedySampler())
	if err != nil {
		t.Fatalf("NewEngine error: %v", err)
	}
	if _, err := engine.Generate("", 1); err == nil {
		t.Fatal("expected empty prompt error")
	}
}
