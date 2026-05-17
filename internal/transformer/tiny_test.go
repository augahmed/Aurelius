package transformer

import "testing"

func TestTinyTransformerForward(t *testing.T) {
	model, err := NewTinyTransformer(DefaultTinyConfig(256))
	if err != nil {
		t.Fatalf("NewTinyTransformer error: %v", err)
	}
	logits, err := model.Forward([]int{104, 101, 108, 108, 111}, nil)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if len(logits) != model.Config().VocabSize {
		t.Fatalf("logits length = %d, want %d", len(logits), model.Config().VocabSize)
	}
}

func TestTinyTransformerRejectsTooLongContext(t *testing.T) {
	cfg := DefaultTinyConfig(256)
	cfg.ContextLength = 2
	model, err := NewTinyTransformer(cfg)
	if err != nil {
		t.Fatalf("NewTinyTransformer error: %v", err)
	}
	if _, err := model.Forward([]int{1, 2, 3}, nil); err == nil {
		t.Fatal("expected context length error")
	}
}
