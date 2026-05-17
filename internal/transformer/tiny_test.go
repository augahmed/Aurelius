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

func TestTinyTransformerForwardWithTransformerCacheMatchesNilCache(t *testing.T) {
	model, err := NewTinyTransformer(DefaultTinyConfig(256))
	if err != nil {
		t.Fatalf("NewTinyTransformer error: %v", err)
	}

	withoutCache, err := model.Forward([]int{104, 101, 108, 108, 111}, nil)
	if err != nil {
		t.Fatalf("Forward(nil) error: %v", err)
	}
	withCache, err := model.Forward([]int{104, 101, 108, 108, 111}, NewTransformerCache(model.Config().NumLayers))
	if err != nil {
		t.Fatalf("Forward(with cache) error: %v", err)
	}
	if len(withoutCache) != len(withCache) {
		t.Fatalf("logit lengths differ: %d vs %d", len(withoutCache), len(withCache))
	}
	for i := range withoutCache {
		if withoutCache[i] != withCache[i] {
			t.Fatalf("logit[%d] = %v, want %v", i, withCache[i], withoutCache[i])
		}
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

func TestTinyTransformerRejectsInvalidConfig(t *testing.T) {
	cfg := DefaultTinyConfig(256)
	cfg.EmbeddingDim = 10
	cfg.NumHeads = 3
	if _, err := NewTinyTransformer(cfg); err == nil {
		t.Fatal("expected invalid config error")
	}
}
