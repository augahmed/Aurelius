package gpt2

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	content := `{
		"model_type": "gpt2",
		"vocab_size": 50257,
		"n_positions": 1024,
		"n_embd": 768,
		"n_layer": 12,
		"n_head": 12,
		"layer_norm_epsilon": 1e-5,
		"bos_token_id": 50256,
		"eos_token_id": 50256
	}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig error: %v", err)
	}

	if cfg.ResolvedContextLength() != 1024 {
		t.Fatalf("ResolvedContextLength() = %d, want %d", cfg.ResolvedContextLength(), 1024)
	}
	if cfg.ResolvedFeedForwardDim() != 3072 {
		t.Fatalf("ResolvedFeedForwardDim() = %d, want %d", cfg.ResolvedFeedForwardDim(), 3072)
	}

	modelCfg := cfg.ModelConfig()
	if modelCfg.VocabSize != 50257 || modelCfg.ContextLength != 1024 || modelCfg.EmbeddingDim != 768 || modelCfg.NumLayers != 12 || modelCfg.NumHeads != 12 {
		t.Fatalf("ModelConfig() = %+v, want GPT-2 dimensions", modelCfg)
	}
}

func TestConfigValidateRejectsInvalidHeads(t *testing.T) {
	cfg := Config{
		VocabSize:     100,
		ContextLength: 32,
		EmbeddingDim:  10,
		NumLayers:     2,
		NumHeads:      3,
	}

	if err := cfg.Validate(); err == nil {
		t.Fatal("expected Validate error")
	}
}
