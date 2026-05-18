package gpt2

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/augahmed/aurelius/internal/model"
)

type Config struct {
	ModelType        string   `json:"model_type"`
	Architectures    []string `json:"architectures"`
	VocabSize        int      `json:"vocab_size"`
	ContextLength    int      `json:"n_ctx"`
	MaxPositions     int      `json:"n_positions"`
	EmbeddingDim     int      `json:"n_embd"`
	NumLayers        int      `json:"n_layer"`
	NumHeads         int      `json:"n_head"`
	FeedForwardDim   int      `json:"n_inner"`
	LayerNormEpsilon float64  `json:"layer_norm_epsilon"`
	BOSTokenID       int      `json:"bos_token_id"`
	EOSTokenID       int      `json:"eos_token_id"`
	Activation       string   `json:"activation_function"`
	ResidualDropout  float64  `json:"resid_pdrop"`
	EmbeddingDropout float64  `json:"embd_pdrop"`
	AttentionDropout float64  `json:"attn_pdrop"`
}

func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config %q: %w", path, err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config %q: %w", path, err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

func (c Config) Validate() error {
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}
	if c.ResolvedContextLength() <= 0 {
		return fmt.Errorf("context length must be positive")
	}
	if c.EmbeddingDim <= 0 {
		return fmt.Errorf("embedding dim must be positive")
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("num layers must be positive")
	}
	if c.NumHeads <= 0 {
		return fmt.Errorf("num heads must be positive")
	}
	if c.EmbeddingDim%c.NumHeads != 0 {
		return fmt.Errorf("embedding dim %d must be divisible by num heads %d", c.EmbeddingDim, c.NumHeads)
	}
	if c.ResolvedFeedForwardDim() <= 0 {
		return fmt.Errorf("feed-forward dim must be positive")
	}
	return nil
}

func (c Config) ResolvedContextLength() int {
	if c.ContextLength > 0 {
		return c.ContextLength
	}
	return c.MaxPositions
}

func (c Config) ResolvedFeedForwardDim() int {
	if c.FeedForwardDim > 0 {
		return c.FeedForwardDim
	}
	return c.EmbeddingDim * 4
}

func (c Config) ResolvedLayerNormEpsilon() float64 {
	if c.LayerNormEpsilon > 0 {
		return c.LayerNormEpsilon
	}
	return 1e-5
}

func (c Config) ModelConfig() model.Config {
	return model.Config{
		VocabSize:     c.VocabSize,
		ContextLength: c.ResolvedContextLength(),
		EmbeddingDim:  c.EmbeddingDim,
		NumLayers:     c.NumLayers,
		NumHeads:      c.NumHeads,
	}
}
