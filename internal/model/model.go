package model

import "fmt"

// Config describes a decoder-only transformer shape at a high level.
type Config struct {
	VocabSize     int
	ContextLength int
	EmbeddingDim  int
	NumLayers     int
	NumHeads      int
}

func (c Config) Validate() error {
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}
	if c.ContextLength <= 0 {
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
	return nil
}

// Cache is a placeholder interface for future KV cache support.
type Cache interface {
	Reset()
}

// NoopCache is a placeholder implementation for prototypes that do not use KV state.
type NoopCache struct{}

func (NoopCache) Reset() {}

// Model is the minimal interface needed by the runtime engine.
type Model interface {
	Config() Config
	Forward(input []int, cache Cache) ([]float64, error)
}
