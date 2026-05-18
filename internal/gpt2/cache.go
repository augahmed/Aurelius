package gpt2

import (
	"fmt"

	sharedmodel "github.com/augahmed/aurelius/internal/model"
)

type KVCache struct {
	Keys   []float64
	Values []float64
	Width  int
	SeqLen int
}

func (c *KVCache) Reset() {
	if c == nil {
		return
	}
	c.Keys = nil
	c.Values = nil
	c.SeqLen = 0
}

func (c *KVCache) SequenceLength() int {
	if c == nil {
		return 0
	}
	return c.SeqLen
}

func (c *KVCache) Append(keys, values []float64, rows, width int) error {
	if c == nil {
		return nil
	}
	if rows < 0 || width <= 0 {
		return fmt.Errorf("cache append requires positive row and width values")
	}
	if len(keys) != rows*width || len(values) != rows*width {
		return fmt.Errorf("cache append shape mismatch for rows=%d width=%d", rows, width)
	}
	if c.Width == 0 {
		c.Width = width
	}
	if c.Width != width {
		return fmt.Errorf("cache width = %d, want %d", c.Width, width)
	}
	c.Keys = append(c.Keys, keys...)
	c.Values = append(c.Values, values...)
	c.SeqLen += rows
	return nil
}

type TransformerCache struct {
	Layers []*KVCache
}

func NewTransformerCache(numLayers int) *TransformerCache {
	if numLayers <= 0 {
		return &TransformerCache{}
	}
	layers := make([]*KVCache, numLayers)
	for i := range layers {
		layers[i] = &KVCache{}
	}
	return &TransformerCache{Layers: layers}
}

func (c *TransformerCache) Reset() {
	if c == nil {
		return
	}
	for _, layer := range c.Layers {
		if layer != nil {
			layer.Reset()
		}
	}
}

func (c *TransformerCache) Layer(index int) *KVCache {
	if c == nil || index < 0 || index >= len(c.Layers) {
		return nil
	}
	if c.Layers[index] == nil {
		c.Layers[index] = &KVCache{}
	}
	return c.Layers[index]
}

func (c *TransformerCache) SequenceLength() int {
	if c == nil || len(c.Layers) == 0 || c.Layers[0] == nil {
		return 0
	}
	return c.Layers[0].SequenceLength()
}

func cacheFromModelCache(cache sharedmodel.Cache) *TransformerCache {
	if cache == nil {
		return nil
	}
	transformerCache, ok := cache.(*TransformerCache)
	if !ok {
		return nil
	}
	return transformerCache
}
