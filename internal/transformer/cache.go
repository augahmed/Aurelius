package transformer

import (
	"fmt"

	"github.com/augahmed/aurelius/internal/tensor"
)

type KVCache struct {
	keys   *tensor.Tensor
	values *tensor.Tensor
}

func (c *KVCache) Reset() {
	if c == nil {
		return
	}
	c.keys = nil
	c.values = nil
}

func (c *KVCache) SequenceLength() int {
	if c == nil || c.keys == nil {
		return 0
	}
	shape := c.keys.Shape()
	if len(shape) != 2 {
		return 0
	}
	return shape[0]
}

func (c *KVCache) State() (*tensor.Tensor, *tensor.Tensor) {
	if c == nil {
		return nil, nil
	}
	return c.keys, c.values
}

func (c *KVCache) Append(keys, values *tensor.Tensor) error {
	if c == nil {
		return nil
	}
	if keys == nil || values == nil {
		return fmt.Errorf("cache append requires non-nil keys and values")
	}
	if !sameMatrixShape(keys, values) {
		return fmt.Errorf("cache append requires matching key/value shapes, got %v and %v", keys.Shape(), values.Shape())
	}
	if c.keys == nil {
		clonedKeys, err := tensor.FromSlice(keys.Data(), keys.Shape()...)
		if err != nil {
			return err
		}
		clonedValues, err := tensor.FromSlice(values.Data(), values.Shape()...)
		if err != nil {
			return err
		}
		c.keys = clonedKeys
		c.values = clonedValues
		return nil
	}
	combinedKeys, err := concatRows(c.keys, keys)
	if err != nil {
		return err
	}
	combinedValues, err := concatRows(c.values, values)
	if err != nil {
		return err
	}
	c.keys = combinedKeys
	c.values = combinedValues
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

func concatRows(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if a == nil {
		if b == nil {
			return nil, nil
		}
		return tensor.FromSlice(b.Data(), b.Shape()...)
	}
	if b == nil {
		return tensor.FromSlice(a.Data(), a.Shape()...)
	}
	shapeA := a.Shape()
	shapeB := b.Shape()
	if len(shapeA) != 2 || len(shapeB) != 2 {
		return nil, fmt.Errorf("concatRows requires rank-2 tensors, got %v and %v", shapeA, shapeB)
	}
	if shapeA[1] != shapeB[1] {
		return nil, fmt.Errorf("concatRows requires equal column counts, got %v and %v", shapeA, shapeB)
	}
	combined := append(a.Data(), b.Data()...)
	return tensor.FromSlice(combined, shapeA[0]+shapeB[0], shapeA[1])
}

func sameMatrixShape(a, b *tensor.Tensor) bool {
	if a == nil || b == nil {
		return false
	}
	shapeA := a.Shape()
	shapeB := b.Shape()
	return len(shapeA) == 2 && len(shapeB) == 2 && shapeA[0] == shapeB[0] && shapeA[1] == shapeB[1]
}
