package transformer

type KVCache struct{}

func (c *KVCache) Reset() {}

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
