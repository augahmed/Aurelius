package transformer

import (
	"fmt"

	"github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/tensor"
)

const defaultNormEpsilon = 1e-5

type TinyTransformer struct {
	cfg                model.Config
	embeddings         *tensor.Tensor
	positionEmbeddings *tensor.Tensor
	blocks             []*DecoderBlock
	output             *tensor.Tensor
}

func NewTinyTransformer(cfg model.Config) (*TinyTransformer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	embeddings, err := newEmbeddingMatrix(cfg.VocabSize, cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	positionEmbeddings, err := newPositionEmbeddingMatrix(cfg.ContextLength, cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}

	blocks := make([]*DecoderBlock, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		block, err := NewDecoderBlock(
			cfg.EmbeddingDim,
			cfg.NumHeads,
			cfg.EmbeddingDim*2,
			i+1,
			cfg.NumLayers+i+1,
			defaultNormEpsilon,
		)
		if err != nil {
			return nil, err
		}
		blocks[i] = block
	}

	output, err := newWeightMatrix(cfg.EmbeddingDim, cfg.VocabSize, cfg.NumLayers*2+4)
	if err != nil {
		return nil, err
	}
	return &TinyTransformer{
		cfg:                cfg,
		embeddings:         embeddings,
		positionEmbeddings: positionEmbeddings,
		blocks:             blocks,
		output:             output,
	}, nil
}

func DefaultTinyConfig(vocabSize int) model.Config {
	return model.Config{
		VocabSize:     vocabSize,
		ContextLength: 128,
		EmbeddingDim:  16,
		NumLayers:     2,
		NumHeads:      4,
	}
}

func (m *TinyTransformer) Config() model.Config {
	return m.cfg
}

func (m *TinyTransformer) NewCache() model.Cache {
	return NewTransformerCache(m.cfg.NumLayers)
}

func (m *TinyTransformer) Forward(input []int, cache model.Cache) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	transformerCache := cacheFromModelCache(cache)
	positionOffset := 0
	if transformerCache != nil {
		positionOffset = transformerCache.SequenceLength()
	}
	totalLength := positionOffset + len(input)
	if totalLength > m.cfg.ContextLength {
		return nil, fmt.Errorf("input length %d with cache length %d exceeds context length %d", len(input), positionOffset, m.cfg.ContextLength)
	}

	state, err := m.embed(input, positionOffset)
	if err != nil {
		return nil, err
	}

	for i, block := range m.blocks {
		state, err = block.Forward(state, blockOptions(transformerCache, i))
		if err != nil {
			return nil, err
		}
	}

	lastTokenState, err := selectRow(state, len(input)-1)
	if err != nil {
		return nil, err
	}
	logits, err := project(lastTokenState, m.output)
	if err != nil {
		return nil, err
	}
	return logits.Data(), nil
}

func (m *TinyTransformer) embed(tokens []int, positionOffset int) (*tensor.Tensor, error) {
	out, err := tensor.New(len(tokens), m.cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	for row, token := range tokens {
		if token < 0 || token >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token %d out of range", token)
		}
		for col := 0; col < m.cfg.EmbeddingDim; col++ {
			tokenValue, err := m.embeddings.At(token, col)
			if err != nil {
				return nil, err
			}
			positionValue, err := m.positionEmbeddings.At(positionOffset+row, col)
			if err != nil {
				return nil, err
			}
			if err := out.Set(tokenValue+positionValue, row, col); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func blockOptions(cache *TransformerCache, index int) *BlockForwardOptions {
	if cache == nil {
		return nil
	}
	return &BlockForwardOptions{
		Attention: &AttentionOptions{
			Cache: cache.Layer(index),
		},
	}
}

func cacheFromModelCache(cache model.Cache) *TransformerCache {
	if cache == nil {
		return nil
	}
	transformerCache, ok := cache.(*TransformerCache)
	if !ok {
		return nil
	}
	return transformerCache
}

func selectRow(t *tensor.Tensor, row int) (*tensor.Tensor, error) {
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("selectRow requires rank-2 tensor, got %v", shape)
	}
	if row < 0 || row >= shape[0] {
		return nil, fmt.Errorf("row %d out of bounds for tensor with %d rows", row, shape[0])
	}
	values := make([]float64, shape[1])
	for col := 0; col < shape[1]; col++ {
		value, err := t.At(row, col)
		if err != nil {
			return nil, err
		}
		values[col] = value
	}
	return tensor.FromSlice(values, shape[1])
}

func newPositionEmbeddingMatrix(contextLength, dim int) (*tensor.Tensor, error) {
	data := make([]float64, contextLength*dim)
	for pos := 0; pos < contextLength; pos++ {
		for col := 0; col < dim; col++ {
			idx := pos*dim + col
			data[idx] = float64(((pos+1)*(col+5))%13-6) / 16.0
		}
	}
	return tensor.FromSlice(data, contextLength, dim)
}

func project(vector *tensor.Tensor, weights *tensor.Tensor) (*tensor.Tensor, error) {
	row, err := tensor.FromSlice(vector.Data(), 1, len(vector.Data()))
	if err != nil {
		return nil, err
	}
	projected, err := tensor.MatMul(row, weights)
	if err != nil {
		return nil, err
	}
	return flattenRow(projected)
}

func flattenRow(t *tensor.Tensor) (*tensor.Tensor, error) {
	shape := t.Shape()
	if len(shape) != 2 || shape[0] != 1 {
		return nil, fmt.Errorf("flattenRow requires shape [1, n], got %v", shape)
	}
	return tensor.FromSlice(t.Data(), shape[1])
}

func newEmbeddingMatrix(vocabSize, dim int) (*tensor.Tensor, error) {
	data := make([]float64, vocabSize*dim)
	for token := 0; token < vocabSize; token++ {
		for col := 0; col < dim; col++ {
			idx := token*dim + col
			data[idx] = float64(((token+1)*(col+3))%17-8) / 8.0
		}
	}
	return tensor.FromSlice(data, vocabSize, dim)
}

func newWeightMatrix(rows, cols, seed int) (*tensor.Tensor, error) {
	data := make([]float64, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			idx := row*cols + col
			data[idx] = float64(((row+1)*(col+seed+1))%11-5) / float64(cols)
		}
	}
	return tensor.FromSlice(data, rows, cols)
}
