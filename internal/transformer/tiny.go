package transformer

import (
	"fmt"

	"github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/tensor"
)

type TinyTransformer struct {
	cfg        model.Config
	embeddings *tensor.Tensor
	blocks     []block
	output     *tensor.Tensor
}

type block struct {
	linear *tensor.Tensor
}

func NewTinyTransformer(cfg model.Config) (*TinyTransformer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	embeddings, err := newEmbeddingMatrix(cfg.VocabSize, cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	blocks := make([]block, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		linear, err := newWeightMatrix(cfg.EmbeddingDim, cfg.EmbeddingDim, i+1)
		if err != nil {
			return nil, err
		}
		blocks[i] = block{linear: linear}
	}
	output, err := newWeightMatrix(cfg.EmbeddingDim, cfg.VocabSize, cfg.NumLayers+2)
	if err != nil {
		return nil, err
	}
	return &TinyTransformer{
		cfg:        cfg,
		embeddings: embeddings,
		blocks:     blocks,
		output:     output,
	}, nil
}

func DefaultTinyConfig(vocabSize int) model.Config {
	return model.Config{
		VocabSize:     vocabSize,
		ContextLength: 128,
		EmbeddingDim:  16,
		NumLayers:     2,
		NumHeads:      1,
	}
}

func (m *TinyTransformer) Config() model.Config {
	return m.cfg
}

func (m *TinyTransformer) Forward(input []int, cache model.Cache) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	if len(input) > m.cfg.ContextLength {
		return nil, fmt.Errorf("input length %d exceeds context length %d", len(input), m.cfg.ContextLength)
	}
	sequence, err := m.embed(input)
	if err != nil {
		return nil, err
	}
	state, err := meanRows(sequence)
	if err != nil {
		return nil, err
	}
	for _, blk := range m.blocks {
		state, err = blk.apply(state)
		if err != nil {
			return nil, err
		}
	}
	logits, err := project(state, m.output)
	if err != nil {
		return nil, err
	}
	return logits.Data(), nil
}

func (m *TinyTransformer) embed(tokens []int) (*tensor.Tensor, error) {
	out, err := tensor.New(len(tokens), m.cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	for row, token := range tokens {
		if token < 0 || token >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token %d out of range", token)
		}
		for col := 0; col < m.cfg.EmbeddingDim; col++ {
			value, err := m.embeddings.At(token, col)
			if err != nil {
				return nil, err
			}
			if err := out.Set(value, row, col); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func (b block) apply(input *tensor.Tensor) (*tensor.Tensor, error) {
	row, err := tensor.FromSlice(input.Data(), 1, len(input.Data()))
	if err != nil {
		return nil, err
	}
	projected, err := tensor.MatMul(row, b.linear)
	if err != nil {
		return nil, err
	}
	activated, err := tensor.GELUApprox(projected)
	if err != nil {
		return nil, err
	}
	activatedVec, err := flattenRow(activated)
	if err != nil {
		return nil, err
	}
	residual, err := tensor.Add(input, activatedVec)
	if err != nil {
		return nil, err
	}
	return tensor.LayerNorm(residual, 1e-5)
}

func meanRows(t *tensor.Tensor) (*tensor.Tensor, error) {
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("meanRows requires rank-2 tensor, got %v", shape)
	}
	rows, cols := shape[0], shape[1]
	out, err := tensor.New(cols)
	if err != nil {
		return nil, err
	}
	for col := 0; col < cols; col++ {
		sum := 0.0
		for row := 0; row < rows; row++ {
			value, err := t.At(row, col)
			if err != nil {
				return nil, err
			}
			sum += value
		}
		if err := out.Set(sum/float64(rows), col); err != nil {
			return nil, err
		}
	}
	return out, nil
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
