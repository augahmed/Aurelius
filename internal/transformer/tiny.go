package transformer

import (
	"fmt"

	"github.com/augahmed/aurelius/internal/model"
	"github.com/augahmed/aurelius/internal/tensor"
)

type TinyTransformer struct {
	cfg                model.Config
	embeddings         *tensor.Tensor
	positionEmbeddings *tensor.Tensor
	blocks             []block
	output             *tensor.Tensor
}

type block struct {
	attention *multiHeadSelfAttention
	mlp       *feedForward
}

type feedForward struct {
	upWeights   *tensor.Tensor
	downWeights *tensor.Tensor
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
	blocks := make([]block, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		attention, err := newMultiHeadSelfAttention(cfg.EmbeddingDim, cfg.NumHeads, i+1)
		if err != nil {
			return nil, err
		}
		mlp, err := newFeedForward(cfg.EmbeddingDim, cfg.EmbeddingDim*2, cfg.NumLayers+i+1)
		if err != nil {
			return nil, err
		}
		blocks[i] = block{
			attention: attention,
			mlp:       mlp,
		}
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

func (m *TinyTransformer) Forward(input []int, cache model.Cache) ([]float64, error) {
	_ = cache
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	if len(input) > m.cfg.ContextLength {
		return nil, fmt.Errorf("input length %d exceeds context length %d", len(input), m.cfg.ContextLength)
	}
	state, err := m.embed(input)
	if err != nil {
		return nil, err
	}
	for _, blk := range m.blocks {
		state, err = blk.apply(state)
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
			tokenValue, err := m.embeddings.At(token, col)
			if err != nil {
				return nil, err
			}
			positionValue, err := m.positionEmbeddings.At(row, col)
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

func (b block) apply(input *tensor.Tensor) (*tensor.Tensor, error) {
	normInput, err := layerNormRows(input, 1e-5)
	if err != nil {
		return nil, err
	}
	attended, err := b.attention.Forward(normInput)
	if err != nil {
		return nil, err
	}
	withAttentionResidual, err := tensor.Add(input, attended)
	if err != nil {
		return nil, err
	}
	normResidual, err := layerNormRows(withAttentionResidual, 1e-5)
	if err != nil {
		return nil, err
	}
	feedForwardOutput, err := b.mlp.Forward(normResidual)
	if err != nil {
		return nil, err
	}
	return tensor.Add(withAttentionResidual, feedForwardOutput)
}

func (f *feedForward) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	expanded, err := tensor.MatMul(input, f.upWeights)
	if err != nil {
		return nil, err
	}
	activated, err := tensor.GELUApprox(expanded)
	if err != nil {
		return nil, err
	}
	return tensor.MatMul(activated, f.downWeights)
}

func newFeedForward(modelDim, hiddenDim, seed int) (*feedForward, error) {
	upWeights, err := newWeightMatrix(modelDim, hiddenDim, seed)
	if err != nil {
		return nil, err
	}
	downWeights, err := newWeightMatrix(hiddenDim, modelDim, seed+1)
	if err != nil {
		return nil, err
	}
	return &feedForward{
		upWeights:   upWeights,
		downWeights: downWeights,
	}, nil
}

func layerNormRows(t *tensor.Tensor, epsilon float64) (*tensor.Tensor, error) {
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("layerNormRows requires rank-2 tensor, got %v", shape)
	}
	rows, cols := shape[0], shape[1]
	out, err := tensor.New(rows, cols)
	if err != nil {
		return nil, err
	}
	for row := 0; row < rows; row++ {
		rowValues := make([]float64, cols)
		for col := 0; col < cols; col++ {
			value, err := t.At(row, col)
			if err != nil {
				return nil, err
			}
			rowValues[col] = value
		}
		rowTensor, err := tensor.FromSlice(rowValues, cols)
		if err != nil {
			return nil, err
		}
		normRow, err := tensor.LayerNorm(rowTensor, epsilon)
		if err != nil {
			return nil, err
		}
		for col := 0; col < cols; col++ {
			value, err := normRow.At(col)
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
