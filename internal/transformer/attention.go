package transformer

import (
	"fmt"
	"math"

	"github.com/augahmed/aurelius/internal/tensor"
)

type AttentionOptions struct {
	Cache *KVCache
}

type SelfAttention struct {
	numHeads     int
	headDim      int
	queryWeights *tensor.Tensor
	keyWeights   *tensor.Tensor
	valueWeights *tensor.Tensor
	outWeights   *tensor.Tensor
}

func NewSelfAttention(modelDim, numHeads, seed int) (*SelfAttention, error) {
	if modelDim <= 0 {
		return nil, fmt.Errorf("model dimension must be positive")
	}
	if numHeads <= 0 {
		return nil, fmt.Errorf("number of heads must be positive")
	}
	if modelDim%numHeads != 0 {
		return nil, fmt.Errorf("model dimension %d must be divisible by number of heads %d", modelDim, numHeads)
	}

	queryWeights, err := newWeightMatrix(modelDim, modelDim, seed)
	if err != nil {
		return nil, err
	}
	keyWeights, err := newWeightMatrix(modelDim, modelDim, seed+1)
	if err != nil {
		return nil, err
	}
	valueWeights, err := newWeightMatrix(modelDim, modelDim, seed+2)
	if err != nil {
		return nil, err
	}
	outWeights, err := newWeightMatrix(modelDim, modelDim, seed+3)
	if err != nil {
		return nil, err
	}

	return &SelfAttention{
		numHeads:     numHeads,
		headDim:      modelDim / numHeads,
		queryWeights: queryWeights,
		keyWeights:   keyWeights,
		valueWeights: valueWeights,
		outWeights:   outWeights,
	}, nil
}

func (s *SelfAttention) Forward(input *tensor.Tensor, options *AttentionOptions) (*tensor.Tensor, error) {
	_ = options

	shape := input.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("attention input must be rank-2, got %v", shape)
	}
	seqLen, modelDim := shape[0], shape[1]
	if modelDim != s.numHeads*s.headDim {
		return nil, fmt.Errorf("attention input dimension %d does not match configured dimension %d", modelDim, s.numHeads*s.headDim)
	}

	queries, err := tensor.MatMul(input, s.queryWeights)
	if err != nil {
		return nil, err
	}
	keys, err := tensor.MatMul(input, s.keyWeights)
	if err != nil {
		return nil, err
	}
	values, err := tensor.MatMul(input, s.valueWeights)
	if err != nil {
		return nil, err
	}

	combined, err := tensor.New(seqLen, modelDim)
	if err != nil {
		return nil, err
	}

	for head := 0; head < s.numHeads; head++ {
		headOffset := head * s.headDim
		for queryIndex := 0; queryIndex < seqLen; queryIndex++ {
			weights, err := s.causalWeights(queries, keys, queryIndex, headOffset)
			if err != nil {
				return nil, err
			}
			for valueDim := 0; valueDim < s.headDim; valueDim++ {
				sum := 0.0
				for keyIndex := 0; keyIndex < seqLen; keyIndex++ {
					weight, err := weights.At(keyIndex)
					if err != nil {
						return nil, err
					}
					value, err := values.At(keyIndex, headOffset+valueDim)
					if err != nil {
						return nil, err
					}
					sum += weight * value
				}
				if err := combined.Set(sum, queryIndex, headOffset+valueDim); err != nil {
					return nil, err
				}
			}
		}
	}

	return tensor.MatMul(combined, s.outWeights)
}

func (s *SelfAttention) causalWeights(queries, keys *tensor.Tensor, queryIndex, headOffset int) (*tensor.Tensor, error) {
	shape := queries.Shape()
	seqLen := shape[0]
	scores := make([]float64, seqLen)
	scale := math.Sqrt(float64(s.headDim))

	for keyIndex := 0; keyIndex < seqLen; keyIndex++ {
		if keyIndex > queryIndex {
			scores[keyIndex] = math.Inf(-1)
			continue
		}
		dot := 0.0
		for dim := 0; dim < s.headDim; dim++ {
			queryValue, err := queries.At(queryIndex, headOffset+dim)
			if err != nil {
				return nil, err
			}
			keyValue, err := keys.At(keyIndex, headOffset+dim)
			if err != nil {
				return nil, err
			}
			dot += queryValue * keyValue
		}
		scores[keyIndex] = dot / scale
	}

	scoreTensor, err := tensor.FromSlice(scores, seqLen)
	if err != nil {
		return nil, err
	}
	return tensor.SoftmaxVector(scoreTensor)
}
