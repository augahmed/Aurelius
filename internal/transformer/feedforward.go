package transformer

import "github.com/augahmed/aurelius/internal/tensor"

type FeedForward struct {
	upWeights   *tensor.Tensor
	downWeights *tensor.Tensor
}

func NewFeedForward(modelDim, hiddenDim, seed int) (*FeedForward, error) {
	upWeights, err := newWeightMatrix(modelDim, hiddenDim, seed)
	if err != nil {
		return nil, err
	}
	downWeights, err := newWeightMatrix(hiddenDim, modelDim, seed+1)
	if err != nil {
		return nil, err
	}
	return &FeedForward{
		upWeights:   upWeights,
		downWeights: downWeights,
	}, nil
}

func (f *FeedForward) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
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
