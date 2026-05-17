package transformer

import "github.com/augahmed/aurelius/internal/tensor"

type BlockForwardOptions struct {
	Attention *AttentionOptions
}

type DecoderBlock struct {
	preAttentionNorm *LayerNorm
	attention        *SelfAttention
	preMLPNorm       *LayerNorm
	feedForward      *FeedForward
}

func NewDecoderBlock(modelDim, numHeads, hiddenDim, attentionSeed, feedForwardSeed int, epsilon float64) (*DecoderBlock, error) {
	attention, err := NewSelfAttention(modelDim, numHeads, attentionSeed)
	if err != nil {
		return nil, err
	}
	feedForward, err := NewFeedForward(modelDim, hiddenDim, feedForwardSeed)
	if err != nil {
		return nil, err
	}
	return &DecoderBlock{
		preAttentionNorm: NewLayerNorm(epsilon),
		attention:        attention,
		preMLPNorm:       NewLayerNorm(epsilon),
		feedForward:      feedForward,
	}, nil
}

func (b *DecoderBlock) Forward(input *tensor.Tensor, options *BlockForwardOptions) (*tensor.Tensor, error) {
	normInput, err := b.preAttentionNorm.Forward(input)
	if err != nil {
		return nil, err
	}

	var attentionOptions *AttentionOptions
	if options != nil {
		attentionOptions = options.Attention
	}
	attended, err := b.attention.Forward(normInput, attentionOptions)
	if err != nil {
		return nil, err
	}
	withAttentionResidual, err := tensor.Add(input, attended)
	if err != nil {
		return nil, err
	}

	normResidual, err := b.preMLPNorm.Forward(withAttentionResidual)
	if err != nil {
		return nil, err
	}
	feedForwardOutput, err := b.feedForward.Forward(normResidual)
	if err != nil {
		return nil, err
	}
	return tensor.Add(withAttentionResidual, feedForwardOutput)
}
