package transformer

import (
	"testing"

	"github.com/augahmed/aurelius/internal/tensor"
)

func TestDecoderBlockForwardPreservesShape(t *testing.T) {
	block, err := NewDecoderBlock(4, 2, 8, 1, 3, defaultNormEpsilon)
	if err != nil {
		t.Fatalf("NewDecoderBlock error: %v", err)
	}
	input := mustTensorFromSlice(tensor.FromSlice([]float64{
		1, 0, 0, 1,
		0, 1, 1, 0,
		1, 1, 0, 0,
	}, 3, 4))

	output, err := block.Forward(input, nil)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	shape := output.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 4 {
		t.Fatalf("output shape = %v, want [3 4]", shape)
	}
}
