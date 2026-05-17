package transformer

import (
	"math"
	"testing"

	"github.com/augahmed/aurelius/internal/tensor"
)

func TestMultiHeadSelfAttentionForwardShape(t *testing.T) {
	attention, err := newMultiHeadSelfAttention(4, 2, 1)
	if err != nil {
		t.Fatalf("newMultiHeadSelfAttention error: %v", err)
	}
	input := mustTensorFromSlice(tensor.FromSlice([]float64{
		1, 0, 0, 1,
		0, 1, 1, 0,
		1, 1, 0, 0,
	}, 3, 4))

	output, err := attention.Forward(input)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	shape := output.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 4 {
		t.Fatalf("output shape = %v, want [3 4]", shape)
	}
}

func TestMultiHeadSelfAttentionAppliesCausalMask(t *testing.T) {
	attention := identityAttention(2, 1)
	base := mustTensorFromSlice(tensor.FromSlice([]float64{
		1, 0,
		0, 1,
		2, 2,
	}, 3, 2))
	modifiedFuture := mustTensorFromSlice(tensor.FromSlice([]float64{
		1, 0,
		0, 1,
		-3, 5,
	}, 3, 2))

	baseOutput, err := attention.Forward(base)
	if err != nil {
		t.Fatalf("Forward(base) error: %v", err)
	}
	modifiedOutput, err := attention.Forward(modifiedFuture)
	if err != nil {
		t.Fatalf("Forward(modifiedFuture) error: %v", err)
	}

	for _, row := range []int{0, 1} {
		for col := 0; col < 2; col++ {
			got, err := baseOutput.At(row, col)
			if err != nil {
				t.Fatalf("baseOutput.At(%d, %d) error: %v", row, col, err)
			}
			want, err := modifiedOutput.At(row, col)
			if err != nil {
				t.Fatalf("modifiedOutput.At(%d, %d) error: %v", row, col, err)
			}
			if math.Abs(got-want) > 1e-9 {
				t.Fatalf("row %d col %d changed with future token: got %v want %v", row, col, got, want)
			}
		}
	}
}

func TestMultiHeadSelfAttentionDeterministicOutput(t *testing.T) {
	attention := identityAttention(2, 1)
	input := mustTensorFromSlice(tensor.FromSlice([]float64{
		1, 0,
		0, 1,
	}, 2, 2))

	output, err := attention.Forward(input)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}

	score := 1 / math.Sqrt2
	secondRowFirst := 1 / (1 + math.Exp(score))
	secondRowSecond := math.Exp(score) / (1 + math.Exp(score))

	assertClose2D(t, output, [][]float64{
		{1, 0},
		{secondRowFirst, secondRowSecond},
	}, 1e-9)
}

func TestNewMultiHeadSelfAttentionRejectsInvalidHeadLayout(t *testing.T) {
	if _, err := newMultiHeadSelfAttention(3, 2, 1); err == nil {
		t.Fatal("expected invalid head layout error")
	}
}

func identityAttention(modelDim, numHeads int) *multiHeadSelfAttention {
	identity := identityMatrix(modelDim)
	return &multiHeadSelfAttention{
		numHeads:     numHeads,
		headDim:      modelDim / numHeads,
		queryWeights: identity,
		keyWeights:   identity,
		valueWeights: identity,
		outWeights:   identity,
	}
}

func identityMatrix(size int) *tensor.Tensor {
	data := make([]float64, size*size)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1
	}
	return mustTensorFromSlice(tensor.FromSlice(data, size, size))
}

func assertClose2D(t *testing.T, got *tensor.Tensor, want [][]float64, tolerance float64) {
	t.Helper()
	shape := got.Shape()
	if len(shape) != 2 || shape[0] != len(want) || shape[1] != len(want[0]) {
		t.Fatalf("shape = %v, want [%d %d]", shape, len(want), len(want[0]))
	}
	for row := range want {
		for col := range want[row] {
			value, err := got.At(row, col)
			if err != nil {
				t.Fatalf("At(%d, %d) error: %v", row, col, err)
			}
			if math.Abs(value-want[row][col]) > tolerance {
				t.Fatalf("value at (%d, %d) = %.12f, want %.12f", row, col, value, want[row][col])
			}
		}
	}
}

func mustTensorFromSlice(tn *tensor.Tensor, err error) *tensor.Tensor {
	if err != nil {
		panic(err)
	}
	return tn
}
