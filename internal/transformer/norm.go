package transformer

import (
	"fmt"

	"github.com/augahmed/aurelius/internal/tensor"
)

type LayerNorm struct {
	epsilon float64
}

func NewLayerNorm(epsilon float64) *LayerNorm {
	return &LayerNorm{epsilon: epsilon}
}

func (n *LayerNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	switch input.Rank() {
	case 1:
		return tensor.LayerNorm(input, n.epsilon)
	case 2:
		return n.forwardRows(input)
	default:
		return nil, fmt.Errorf("layer norm requires rank-1 or rank-2 tensor, got rank %d", input.Rank())
	}
}

func (n *LayerNorm) forwardRows(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.Shape()
	rows, cols := shape[0], shape[1]
	out, err := tensor.New(rows, cols)
	if err != nil {
		return nil, err
	}
	for row := 0; row < rows; row++ {
		rowValues := make([]float64, cols)
		for col := 0; col < cols; col++ {
			value, err := input.At(row, col)
			if err != nil {
				return nil, err
			}
			rowValues[col] = value
		}
		rowTensor, err := tensor.FromSlice(rowValues, cols)
		if err != nil {
			return nil, err
		}
		normRow, err := tensor.LayerNorm(rowTensor, n.epsilon)
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
