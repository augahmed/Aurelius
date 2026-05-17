package tensor

import (
	"fmt"
	"math"
)

// Tensor is a dense float64 tensor stored in row-major order.
type Tensor struct {
	data  []float64
	shape []int
}

func New(shape ...int) (*Tensor, error) {
	size, err := elementCount(shape)
	if err != nil {
		return nil, err
	}
	return &Tensor{
		data:  make([]float64, size),
		shape: append([]int(nil), shape...),
	}, nil
}

func FromSlice(data []float64, shape ...int) (*Tensor, error) {
	size, err := elementCount(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != size {
		return nil, fmt.Errorf("tensor data length %d does not match shape size %d", len(data), size)
	}
	copied := append([]float64(nil), data...)
	return &Tensor{
		data:  copied,
		shape: append([]int(nil), shape...),
	}, nil
}

func (t *Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}

func (t *Tensor) Data() []float64 {
	return append([]float64(nil), t.data...)
}

func (t *Tensor) Rank() int {
	return len(t.shape)
}

func (t *Tensor) At(indices ...int) (float64, error) {
	offset, err := t.offset(indices...)
	if err != nil {
		return 0, err
	}
	return t.data[offset], nil
}

func (t *Tensor) Set(value float64, indices ...int) error {
	offset, err := t.offset(indices...)
	if err != nil {
		return err
	}
	t.data[offset] = value
	return nil
}

func (t *Tensor) offset(indices ...int) (int, error) {
	if len(indices) != len(t.shape) {
		return 0, fmt.Errorf("expected %d indices, got %d", len(t.shape), len(indices))
	}
	offset := 0
	stride := 1
	for dim := len(t.shape) - 1; dim >= 0; dim-- {
		index := indices[dim]
		size := t.shape[dim]
		if index < 0 || index >= size {
			return 0, fmt.Errorf("index %d out of bounds for dimension %d with size %d", index, dim, size)
		}
		offset += index * stride
		stride *= size
	}
	return offset, nil
}

func Add(a, b *Tensor) (*Tensor, error) {
	if !sameShape(a.shape, b.shape) {
		return nil, fmt.Errorf("cannot add tensors with shapes %v and %v", a.shape, b.shape)
	}
	out, err := New(a.shape...)
	if err != nil {
		return nil, err
	}
	for i := range a.data {
		out.data[i] = a.data[i] + b.data[i]
	}
	return out, nil
}

func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, fmt.Errorf("matmul requires rank-2 tensors, got %d and %d", a.Rank(), b.Rank())
	}
	rowsA, colsA := a.shape[0], a.shape[1]
	rowsB, colsB := b.shape[0], b.shape[1]
	if colsA != rowsB {
		return nil, fmt.Errorf("matmul shape mismatch: %v x %v", a.shape, b.shape)
	}
	out, err := New(rowsA, colsB)
	if err != nil {
		return nil, err
	}
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := 0.0
			for k := 0; k < colsA; k++ {
				sum += a.data[i*colsA+k] * b.data[k*colsB+j]
			}
			out.data[i*colsB+j] = sum
		}
	}
	return out, nil
}

func SoftmaxVector(t *Tensor) (*Tensor, error) {
	if t.Rank() != 1 {
		return nil, fmt.Errorf("softmax requires a rank-1 tensor, got rank %d", t.Rank())
	}
	if len(t.data) == 0 {
		return nil, fmt.Errorf("softmax requires a non-empty tensor")
	}
	maxVal := t.data[0]
	for _, v := range t.data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	out, err := New(len(t.data))
	if err != nil {
		return nil, err
	}
	for i, v := range t.data {
		expVal := math.Exp(v - maxVal)
		out.data[i] = expVal
		expSum += expVal
	}
	for i := range out.data {
		out.data[i] /= expSum
	}
	return out, nil
}

func ReLU(t *Tensor) (*Tensor, error) {
	out, err := New(t.shape...)
	if err != nil {
		return nil, err
	}
	for i, v := range t.data {
		if v > 0 {
			out.data[i] = v
		}
	}
	return out, nil
}

func GELUApprox(t *Tensor) (*Tensor, error) {
	out, err := New(t.shape...)
	if err != nil {
		return nil, err
	}
	const coeff = 0.044715
	for i, x := range t.data {
		out.data[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+coeff*x*x*x)))
	}
	return out, nil
}

func LayerNorm(t *Tensor, epsilon float64) (*Tensor, error) {
	if t.Rank() != 1 {
		return nil, fmt.Errorf("layer norm requires a rank-1 tensor, got rank %d", t.Rank())
	}
	if len(t.data) == 0 {
		return nil, fmt.Errorf("layer norm requires a non-empty tensor")
	}
	mean := 0.0
	for _, v := range t.data {
		mean += v
	}
	mean /= float64(len(t.data))

	variance := 0.0
	for _, v := range t.data {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(t.data))

	out, err := New(len(t.data))
	if err != nil {
		return nil, err
	}
	denom := math.Sqrt(variance + epsilon)
	for i, v := range t.data {
		out.data[i] = (v - mean) / denom
	}
	return out, nil
}

func elementCount(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("tensor shape cannot be empty")
	}
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			return 0, fmt.Errorf("tensor dimensions must be positive, got %d", dim)
		}
		size *= dim
	}
	return size, nil
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
