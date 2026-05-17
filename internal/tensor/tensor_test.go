package tensor

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	a, err := FromSlice([]float64{1, 2, 3}, 3)
	if err != nil {
		t.Fatalf("FromSlice(a) error: %v", err)
	}
	b, err := FromSlice([]float64{4, 5, 6}, 3)
	if err != nil {
		t.Fatalf("FromSlice(b) error: %v", err)
	}
	got, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add error: %v", err)
	}
	want := []float64{5, 7, 9}
	for i, v := range got.data {
		if v != want[i] {
			t.Fatalf("Add[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestAddShapeMismatch(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2}, 2)
	b, _ := FromSlice([]float64{1, 2, 3}, 3)
	if _, err := Add(a, b); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestMatMul(t *testing.T) {
	a, _ := FromSlice([]float64{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	b, _ := FromSlice([]float64{
		7, 8,
		9, 10,
		11, 12,
	}, 3, 2)
	got, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul error: %v", err)
	}
	want := []float64{58, 64, 139, 154}
	for i, v := range got.data {
		if v != want[i] {
			t.Fatalf("MatMul[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestSoftmaxVector(t *testing.T) {
	input, _ := FromSlice([]float64{1, 2, 3}, 3)
	got, err := SoftmaxVector(input)
	if err != nil {
		t.Fatalf("SoftmaxVector error: %v", err)
	}
	sum := 0.0
	for _, v := range got.data {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Fatalf("softmax sum = %v, want 1", sum)
	}
	if !(got.data[2] > got.data[1] && got.data[1] > got.data[0]) {
		t.Fatalf("softmax ordering unexpected: %v", got.data)
	}
}

func TestLayerNorm(t *testing.T) {
	input, _ := FromSlice([]float64{1, 2, 3}, 3)
	got, err := LayerNorm(input, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm error: %v", err)
	}
	mean := 0.0
	for _, v := range got.data {
		mean += v
	}
	mean /= float64(len(got.data))
	if math.Abs(mean) > 1e-9 {
		t.Fatalf("layer norm mean = %v, want near 0", mean)
	}
}

func TestGELUApprox(t *testing.T) {
	input, _ := FromSlice([]float64{-1, 0, 1}, 3)
	got, err := GELUApprox(input)
	if err != nil {
		t.Fatalf("GELUApprox error: %v", err)
	}
	if got.data[0] >= 0 {
		t.Fatalf("expected negative gelu output for -1, got %v", got.data[0])
	}
	if math.Abs(got.data[1]) > 1e-9 {
		t.Fatalf("expected gelu(0) near 0, got %v", got.data[1])
	}
	if got.data[2] <= 0 {
		t.Fatalf("expected positive gelu output for 1, got %v", got.data[2])
	}
}
