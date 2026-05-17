package transformer

import (
	"testing"

	"github.com/augahmed/aurelius/internal/tensor"
)

func TestKVCacheAppend(t *testing.T) {
	cache := &KVCache{}
	keysA := mustTensorFromSlice(tensor.FromSlice([]float64{
		1, 2,
		3, 4,
	}, 2, 2))
	valuesA := mustTensorFromSlice(tensor.FromSlice([]float64{
		5, 6,
		7, 8,
	}, 2, 2))
	keysB := mustTensorFromSlice(tensor.FromSlice([]float64{
		9, 10,
	}, 1, 2))
	valuesB := mustTensorFromSlice(tensor.FromSlice([]float64{
		11, 12,
	}, 1, 2))

	if err := cache.Append(keysA, valuesA); err != nil {
		t.Fatalf("Append(first) error: %v", err)
	}
	if got := cache.SequenceLength(); got != 2 {
		t.Fatalf("cache length = %d, want 2", got)
	}

	if err := cache.Append(keysB, valuesB); err != nil {
		t.Fatalf("Append(second) error: %v", err)
	}
	if got := cache.SequenceLength(); got != 3 {
		t.Fatalf("cache length = %d, want 3", got)
	}

	keys, values := cache.State()
	assertClose2D(t, keys, [][]float64{
		{1, 2},
		{3, 4},
		{9, 10},
	}, 1e-9)
	assertClose2D(t, values, [][]float64{
		{5, 6},
		{7, 8},
		{11, 12},
	}, 1e-9)
}
