package gpt2

import "testing"

func TestKVCacheAppend(t *testing.T) {
	cache := &KVCache{}

	if err := cache.Append([]float64{
		1, 2,
		3, 4,
	}, []float64{
		5, 6,
		7, 8,
	}, 2, 2); err != nil {
		t.Fatalf("Append(first) error: %v", err)
	}
	if got := cache.SequenceLength(); got != 2 {
		t.Fatalf("cache length = %d, want %d", got, 2)
	}

	if err := cache.Append([]float64{
		9, 10,
	}, []float64{
		11, 12,
	}, 1, 2); err != nil {
		t.Fatalf("Append(second) error: %v", err)
	}
	if got := cache.SequenceLength(); got != 3 {
		t.Fatalf("cache length = %d, want %d", got, 3)
	}
	if len(cache.Keys) != 6 || len(cache.Values) != 6 {
		t.Fatalf("cache sizes = (%d, %d), want (6, 6)", len(cache.Keys), len(cache.Values))
	}
}
