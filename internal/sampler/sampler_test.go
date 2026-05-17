package sampler

import (
	"math/rand"
	"testing"
)

func TestGreedySampler(t *testing.T) {
	s := NewGreedySampler()
	got, err := s.Sample([]float64{0.5, 2.0, 1.0})
	if err != nil {
		t.Fatalf("Sample error: %v", err)
	}
	if got != 1 {
		t.Fatalf("got token %d, want 1", got)
	}
}

func TestTemperatureSamplerDeterministicSeed(t *testing.T) {
	s := NewTemperatureSampler(0.5, rand.New(rand.NewSource(7)))
	got, err := s.Sample([]float64{0.1, 0.2, 3.0})
	if err != nil {
		t.Fatalf("Sample error: %v", err)
	}
	if got != 2 {
		t.Fatalf("got token %d, want 2", got)
	}
}

func TestTemperatureSamplerRejectsInvalidTemperature(t *testing.T) {
	s := NewTemperatureSampler(0, rand.New(rand.NewSource(1)))
	if _, err := s.Sample([]float64{1, 2, 3}); err == nil {
		t.Fatal("expected invalid temperature error")
	}
}
