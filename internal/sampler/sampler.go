package sampler

import (
	"fmt"
	"math"
	"math/rand"
)

type Sampler interface {
	Sample(logits []float64) (int, error)
}

type GreedySampler struct{}

func NewGreedySampler() *GreedySampler {
	return &GreedySampler{}
}

func (s *GreedySampler) Sample(logits []float64) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("cannot sample from empty logits")
	}
	bestIdx := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return bestIdx, nil
}

type TemperatureSampler struct {
	Temperature float64
	Rand        *rand.Rand
}

func NewTemperatureSampler(temperature float64, source *rand.Rand) *TemperatureSampler {
	return &TemperatureSampler{
		Temperature: temperature,
		Rand:        source,
	}
}

func (s *TemperatureSampler) Sample(logits []float64) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("cannot sample from empty logits")
	}
	if s.Temperature <= 0 {
		return 0, fmt.Errorf("temperature must be positive")
	}
	source := s.Rand
	if source == nil {
		source = rand.New(rand.NewSource(1))
	}

	scaled := make([]float64, len(logits))
	maxVal := logits[0] / s.Temperature
	for i, logit := range logits {
		scaled[i] = logit / s.Temperature
		if scaled[i] > maxVal {
			maxVal = scaled[i]
		}
	}

	total := 0.0
	for i := range scaled {
		scaled[i] = math.Exp(scaled[i] - maxVal)
		total += scaled[i]
	}

	threshold := source.Float64()
	cumulative := 0.0
	for i, weight := range scaled {
		cumulative += weight / total
		if threshold <= cumulative {
			return i, nil
		}
	}
	return len(logits) - 1, nil
}
