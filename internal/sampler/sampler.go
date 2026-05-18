package sampler

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
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
		source = rand.New(rand.NewSource(time.Now().UnixNano()))
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

type TopKSampler struct {
	K           int
	Temperature float64
	Rand        *rand.Rand
}

func NewTopKSampler(k int, temperature float64, source *rand.Rand) *TopKSampler {
	return &TopKSampler{
		K:           k,
		Temperature: temperature,
		Rand:        source,
	}
}

func (s *TopKSampler) Sample(logits []float64) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("cannot sample from empty logits")
	}
	if s.K <= 0 {
		return 0, fmt.Errorf("top-k must be positive")
	}
	if s.Temperature <= 0 {
		return 0, fmt.Errorf("temperature must be positive")
	}

	source := s.Rand
	if source == nil {
		source = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	indices := topKIndices(logits, s.K)
	if len(indices) == 1 {
		return indices[0], nil
	}

	scaled := make([]float64, len(indices))
	maxVal := logits[indices[0]] / s.Temperature
	for i, idx := range indices {
		scaled[i] = logits[idx] / s.Temperature
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
			return indices[i], nil
		}
	}
	return indices[len(indices)-1], nil
}

func topKIndices(logits []float64, k int) []int {
	if k > len(logits) {
		k = len(logits)
	}
	indices := make([]int, len(logits))
	for i := range logits {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		left := logits[indices[i]]
		right := logits[indices[j]]
		if left == right {
			return indices[i] < indices[j]
		}
		return left > right
	})
	return indices[:k]
}
