package mathlm

import (
	"fmt"
	"math/rand"

	sharedmodel "github.com/augahmed/aurelius/internal/model"
)

type Config struct {
	VocabSize    int   `json:"vocab_size"`
	ContextSize  int   `json:"context_size"`
	EmbeddingDim int   `json:"embedding_dim"`
	HiddenDim    int   `json:"hidden_dim"`
	Seed         int64 `json:"seed"`
}

func (c Config) Validate() error {
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}
	if c.ContextSize <= 0 {
		return fmt.Errorf("context size must be positive")
	}
	if c.EmbeddingDim <= 0 {
		return fmt.Errorf("embedding dim must be positive")
	}
	if c.HiddenDim <= 0 {
		return fmt.Errorf("hidden dim must be positive")
	}
	return nil
}

type Model struct {
	LMConfig      Config    `json:"config"`
	Embeddings    []float64 `json:"embeddings"`
	HiddenWeights []float64 `json:"hidden_weights"`
	HiddenBias    []float64 `json:"hidden_bias"`
	OutputWeights []float64 `json:"output_weights"`
	OutputBias    []float64 `json:"output_bias"`
}

func NewModel(cfg Config) (*Model, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	model := &Model{
		LMConfig:      cfg,
		Embeddings:    make([]float64, cfg.VocabSize*cfg.EmbeddingDim),
		HiddenWeights: make([]float64, cfg.ContextSize*cfg.EmbeddingDim*cfg.HiddenDim),
		HiddenBias:    make([]float64, cfg.HiddenDim),
		OutputWeights: make([]float64, cfg.HiddenDim*cfg.VocabSize),
		OutputBias:    make([]float64, cfg.VocabSize),
	}
	initWeights(model.Embeddings, rng, 0.05)
	initWeights(model.HiddenWeights, rng, 0.05)
	initWeights(model.OutputWeights, rng, 0.05)
	return model, nil
}

func (m *Model) ModelConfig() Config {
	return m.LMConfig
}

func (m *Model) Config() sharedmodel.Config {
	return sharedmodel.Config{
		VocabSize:     m.LMConfig.VocabSize,
		ContextLength: m.LMConfig.ContextSize,
		EmbeddingDim:  m.LMConfig.EmbeddingDim,
		NumLayers:     1,
		NumHeads:      1,
	}
}

func (m *Model) Forward(input []int, _ sharedmodel.Cache) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	context, err := m.contextFromInput(input)
	if err != nil {
		return nil, err
	}
	_, _, logits, err := m.forwardContext(context)
	if err != nil {
		return nil, err
	}
	return logits, nil
}

func (m *Model) contextFromInput(input []int) ([]int, error) {
	context := make([]int, m.LMConfig.ContextSize)
	start := max(0, len(input)-m.LMConfig.ContextSize)
	window := input[start:]
	for _, token := range window {
		if token < 0 || token >= m.LMConfig.VocabSize {
			return nil, fmt.Errorf("token %d out of range", token)
		}
	}
	copy(context[m.LMConfig.ContextSize-len(window):], window)
	return context, nil
}

func (m *Model) forwardContext(context []int) ([]float64, []float64, []float64, error) {
	if len(context) != m.LMConfig.ContextSize {
		return nil, nil, nil, fmt.Errorf("context length = %d, want %d", len(context), m.LMConfig.ContextSize)
	}
	for _, token := range context {
		if token < 0 || token >= m.LMConfig.VocabSize {
			return nil, nil, nil, fmt.Errorf("token %d out of range", token)
		}
	}

	inputVector := make([]float64, m.LMConfig.ContextSize*m.LMConfig.EmbeddingDim)
	for pos, token := range context {
		src := token * m.LMConfig.EmbeddingDim
		dst := pos * m.LMConfig.EmbeddingDim
		copy(inputVector[dst:dst+m.LMConfig.EmbeddingDim], m.Embeddings[src:src+m.LMConfig.EmbeddingDim])
	}

	hidden := make([]float64, m.LMConfig.HiddenDim)
	for hiddenIndex := 0; hiddenIndex < m.LMConfig.HiddenDim; hiddenIndex++ {
		sum := m.HiddenBias[hiddenIndex]
		for inputIndex, value := range inputVector {
			sum += value * m.HiddenWeights[inputIndex*m.LMConfig.HiddenDim+hiddenIndex]
		}
		hidden[hiddenIndex] = tanh(sum)
	}

	logits := make([]float64, m.LMConfig.VocabSize)
	for token := 0; token < m.LMConfig.VocabSize; token++ {
		sum := m.OutputBias[token]
		for hiddenIndex, value := range hidden {
			sum += value * m.OutputWeights[hiddenIndex*m.LMConfig.VocabSize+token]
		}
		logits[token] = sum
	}
	return inputVector, hidden, logits, nil
}

func initWeights(weights []float64, rng *rand.Rand, scale float64) {
	for i := range weights {
		weights[i] = (rng.Float64()*2 - 1) * scale
	}
}
