package mathlm

import (
	"fmt"
	"math"
	"math/rand"

	sharedmodel "github.com/augahmed/aurelius/internal/model"
)

type TransformerConfig struct {
	VocabSize    int   `json:"vocab_size"`
	ContextSize  int   `json:"context_size"`
	EmbeddingDim int   `json:"embedding_dim"`
	NumHeads     int   `json:"num_heads"`
	MLPDim       int   `json:"mlp_dim"`
	Seed         int64 `json:"seed"`
}

func (c TransformerConfig) Validate() error {
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}
	if c.ContextSize <= 0 {
		return fmt.Errorf("context size must be positive")
	}
	if c.EmbeddingDim <= 0 {
		return fmt.Errorf("embedding dim must be positive")
	}
	if c.NumHeads <= 0 {
		return fmt.Errorf("num heads must be positive")
	}
	if c.EmbeddingDim%c.NumHeads != 0 {
		return fmt.Errorf("embedding dim %d must be divisible by num heads %d", c.EmbeddingDim, c.NumHeads)
	}
	if c.MLPDim <= 0 {
		return fmt.Errorf("mlp dim must be positive")
	}
	return nil
}

type TransformerModel struct {
	LMConfig           TransformerConfig `json:"config"`
	TokenEmbeddings    []float64         `json:"token_embeddings"`
	PositionEmbeddings []float64         `json:"position_embeddings"`
	LN1Gamma           []float64         `json:"ln1_gamma"`
	LN1Beta            []float64         `json:"ln1_beta"`
	QueryWeights       []float64         `json:"query_weights"`
	KeyWeights         []float64         `json:"key_weights"`
	ValueWeights       []float64         `json:"value_weights"`
	AttentionWeights   []float64         `json:"attention_weights"`
	LN2Gamma           []float64         `json:"ln2_gamma"`
	LN2Beta            []float64         `json:"ln2_beta"`
	MLPInputWeights    []float64         `json:"mlp_input_weights"`
	MLPInputBias       []float64         `json:"mlp_input_bias"`
	MLPOutputWeights   []float64         `json:"mlp_output_weights"`
	MLPOutputBias      []float64         `json:"mlp_output_bias"`
	OutputWeights      []float64         `json:"output_weights"`
	OutputBias         []float64         `json:"output_bias"`
}

func NewTransformerModel(cfg TransformerConfig) (*TransformerModel, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	model := &TransformerModel{
		LMConfig:           cfg,
		TokenEmbeddings:    make([]float64, cfg.VocabSize*cfg.EmbeddingDim),
		PositionEmbeddings: make([]float64, cfg.ContextSize*cfg.EmbeddingDim),
		LN1Gamma:           ones(cfg.EmbeddingDim),
		LN1Beta:            make([]float64, cfg.EmbeddingDim),
		QueryWeights:       make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		KeyWeights:         make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		ValueWeights:       make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		AttentionWeights:   make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		LN2Gamma:           ones(cfg.EmbeddingDim),
		LN2Beta:            make([]float64, cfg.EmbeddingDim),
		MLPInputWeights:    make([]float64, cfg.EmbeddingDim*cfg.MLPDim),
		MLPInputBias:       make([]float64, cfg.MLPDim),
		MLPOutputWeights:   make([]float64, cfg.MLPDim*cfg.EmbeddingDim),
		MLPOutputBias:      make([]float64, cfg.EmbeddingDim),
		OutputWeights:      make([]float64, cfg.EmbeddingDim*cfg.VocabSize),
		OutputBias:         make([]float64, cfg.VocabSize),
	}
	initWeights(model.TokenEmbeddings, rng, 0.04)
	initWeights(model.PositionEmbeddings, rng, 0.04)
	initWeights(model.QueryWeights, rng, 0.04)
	initWeights(model.KeyWeights, rng, 0.04)
	initWeights(model.ValueWeights, rng, 0.04)
	initWeights(model.AttentionWeights, rng, 0.04)
	initWeights(model.MLPInputWeights, rng, 0.04)
	initWeights(model.MLPOutputWeights, rng, 0.04)
	initWeights(model.OutputWeights, rng, 0.04)
	return model, nil
}

func (m *TransformerModel) Config() sharedmodel.Config {
	return sharedmodel.Config{
		VocabSize:     m.LMConfig.VocabSize,
		ContextLength: m.LMConfig.ContextSize,
		EmbeddingDim:  m.LMConfig.EmbeddingDim,
		NumLayers:     1,
		NumHeads:      m.LMConfig.NumHeads,
	}
}

func (m *TransformerModel) Forward(input []int, _ sharedmodel.Cache) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	context, err := m.contextFromInput(input)
	if err != nil {
		return nil, err
	}
	states, err := m.forwardContext(context)
	if err != nil {
		return nil, err
	}
	return m.logitsForState(states[len(states)-1]), nil
}

func (m *TransformerModel) ForwardAll(input []int) ([][]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	context, err := m.contextFromInput(input)
	if err != nil {
		return nil, err
	}
	states, err := m.forwardContext(context)
	if err != nil {
		return nil, err
	}
	logits := make([][]float64, len(states))
	for pos, state := range states {
		logits[pos] = m.logitsForState(state)
	}
	return logits, nil
}

func (m *TransformerModel) contextFromInput(input []int) ([]int, error) {
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

func (m *TransformerModel) forwardContext(context []int) ([][]float64, error) {
	if len(context) != m.LMConfig.ContextSize {
		return nil, fmt.Errorf("context length = %d, want %d", len(context), m.LMConfig.ContextSize)
	}
	cfg := m.LMConfig
	x := make([][]float64, cfg.ContextSize)
	for pos, token := range context {
		if token < 0 || token >= cfg.VocabSize {
			return nil, fmt.Errorf("token %d out of range", token)
		}
		x[pos] = make([]float64, cfg.EmbeddingDim)
		tokenOffset := token * cfg.EmbeddingDim
		posOffset := pos * cfg.EmbeddingDim
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			x[pos][dim] = m.TokenEmbeddings[tokenOffset+dim] + m.PositionEmbeddings[posOffset+dim]
		}
	}

	norm1 := make([][]float64, cfg.ContextSize)
	q := make([][]float64, cfg.ContextSize)
	k := make([][]float64, cfg.ContextSize)
	v := make([][]float64, cfg.ContextSize)
	for pos := range x {
		norm1[pos] = layerNorm(x[pos], m.LN1Gamma, m.LN1Beta)
		q[pos] = linear(norm1[pos], m.QueryWeights, cfg.EmbeddingDim)
		k[pos] = linear(norm1[pos], m.KeyWeights, cfg.EmbeddingDim)
		v[pos] = linear(norm1[pos], m.ValueWeights, cfg.EmbeddingDim)
	}

	headDim := cfg.EmbeddingDim / cfg.NumHeads
	scale := 1 / math.Sqrt(float64(headDim))
	attended := make([][]float64, cfg.ContextSize)
	for pos := 0; pos < cfg.ContextSize; pos++ {
		attended[pos] = make([]float64, cfg.EmbeddingDim)
		for head := 0; head < cfg.NumHeads; head++ {
			headStart := head * headDim
			scores := make([]float64, pos+1)
			for src := 0; src <= pos; src++ {
				dot := 0.0
				for dim := 0; dim < headDim; dim++ {
					dot += q[pos][headStart+dim] * k[src][headStart+dim]
				}
				scores[src] = dot * scale
			}
			probs := softmax(scores)
			for src, prob := range probs {
				for dim := 0; dim < headDim; dim++ {
					attended[pos][headStart+dim] += prob * v[src][headStart+dim]
				}
			}
		}
	}

	residual1 := make([][]float64, cfg.ContextSize)
	for pos := range attended {
		proj := linear(attended[pos], m.AttentionWeights, cfg.EmbeddingDim)
		residual1[pos] = make([]float64, cfg.EmbeddingDim)
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			residual1[pos][dim] = x[pos][dim] + proj[dim]
		}
	}

	states := make([][]float64, cfg.ContextSize)
	for pos := range residual1 {
		norm2 := layerNorm(residual1[pos], m.LN2Gamma, m.LN2Beta)
		hidden := linearWithBias(norm2, m.MLPInputWeights, m.MLPInputBias, cfg.MLPDim)
		for i := range hidden {
			hidden[i] = gelu(hidden[i])
		}
		mlpOut := linearWithBias(hidden, m.MLPOutputWeights, m.MLPOutputBias, cfg.EmbeddingDim)
		states[pos] = make([]float64, cfg.EmbeddingDim)
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			states[pos][dim] = residual1[pos][dim] + mlpOut[dim]
		}
	}
	return states, nil
}

func (m *TransformerModel) logitsForState(state []float64) []float64 {
	return linearWithBias(state, m.OutputWeights, m.OutputBias, m.LMConfig.VocabSize)
}

func ones(size int) []float64 {
	values := make([]float64, size)
	for i := range values {
		values[i] = 1
	}
	return values
}

func linear(input, weights []float64, outDim int) []float64 {
	output := make([]float64, outDim)
	for inIndex, value := range input {
		rowOffset := inIndex * outDim
		for outIndex := 0; outIndex < outDim; outIndex++ {
			output[outIndex] += value * weights[rowOffset+outIndex]
		}
	}
	return output
}

func linearWithBias(input, weights, bias []float64, outDim int) []float64 {
	output := linear(input, weights, outDim)
	for i := range output {
		output[i] += bias[i]
	}
	return output
}

func layerNorm(input, gamma, beta []float64) []float64 {
	mean := 0.0
	for _, value := range input {
		mean += value
	}
	mean /= float64(len(input))
	variance := 0.0
	for _, value := range input {
		diff := value - mean
		variance += diff * diff
	}
	variance /= float64(len(input))
	scale := 1 / math.Sqrt(variance+1e-5)
	output := make([]float64, len(input))
	for i, value := range input {
		output[i] = (value-mean)*scale*gamma[i] + beta[i]
	}
	return output
}

func gelu(value float64) float64 {
	return 0.5 * value * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(value+0.044715*value*value*value)))
}
