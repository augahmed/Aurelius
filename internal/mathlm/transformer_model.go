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
	NumLayers    int   `json:"num_layers,omitempty"`
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
	if c.NumLayers < 0 {
		return fmt.Errorf("num layers cannot be negative")
	}
	if c.EmbeddingDim%c.NumHeads != 0 {
		return fmt.Errorf("embedding dim %d must be divisible by num heads %d", c.EmbeddingDim, c.NumHeads)
	}
	if c.MLPDim <= 0 {
		return fmt.Errorf("mlp dim must be positive")
	}
	return nil
}

type TransformerBlock struct {
	LN1Gamma         []float64 `json:"ln1_gamma"`
	LN1Beta          []float64 `json:"ln1_beta"`
	QueryWeights     []float64 `json:"query_weights"`
	KeyWeights       []float64 `json:"key_weights"`
	ValueWeights     []float64 `json:"value_weights"`
	AttentionWeights []float64 `json:"attention_weights"`
	LN2Gamma         []float64 `json:"ln2_gamma"`
	LN2Beta          []float64 `json:"ln2_beta"`
	MLPInputWeights  []float64 `json:"mlp_input_weights"`
	MLPInputBias     []float64 `json:"mlp_input_bias"`
	MLPOutputWeights []float64 `json:"mlp_output_weights"`
	MLPOutputBias    []float64 `json:"mlp_output_bias"`
}

type TransformerModel struct {
	LMConfig           TransformerConfig  `json:"config"`
	TokenEmbeddings    []float64          `json:"token_embeddings"`
	PositionEmbeddings []float64          `json:"position_embeddings"`
	Layers             []TransformerBlock `json:"layers,omitempty"`
	LN1Gamma           []float64          `json:"ln1_gamma"`
	LN1Beta            []float64          `json:"ln1_beta"`
	QueryWeights       []float64          `json:"query_weights"`
	KeyWeights         []float64          `json:"key_weights"`
	ValueWeights       []float64          `json:"value_weights"`
	AttentionWeights   []float64          `json:"attention_weights"`
	LN2Gamma           []float64          `json:"ln2_gamma"`
	LN2Beta            []float64          `json:"ln2_beta"`
	MLPInputWeights    []float64          `json:"mlp_input_weights"`
	MLPInputBias       []float64          `json:"mlp_input_bias"`
	MLPOutputWeights   []float64          `json:"mlp_output_weights"`
	MLPOutputBias      []float64          `json:"mlp_output_bias"`
	FinalLNGamma       []float64          `json:"final_ln_gamma,omitempty"`
	FinalLNBeta        []float64          `json:"final_ln_beta,omitempty"`
	OutputWeights      []float64          `json:"output_weights"`
	OutputBias         []float64          `json:"output_bias"`
}

type transformerForwardCache struct {
	Context    []int
	Embeddings [][]float64
	Layers     []transformerBlockForwardCache
}

type transformerBlockForwardCache struct {
	X         [][]float64
	Norm1     [][]float64
	Q         [][]float64
	K         [][]float64
	V         [][]float64
	AttnProbs [][][]float64
	Attended  [][]float64
	Residual1 [][]float64
	Norm2     [][]float64
	MLPPre    [][]float64
	MLPHidden [][]float64
	MLPOut    [][]float64
	States    [][]float64
}

func NewTransformerModel(cfg TransformerConfig) (*TransformerModel, error) {
	cfg.NumLayers = transformerLayerCount(cfg)
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	model := &TransformerModel{
		LMConfig:           cfg,
		TokenEmbeddings:    make([]float64, cfg.VocabSize*cfg.EmbeddingDim),
		PositionEmbeddings: make([]float64, cfg.ContextSize*cfg.EmbeddingDim),
		Layers:             make([]TransformerBlock, cfg.NumLayers),
		FinalLNGamma:       ones(cfg.EmbeddingDim),
		FinalLNBeta:        make([]float64, cfg.EmbeddingDim),
		OutputWeights:      make([]float64, cfg.EmbeddingDim*cfg.VocabSize),
		OutputBias:         make([]float64, cfg.VocabSize),
	}
	initWeights(model.TokenEmbeddings, rng, 0.04)
	initWeights(model.PositionEmbeddings, rng, 0.04)
	for layer := range model.Layers {
		model.Layers[layer] = newTransformerBlock(cfg, rng)
	}
	initWeights(model.OutputWeights, rng, 0.04)
	model.syncLegacyBlockFields()
	return model, nil
}

func (m *TransformerModel) Config() sharedmodel.Config {
	return sharedmodel.Config{
		VocabSize:     m.LMConfig.VocabSize,
		ContextLength: m.LMConfig.ContextSize,
		EmbeddingDim:  m.LMConfig.EmbeddingDim,
		NumLayers:     transformerLayerCount(m.LMConfig),
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
	states, _, err := m.forwardContextWithCache(context)
	return states, err
}

func (m *TransformerModel) forwardContextWithCache(context []int) ([][]float64, *transformerForwardCache, error) {
	m.ensureLayerBlocks()
	if len(context) != m.LMConfig.ContextSize {
		return nil, nil, fmt.Errorf("context length = %d, want %d", len(context), m.LMConfig.ContextSize)
	}
	cfg := m.LMConfig
	x := make([][]float64, cfg.ContextSize)
	for pos, token := range context {
		if token < 0 || token >= cfg.VocabSize {
			return nil, nil, fmt.Errorf("token %d out of range", token)
		}
		x[pos] = make([]float64, cfg.EmbeddingDim)
		tokenOffset := token * cfg.EmbeddingDim
		posOffset := pos * cfg.EmbeddingDim
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			x[pos][dim] = m.TokenEmbeddings[tokenOffset+dim] + m.PositionEmbeddings[posOffset+dim]
		}
	}

	states := x
	layerCaches := make([]transformerBlockForwardCache, len(m.Layers))
	for layerIndex := range m.Layers {
		var layerCache transformerBlockForwardCache
		states, layerCache = forwardTransformerBlock(cfg, &m.Layers[layerIndex], states)
		layerCaches[layerIndex] = layerCache
	}
	cache := &transformerForwardCache{
		Context:    append([]int(nil), context...),
		Embeddings: x,
		Layers:     layerCaches,
	}
	return states, cache, nil
}

func forwardTransformerBlock(cfg TransformerConfig, block *TransformerBlock, x [][]float64) ([][]float64, transformerBlockForwardCache) {
	norm1 := make([][]float64, cfg.ContextSize)
	q := make([][]float64, cfg.ContextSize)
	k := make([][]float64, cfg.ContextSize)
	v := make([][]float64, cfg.ContextSize)
	for pos := range x {
		norm1[pos] = layerNorm(x[pos], block.LN1Gamma, block.LN1Beta)
		q[pos] = linear(norm1[pos], block.QueryWeights, cfg.EmbeddingDim)
		k[pos] = linear(norm1[pos], block.KeyWeights, cfg.EmbeddingDim)
		v[pos] = linear(norm1[pos], block.ValueWeights, cfg.EmbeddingDim)
	}

	headDim := cfg.EmbeddingDim / cfg.NumHeads
	scale := 1 / math.Sqrt(float64(headDim))
	attended := make([][]float64, cfg.ContextSize)
	attnProbs := make([][][]float64, cfg.ContextSize)
	for pos := 0; pos < cfg.ContextSize; pos++ {
		attended[pos] = make([]float64, cfg.EmbeddingDim)
		attnProbs[pos] = make([][]float64, cfg.NumHeads)
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
			attnProbs[pos][head] = probs
			for src, prob := range probs {
				for dim := 0; dim < headDim; dim++ {
					attended[pos][headStart+dim] += prob * v[src][headStart+dim]
				}
			}
		}
	}

	residual1 := make([][]float64, cfg.ContextSize)
	for pos := range attended {
		proj := linear(attended[pos], block.AttentionWeights, cfg.EmbeddingDim)
		residual1[pos] = make([]float64, cfg.EmbeddingDim)
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			residual1[pos][dim] = x[pos][dim] + proj[dim]
		}
	}

	states := make([][]float64, cfg.ContextSize)
	norm2Values := make([][]float64, cfg.ContextSize)
	mlpPreValues := make([][]float64, cfg.ContextSize)
	mlpHiddenValues := make([][]float64, cfg.ContextSize)
	mlpOutValues := make([][]float64, cfg.ContextSize)
	for pos := range residual1 {
		norm2 := layerNorm(residual1[pos], block.LN2Gamma, block.LN2Beta)
		hiddenPre := linearWithBias(norm2, block.MLPInputWeights, block.MLPInputBias, cfg.MLPDim)
		hidden := make([]float64, len(hiddenPre))
		for i := range hiddenPre {
			hidden[i] = gelu(hiddenPre[i])
		}
		mlpOut := linearWithBias(hidden, block.MLPOutputWeights, block.MLPOutputBias, cfg.EmbeddingDim)
		norm2Values[pos] = norm2
		mlpPreValues[pos] = hiddenPre
		mlpHiddenValues[pos] = hidden
		mlpOutValues[pos] = mlpOut
		states[pos] = make([]float64, cfg.EmbeddingDim)
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			states[pos][dim] = residual1[pos][dim] + mlpOut[dim]
		}
	}
	cache := transformerBlockForwardCache{
		X:         x,
		Norm1:     norm1,
		Q:         q,
		K:         k,
		V:         v,
		AttnProbs: attnProbs,
		Attended:  attended,
		Residual1: residual1,
		Norm2:     norm2Values,
		MLPPre:    mlpPreValues,
		MLPHidden: mlpHiddenValues,
		MLPOut:    mlpOutValues,
		States:    states,
	}
	return states, cache
}

func (m *TransformerModel) logitsForState(state []float64) []float64 {
	m.ensureFinalLayerNorm()
	normalized := layerNorm(state, m.FinalLNGamma, m.FinalLNBeta)
	return linearWithBias(normalized, m.OutputWeights, m.OutputBias, m.LMConfig.VocabSize)
}

func transformerLayerCount(cfg TransformerConfig) int {
	if cfg.NumLayers <= 0 {
		return 1
	}
	return cfg.NumLayers
}

func newTransformerBlock(cfg TransformerConfig, rng *rand.Rand) TransformerBlock {
	block := TransformerBlock{
		LN1Gamma:         ones(cfg.EmbeddingDim),
		LN1Beta:          make([]float64, cfg.EmbeddingDim),
		QueryWeights:     make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		KeyWeights:       make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		ValueWeights:     make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		AttentionWeights: make([]float64, cfg.EmbeddingDim*cfg.EmbeddingDim),
		LN2Gamma:         ones(cfg.EmbeddingDim),
		LN2Beta:          make([]float64, cfg.EmbeddingDim),
		MLPInputWeights:  make([]float64, cfg.EmbeddingDim*cfg.MLPDim),
		MLPInputBias:     make([]float64, cfg.MLPDim),
		MLPOutputWeights: make([]float64, cfg.MLPDim*cfg.EmbeddingDim),
		MLPOutputBias:    make([]float64, cfg.EmbeddingDim),
	}
	initWeights(block.QueryWeights, rng, 0.04)
	initWeights(block.KeyWeights, rng, 0.04)
	initWeights(block.ValueWeights, rng, 0.04)
	initWeights(block.AttentionWeights, rng, 0.04)
	initWeights(block.MLPInputWeights, rng, 0.04)
	initWeights(block.MLPOutputWeights, rng, 0.04)
	return block
}

func (m *TransformerModel) ensureLayerBlocks() {
	m.LMConfig.NumLayers = transformerLayerCount(m.LMConfig)
	m.ensureFinalLayerNorm()
	if len(m.Layers) == 0 {
		m.Layers = []TransformerBlock{{
			LN1Gamma:         m.LN1Gamma,
			LN1Beta:          m.LN1Beta,
			QueryWeights:     m.QueryWeights,
			KeyWeights:       m.KeyWeights,
			ValueWeights:     m.ValueWeights,
			AttentionWeights: m.AttentionWeights,
			LN2Gamma:         m.LN2Gamma,
			LN2Beta:          m.LN2Beta,
			MLPInputWeights:  m.MLPInputWeights,
			MLPInputBias:     m.MLPInputBias,
			MLPOutputWeights: m.MLPOutputWeights,
			MLPOutputBias:    m.MLPOutputBias,
		}}
	}
	m.syncLegacyBlockFields()
}

func (m *TransformerModel) ensureFinalLayerNorm() {
	if len(m.FinalLNGamma) != m.LMConfig.EmbeddingDim {
		m.FinalLNGamma = ones(m.LMConfig.EmbeddingDim)
	}
	if len(m.FinalLNBeta) != m.LMConfig.EmbeddingDim {
		m.FinalLNBeta = make([]float64, m.LMConfig.EmbeddingDim)
	}
}

func (m *TransformerModel) syncLegacyBlockFields() {
	if len(m.Layers) == 0 {
		return
	}
	first := &m.Layers[0]
	m.LN1Gamma = first.LN1Gamma
	m.LN1Beta = first.LN1Beta
	m.QueryWeights = first.QueryWeights
	m.KeyWeights = first.KeyWeights
	m.ValueWeights = first.ValueWeights
	m.AttentionWeights = first.AttentionWeights
	m.LN2Gamma = first.LN2Gamma
	m.LN2Beta = first.LN2Beta
	m.MLPInputWeights = first.MLPInputWeights
	m.MLPInputBias = first.MLPInputBias
	m.MLPOutputWeights = first.MLPOutputWeights
	m.MLPOutputBias = first.MLPOutputBias
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

func geluDerivative(value float64) float64 {
	inner := math.Sqrt(2/math.Pi) * (value + 0.044715*value*value*value)
	tanhInner := math.Tanh(inner)
	innerDerivative := math.Sqrt(2/math.Pi) * (1 + 3*0.044715*value*value)
	return 0.5*(1+tanhInner) + 0.5*value*(1-tanhInner*tanhInner)*innerDerivative
}
