package gpt2

import (
	"fmt"
	"math"

	sharedmodel "github.com/augahmed/aurelius/internal/model"
)

type Model struct {
	cfg               Config
	tokenEmbeddings   Tensor
	positionEmbedding Tensor
	blocks            []Block
	finalNorm         LayerNorm
	lmHead            Tensor
}

type Block struct {
	AttentionNorm LayerNorm
	Attention     Attention
	MLPNorm       LayerNorm
	MLP           MLP
}

type LayerNorm struct {
	Weight []float64
	Bias   []float64
}

type Attention struct {
	CombinedWeight Tensor
	CombinedBias   []float64
	ProjectWeight  Tensor
	ProjectBias    []float64
	NumHeads       int
}

type MLP struct {
	UpWeight   Tensor
	UpBias     []float64
	DownWeight Tensor
	DownBias   []float64
}

func LoadModel(configPath, weightsPath string) (*Model, error) {
	cfg, err := LoadConfig(configPath)
	if err != nil {
		return nil, err
	}

	state, err := LoadSafeTensors(weightsPath)
	if err != nil {
		return nil, err
	}

	return NewModel(cfg, state)
}

func NewModel(cfg Config, state map[string]Tensor) (*Model, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	model := &Model{
		cfg:    cfg,
		blocks: make([]Block, cfg.NumLayers),
	}

	tokenEmbeddings, err := requireTensorAlias(state, []string{"transformer.wte.weight", "wte.weight"}, cfg.VocabSize, cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	positionEmbeddings, err := requireTensorAlias(state, []string{"transformer.wpe.weight", "wpe.weight"}, cfg.ResolvedContextLength(), cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	model.tokenEmbeddings = tokenEmbeddings
	model.positionEmbedding = positionEmbeddings

	lmHead, ok := state["lm_head.weight"]
	if ok {
		if err := expectShape("lm_head.weight", lmHead, cfg.VocabSize, cfg.EmbeddingDim); err != nil {
			return nil, err
		}
		model.lmHead = lmHead
	} else {
		model.lmHead = tokenEmbeddings
	}

	finalNorm, err := requireLayerNormAlias(state, []string{"transformer.ln_f", "ln_f"}, cfg.EmbeddingDim)
	if err != nil {
		return nil, err
	}
	model.finalNorm = finalNorm

	feedForwardDim := cfg.ResolvedFeedForwardDim()
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("transformer.h.%d", i)
		hfPrefix := fmt.Sprintf("h.%d", i)

		attentionNorm, err := requireLayerNormAlias(state, []string{prefix + ".ln_1", hfPrefix + ".ln_1"}, cfg.EmbeddingDim)
		if err != nil {
			return nil, err
		}
		attentionCombinedWeight, err := requireTensorAlias(state, []string{prefix + ".attn.c_attn.weight", hfPrefix + ".attn.c_attn.weight"}, cfg.EmbeddingDim, cfg.EmbeddingDim*3)
		if err != nil {
			return nil, err
		}
		attentionCombinedBias, err := requireVectorAlias(state, []string{prefix + ".attn.c_attn.bias", hfPrefix + ".attn.c_attn.bias"}, cfg.EmbeddingDim*3)
		if err != nil {
			return nil, err
		}
		attentionProjectWeight, err := requireTensorAlias(state, []string{prefix + ".attn.c_proj.weight", hfPrefix + ".attn.c_proj.weight"}, cfg.EmbeddingDim, cfg.EmbeddingDim)
		if err != nil {
			return nil, err
		}
		attentionProjectBias, err := requireVectorAlias(state, []string{prefix + ".attn.c_proj.bias", hfPrefix + ".attn.c_proj.bias"}, cfg.EmbeddingDim)
		if err != nil {
			return nil, err
		}

		mlpNorm, err := requireLayerNormAlias(state, []string{prefix + ".ln_2", hfPrefix + ".ln_2"}, cfg.EmbeddingDim)
		if err != nil {
			return nil, err
		}
		mlpUpWeight, err := requireTensorAlias(state, []string{prefix + ".mlp.c_fc.weight", hfPrefix + ".mlp.c_fc.weight"}, cfg.EmbeddingDim, feedForwardDim)
		if err != nil {
			return nil, err
		}
		mlpUpBias, err := requireVectorAlias(state, []string{prefix + ".mlp.c_fc.bias", hfPrefix + ".mlp.c_fc.bias"}, feedForwardDim)
		if err != nil {
			return nil, err
		}
		mlpDownWeight, err := requireTensorAlias(state, []string{prefix + ".mlp.c_proj.weight", hfPrefix + ".mlp.c_proj.weight"}, feedForwardDim, cfg.EmbeddingDim)
		if err != nil {
			return nil, err
		}
		mlpDownBias, err := requireVectorAlias(state, []string{prefix + ".mlp.c_proj.bias", hfPrefix + ".mlp.c_proj.bias"}, cfg.EmbeddingDim)
		if err != nil {
			return nil, err
		}

		model.blocks[i] = Block{
			AttentionNorm: attentionNorm,
			Attention: Attention{
				CombinedWeight: attentionCombinedWeight,
				CombinedBias:   attentionCombinedBias,
				ProjectWeight:  attentionProjectWeight,
				ProjectBias:    attentionProjectBias,
				NumHeads:       cfg.NumHeads,
			},
			MLPNorm: mlpNorm,
			MLP: MLP{
				UpWeight:   mlpUpWeight,
				UpBias:     mlpUpBias,
				DownWeight: mlpDownWeight,
				DownBias:   mlpDownBias,
			},
		}
	}

	return model, nil
}

func (m *Model) Config() sharedmodel.Config {
	return m.cfg.ModelConfig()
}

func (m *Model) Forward(input []int, _ sharedmodel.Cache) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input token sequence cannot be empty")
	}
	if len(input) > m.cfg.ResolvedContextLength() {
		return nil, fmt.Errorf("input length %d exceeds context length %d", len(input), m.cfg.ResolvedContextLength())
	}

	hidden, err := m.embed(input)
	if err != nil {
		return nil, err
	}

	for _, block := range m.blocks {
		attentionInput := applyLayerNorm(hidden, len(input), m.cfg.EmbeddingDim, block.AttentionNorm, m.cfg.ResolvedLayerNormEpsilon())
		attentionOutput := block.Attention.Forward(attentionInput, len(input), m.cfg.EmbeddingDim)
		addInPlace(hidden, attentionOutput)

		mlpInput := applyLayerNorm(hidden, len(input), m.cfg.EmbeddingDim, block.MLPNorm, m.cfg.ResolvedLayerNormEpsilon())
		mlpOutput := block.MLP.Forward(mlpInput, len(input), m.cfg.EmbeddingDim)
		addInPlace(hidden, mlpOutput)
	}

	lastHidden := hidden[(len(input)-1)*m.cfg.EmbeddingDim : len(input)*m.cfg.EmbeddingDim]
	finalHidden := applyLayerNormRow(lastHidden, m.finalNorm, m.cfg.ResolvedLayerNormEpsilon())

	logits := make([]float64, m.cfg.VocabSize)
	for token := 0; token < m.cfg.VocabSize; token++ {
		sum := 0.0
		for dim := 0; dim < m.cfg.EmbeddingDim; dim++ {
			sum += finalHidden[dim] * m.lmHead.Data[token*m.cfg.EmbeddingDim+dim]
		}
		logits[token] = sum
	}

	return logits, nil
}

func (m *Model) embed(tokens []int) ([]float64, error) {
	hidden := make([]float64, len(tokens)*m.cfg.EmbeddingDim)
	for position, token := range tokens {
		if token < 0 || token >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token %d at position %d out of range", token, position)
		}
		for dim := 0; dim < m.cfg.EmbeddingDim; dim++ {
			hidden[position*m.cfg.EmbeddingDim+dim] =
				m.tokenEmbeddings.Data[token*m.cfg.EmbeddingDim+dim] +
					m.positionEmbedding.Data[position*m.cfg.EmbeddingDim+dim]
		}
	}
	return hidden, nil
}

func (a Attention) Forward(hidden []float64, seqLen, embDim int) []float64 {
	qkv := affineRows(hidden, seqLen, embDim, a.CombinedWeight, a.CombinedBias)
	queries := make([]float64, seqLen*embDim)
	keys := make([]float64, seqLen*embDim)
	values := make([]float64, seqLen*embDim)
	for row := 0; row < seqLen; row++ {
		offset := row * embDim * 3
		copy(queries[row*embDim:(row+1)*embDim], qkv[offset:offset+embDim])
		copy(keys[row*embDim:(row+1)*embDim], qkv[offset+embDim:offset+2*embDim])
		copy(values[row*embDim:(row+1)*embDim], qkv[offset+2*embDim:offset+3*embDim])
	}

	headDim := embDim / a.NumHeads
	context := make([]float64, seqLen*embDim)
	scale := math.Sqrt(float64(headDim))
	for row := 0; row < seqLen; row++ {
		for head := 0; head < a.NumHeads; head++ {
			headOffset := head * headDim
			scores := make([]float64, row+1)
			maxScore := math.Inf(-1)
			for keyRow := 0; keyRow <= row; keyRow++ {
				dot := 0.0
				for dim := 0; dim < headDim; dim++ {
					dot += queries[row*embDim+headOffset+dim] * keys[keyRow*embDim+headOffset+dim]
				}
				score := dot / scale
				scores[keyRow] = score
				if score > maxScore {
					maxScore = score
				}
			}

			weightSum := 0.0
			for i, score := range scores {
				weight := math.Exp(score - maxScore)
				scores[i] = weight
				weightSum += weight
			}

			for keyRow, weight := range scores {
				normalized := weight / weightSum
				for dim := 0; dim < headDim; dim++ {
					context[row*embDim+headOffset+dim] += normalized * values[keyRow*embDim+headOffset+dim]
				}
			}
		}
	}

	return affineRows(context, seqLen, embDim, a.ProjectWeight, a.ProjectBias)
}

func (m MLP) Forward(hidden []float64, seqLen, embDim int) []float64 {
	up := affineRows(hidden, seqLen, embDim, m.UpWeight, m.UpBias)
	for i, value := range up {
		up[i] = gelu(value)
	}
	return affineRows(up, seqLen, len(m.UpBias), m.DownWeight, m.DownBias)
}

func applyLayerNorm(hidden []float64, rows, cols int, norm LayerNorm, epsilon float64) []float64 {
	out := make([]float64, len(hidden))
	for row := 0; row < rows; row++ {
		copy(out[row*cols:(row+1)*cols], applyLayerNormRow(hidden[row*cols:(row+1)*cols], norm, epsilon))
	}
	return out
}

func applyLayerNormRow(row []float64, norm LayerNorm, epsilon float64) []float64 {
	mean := 0.0
	for _, value := range row {
		mean += value
	}
	mean /= float64(len(row))

	variance := 0.0
	for _, value := range row {
		diff := value - mean
		variance += diff * diff
	}
	variance /= float64(len(row))

	denom := math.Sqrt(variance + epsilon)
	out := make([]float64, len(row))
	for i, value := range row {
		out[i] = ((value - mean) / denom * norm.Weight[i]) + norm.Bias[i]
	}
	return out
}

func affineRows(hidden []float64, rows, inDim int, weight Tensor, bias []float64) []float64 {
	outDim := len(bias)
	out := make([]float64, rows*outDim)
	for row := 0; row < rows; row++ {
		inputOffset := row * inDim
		outputOffset := row * outDim
		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := bias[outIndex]
			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += hidden[inputOffset+inIndex] * weight.Data[inIndex*outDim+outIndex]
			}
			out[outputOffset+outIndex] = sum
		}
	}
	return out
}

func addInPlace(dst, src []float64) {
	for i, value := range src {
		dst[i] += value
	}
}

func gelu(value float64) float64 {
	const coeff = 0.044715
	return 0.5 * value * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(value+coeff*value*value*value)))
}

func requireLayerNorm(state map[string]Tensor, prefix string, width int) (LayerNorm, error) {
	return requireLayerNormAlias(state, []string{prefix}, width)
}

func requireLayerNormAlias(state map[string]Tensor, prefixes []string, width int) (LayerNorm, error) {
	weightNames := make([]string, 0, len(prefixes))
	biasNames := make([]string, 0, len(prefixes))
	for _, prefix := range prefixes {
		weightNames = append(weightNames, prefix+".weight")
		biasNames = append(biasNames, prefix+".bias")
	}
	weight, err := requireVectorAlias(state, weightNames, width)
	if err != nil {
		return LayerNorm{}, err
	}
	bias, err := requireVectorAlias(state, biasNames, width)
	if err != nil {
		return LayerNorm{}, err
	}
	return LayerNorm{Weight: weight, Bias: bias}, nil
}

func requireTensor(state map[string]Tensor, name string, shape ...int) (Tensor, error) {
	return requireTensorAlias(state, []string{name}, shape...)
}

func requireTensorAlias(state map[string]Tensor, names []string, shape ...int) (Tensor, error) {
	for _, name := range names {
		value, ok := state[name]
		if !ok {
			continue
		}
		if err := expectShape(name, value, shape...); err != nil {
			return Tensor{}, err
		}
		return value, nil
	}
	return Tensor{}, fmt.Errorf("missing tensor %q", names[0])
}

func requireVectorAlias(state map[string]Tensor, names []string, length int) ([]float64, error) {
	value, err := requireTensorAlias(state, names, length)
	if err != nil {
		return nil, err
	}
	return append([]float64(nil), value.Data...), nil
}

func requireVector(state map[string]Tensor, name string, length int) ([]float64, error) {
	value, ok := state[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	value, err := requireTensor(state, name, length)
	if err != nil {
		return nil, err
	}
	return append([]float64(nil), value.Data...), nil
}

func expectShape(name string, tensor Tensor, shape ...int) error {
	if len(tensor.Shape) != len(shape) {
		return fmt.Errorf("tensor %q shape %v does not match expected %v", name, tensor.Shape, shape)
	}
	for i := range shape {
		if tensor.Shape[i] != shape[i] {
			return fmt.Errorf("tensor %q shape %v does not match expected %v", name, tensor.Shape, shape)
		}
	}
	return nil
}
