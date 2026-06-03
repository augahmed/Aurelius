package mathlm

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

type TransformerTrainer struct {
	Model *TransformerModel          `json:"model"`
	Adam  *TransformerOptimizerState `json:"adam"`
	Step  int                        `json:"step"`
}

type TransformerOptimizerState struct {
	Layers              []TransformerBlockOptimizerState `json:"layers,omitempty"`
	TokenEmbeddingsM    []float64                        `json:"token_embeddings_m"`
	TokenEmbeddingsV    []float64                        `json:"token_embeddings_v"`
	PositionEmbeddingsM []float64                        `json:"position_embeddings_m"`
	PositionEmbeddingsV []float64                        `json:"position_embeddings_v"`
	LN1GammaM           []float64                        `json:"ln1_gamma_m"`
	LN1GammaV           []float64                        `json:"ln1_gamma_v"`
	LN1BetaM            []float64                        `json:"ln1_beta_m"`
	LN1BetaV            []float64                        `json:"ln1_beta_v"`
	QueryWeightsM       []float64                        `json:"query_weights_m"`
	QueryWeightsV       []float64                        `json:"query_weights_v"`
	KeyWeightsM         []float64                        `json:"key_weights_m"`
	KeyWeightsV         []float64                        `json:"key_weights_v"`
	ValueWeightsM       []float64                        `json:"value_weights_m"`
	ValueWeightsV       []float64                        `json:"value_weights_v"`
	AttentionWeightsM   []float64                        `json:"attention_weights_m"`
	AttentionWeightsV   []float64                        `json:"attention_weights_v"`
	LN2GammaM           []float64                        `json:"ln2_gamma_m"`
	LN2GammaV           []float64                        `json:"ln2_gamma_v"`
	LN2BetaM            []float64                        `json:"ln2_beta_m"`
	LN2BetaV            []float64                        `json:"ln2_beta_v"`
	MLPInputWeightsM    []float64                        `json:"mlp_input_weights_m"`
	MLPInputWeightsV    []float64                        `json:"mlp_input_weights_v"`
	MLPInputBiasM       []float64                        `json:"mlp_input_bias_m"`
	MLPInputBiasV       []float64                        `json:"mlp_input_bias_v"`
	MLPOutputWeightsM   []float64                        `json:"mlp_output_weights_m"`
	MLPOutputWeightsV   []float64                        `json:"mlp_output_weights_v"`
	MLPOutputBiasM      []float64                        `json:"mlp_output_bias_m"`
	MLPOutputBiasV      []float64                        `json:"mlp_output_bias_v"`
	FinalLNGammaM       []float64                        `json:"final_ln_gamma_m,omitempty"`
	FinalLNGammaV       []float64                        `json:"final_ln_gamma_v,omitempty"`
	FinalLNBetaM        []float64                        `json:"final_ln_beta_m,omitempty"`
	FinalLNBetaV        []float64                        `json:"final_ln_beta_v,omitempty"`
	OutputWeightsM      []float64                        `json:"output_weights_m"`
	OutputWeightsV      []float64                        `json:"output_weights_v"`
	OutputBiasM         []float64                        `json:"output_bias_m"`
	OutputBiasV         []float64                        `json:"output_bias_v"`
}

type TransformerBlockOptimizerState struct {
	LN1GammaM         []float64 `json:"ln1_gamma_m"`
	LN1GammaV         []float64 `json:"ln1_gamma_v"`
	LN1BetaM          []float64 `json:"ln1_beta_m"`
	LN1BetaV          []float64 `json:"ln1_beta_v"`
	QueryWeightsM     []float64 `json:"query_weights_m"`
	QueryWeightsV     []float64 `json:"query_weights_v"`
	KeyWeightsM       []float64 `json:"key_weights_m"`
	KeyWeightsV       []float64 `json:"key_weights_v"`
	ValueWeightsM     []float64 `json:"value_weights_m"`
	ValueWeightsV     []float64 `json:"value_weights_v"`
	AttentionWeightsM []float64 `json:"attention_weights_m"`
	AttentionWeightsV []float64 `json:"attention_weights_v"`
	LN2GammaM         []float64 `json:"ln2_gamma_m"`
	LN2GammaV         []float64 `json:"ln2_gamma_v"`
	LN2BetaM          []float64 `json:"ln2_beta_m"`
	LN2BetaV          []float64 `json:"ln2_beta_v"`
	MLPInputWeightsM  []float64 `json:"mlp_input_weights_m"`
	MLPInputWeightsV  []float64 `json:"mlp_input_weights_v"`
	MLPInputBiasM     []float64 `json:"mlp_input_bias_m"`
	MLPInputBiasV     []float64 `json:"mlp_input_bias_v"`
	MLPOutputWeightsM []float64 `json:"mlp_output_weights_m"`
	MLPOutputWeightsV []float64 `json:"mlp_output_weights_v"`
	MLPOutputBiasM    []float64 `json:"mlp_output_bias_m"`
	MLPOutputBiasV    []float64 `json:"mlp_output_bias_v"`
}

type transformerGradients struct {
	TokenEmbeddings    []float64
	PositionEmbeddings []float64
	Layers             []transformerBlockGradients
	LN1Gamma           []float64
	LN1Beta            []float64
	QueryWeights       []float64
	KeyWeights         []float64
	ValueWeights       []float64
	AttentionWeights   []float64
	LN2Gamma           []float64
	LN2Beta            []float64
	MLPInputWeights    []float64
	MLPInputBias       []float64
	MLPOutputWeights   []float64
	MLPOutputBias      []float64
	FinalLNGamma       []float64
	FinalLNBeta        []float64
	OutputWeights      []float64
	OutputBias         []float64
}

type transformerBlockGradients struct {
	LN1Gamma         []float64
	LN1Beta          []float64
	QueryWeights     []float64
	KeyWeights       []float64
	ValueWeights     []float64
	AttentionWeights []float64
	LN2Gamma         []float64
	LN2Beta          []float64
	MLPInputWeights  []float64
	MLPInputBias     []float64
	MLPOutputWeights []float64
	MLPOutputBias    []float64
}

func NewTransformerTrainer(model *TransformerModel) (*TransformerTrainer, error) {
	if model == nil {
		return nil, fmt.Errorf("model is required")
	}
	model.ensureLayerBlocks()
	return &TransformerTrainer{
		Model: model,
		Adam:  newTransformerOptimizerState(model),
	}, nil
}

func (t *TransformerTrainer) Train(train, val []arithmetic.SequenceExample, cfg TrainingConfig) (TrainingReport, error) {
	if t == nil || t.Model == nil {
		return TrainingReport{}, fmt.Errorf("trainer model is required")
	}
	if err := cfg.Validate(); err != nil {
		return TrainingReport{}, err
	}
	if len(train) == 0 {
		return TrainingReport{}, fmt.Errorf("training set cannot be empty")
	}
	t.Model.ensureLayerBlocks()
	t.Adam = ensureTransformerOptimizerState(t.Model, t.Adam)

	rng := rand.New(rand.NewSource(cfg.Seed))
	indices := make([]int, len(train))
	for i := range indices {
		indices[i] = i
	}

	startStep := t.Step
	runCfg := cfg
	runCfg.scheduleStepOffset = startStep
	started := time.Now()
	report := TrainingReport{Steps: t.Step}
	batch := make([]arithmetic.SequenceExample, 0, cfg.BatchSize)
	for epoch := 0; epoch < runCfg.Epochs && !reachedMaxSteps(runCfg, t.Step, startStep); epoch++ {
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
		for start := 0; start < len(indices); start += runCfg.BatchSize {
			if reachedMaxSteps(runCfg, t.Step, startStep) {
				break
			}
			end := min(len(indices), start+runCfg.BatchSize)
			batch = batch[:0]
			for i := start; i < end; i++ {
				batch = append(batch, train[indices[i]])
			}
			loss, err := t.trainBatch(batch, runCfg)
			if err != nil {
				return TrainingReport{}, err
			}
			report.TrainLoss = loss
			report.Steps = t.Step
			if err := maybeReportProgress(runCfg, t.Step, startStep, loss, started); err != nil {
				return TrainingReport{}, err
			}
			if err := maybeSaveCheckpoint(runCfg, t.Step, startStep); err != nil {
				return TrainingReport{}, err
			}
		}
	}

	if cfg.MaxSteps == 0 && !cfg.SkipFinalTrainLoss {
		trainLoss, err := AverageTransformerLoss(t.Model, train)
		if err != nil {
			return TrainingReport{}, err
		}
		report.TrainLoss = trainLoss
	}
	if len(val) > 0 && !cfg.SkipValidationLoss {
		valLoss, err := AverageTransformerLoss(t.Model, val)
		if err != nil {
			return TrainingReport{}, err
		}
		report.ValLoss = valLoss
	}
	return report, nil
}

func AverageTransformerLoss(model *TransformerModel, sequences []arithmetic.SequenceExample) (float64, error) {
	if model == nil {
		return 0, fmt.Errorf("model is required")
	}
	model.ensureLayerBlocks()
	if len(sequences) == 0 {
		return 0, nil
	}
	total := 0.0
	for _, sequence := range sequences {
		states, err := model.forwardContext(sequence.Context)
		if err != nil {
			return 0, err
		}
		logits := model.logitsForState(states[len(states)-1])
		total += crossEntropyLoss(logits, sequence.Target)
	}
	return total / float64(len(sequences)), nil
}

func (t *TransformerTrainer) trainBatch(batch []arithmetic.SequenceExample, cfg TrainingConfig) (float64, error) {
	grads := zeroTransformerGradients(t.Model)
	totalLoss := 0.0
	for _, example := range batch {
		loss, err := accumulateTransformerGradients(t.Model, example, grads)
		if err != nil {
			return 0, err
		}
		totalLoss += loss
	}

	scale := 1 / float64(len(batch))
	grads.scale(scale)
	grads.clip(cfg.GradClip)

	t.Step++
	t.applyGradients(grads, cfg)
	return totalLoss / float64(len(batch)), nil
}

func accumulateTransformerGradients(model *TransformerModel, example arithmetic.SequenceExample, grads *transformerGradients) (float64, error) {
	states, cache, err := model.forwardContextWithCache(example.Context)
	if err != nil {
		return 0, err
	}
	last := len(states) - 1
	state := states[last]
	model.ensureFinalLayerNorm()
	finalNorm := layerNorm(state, model.FinalLNGamma, model.FinalLNBeta)
	logits := linearWithBias(finalNorm, model.OutputWeights, model.OutputBias, model.LMConfig.VocabSize)
	probs := softmax(logits)
	loss := -math.Log(maxFloat(probs[example.Target], 1e-12))

	dLogits := make([]float64, len(probs))
	copy(dLogits, probs)
	dLogits[example.Target] -= 1

	dFinalNorm := make([]float64, model.LMConfig.EmbeddingDim)
	for dim, value := range finalNorm {
		rowOffset := dim * model.LMConfig.VocabSize
		for token, delta := range dLogits {
			grads.OutputWeights[rowOffset+token] += value * delta
			dFinalNorm[dim] += model.OutputWeights[rowOffset+token] * delta
		}
	}
	for token, delta := range dLogits {
		grads.OutputBias[token] += delta
	}
	dState := layerNormBackward(state, model.FinalLNGamma, dFinalNorm, grads.FinalLNGamma, grads.FinalLNBeta)

	backpropTransformerState(model, cache, last, dState, grads)
	return loss, nil
}

func backpropTransformerState(model *TransformerModel, cache *transformerForwardCache, pos int, dState []float64, grads *transformerGradients) {
	dLayerOutput := zeros2D(model.LMConfig.ContextSize, model.LMConfig.EmbeddingDim)
	addInPlace(dLayerOutput[pos], dState)
	for layerIndex := len(model.Layers) - 1; layerIndex >= 0; layerIndex-- {
		dLayerInput := zeros2D(model.LMConfig.ContextSize, model.LMConfig.EmbeddingDim)
		for statePos, delta := range dLayerOutput {
			if isZeroSlice(delta) {
				continue
			}
			dInput := backpropTransformerBlockState(model.LMConfig, &model.Layers[layerIndex], &cache.Layers[layerIndex], statePos, delta, &grads.Layers[layerIndex])
			add2DInPlace(dLayerInput, dInput)
		}
		dLayerOutput = dLayerInput
	}
	for src := 0; src < model.LMConfig.ContextSize; src++ {
		tokenOffset := cache.Context[src] * model.LMConfig.EmbeddingDim
		positionOffset := src * model.LMConfig.EmbeddingDim
		for dim, delta := range dLayerOutput[src] {
			grads.TokenEmbeddings[tokenOffset+dim] += delta
			grads.PositionEmbeddings[positionOffset+dim] += delta
		}
	}
}

func backpropTransformerBlockState(cfg TransformerConfig, block *TransformerBlock, cache *transformerBlockForwardCache, pos int, dState []float64, grads *transformerBlockGradients) [][]float64 {
	dResidual1 := zeros2D(cfg.ContextSize, cfg.EmbeddingDim)
	dNorm1 := zeros2D(cfg.ContextSize, cfg.EmbeddingDim)
	dQ := zeros2D(cfg.ContextSize, cfg.EmbeddingDim)
	dK := zeros2D(cfg.ContextSize, cfg.EmbeddingDim)
	dV := zeros2D(cfg.ContextSize, cfg.EmbeddingDim)
	dX := zeros2D(cfg.ContextSize, cfg.EmbeddingDim)

	addInPlace(dResidual1[pos], dState)
	dMLPOut := append([]float64(nil), dState...)

	dHidden := make([]float64, cfg.MLPDim)
	for hiddenIndex, hiddenValue := range cache.MLPHidden[pos] {
		rowOffset := hiddenIndex * cfg.EmbeddingDim
		for dim, delta := range dMLPOut {
			grads.MLPOutputWeights[rowOffset+dim] += hiddenValue * delta
			dHidden[hiddenIndex] += block.MLPOutputWeights[rowOffset+dim] * delta
		}
	}
	for dim, delta := range dMLPOut {
		grads.MLPOutputBias[dim] += delta
	}

	dMLPPre := make([]float64, cfg.MLPDim)
	for hiddenIndex, delta := range dHidden {
		dMLPPre[hiddenIndex] = delta * geluDerivative(cache.MLPPre[pos][hiddenIndex])
		grads.MLPInputBias[hiddenIndex] += dMLPPre[hiddenIndex]
	}

	dNorm2 := make([]float64, cfg.EmbeddingDim)
	for dim, value := range cache.Norm2[pos] {
		rowOffset := dim * cfg.MLPDim
		for hiddenIndex, delta := range dMLPPre {
			grads.MLPInputWeights[rowOffset+hiddenIndex] += value * delta
			dNorm2[dim] += block.MLPInputWeights[rowOffset+hiddenIndex] * delta
		}
	}
	addInPlace(dResidual1[pos], layerNormBackward(cache.Residual1[pos], block.LN2Gamma, dNorm2, grads.LN2Gamma, grads.LN2Beta))

	addInPlace(dX[pos], dResidual1[pos])
	dAttnProj := dResidual1[pos]
	dAttended := make([]float64, cfg.EmbeddingDim)
	for dim, value := range cache.Attended[pos] {
		rowOffset := dim * cfg.EmbeddingDim
		for outDim, delta := range dAttnProj {
			grads.AttentionWeights[rowOffset+outDim] += value * delta
			dAttended[dim] += block.AttentionWeights[rowOffset+outDim] * delta
		}
	}

	backpropAttention(cfg, cache, pos, dAttended, dQ, dK, dV)
	backpropQKV(cfg, block, cache, pos, dQ, dK, dV, dNorm1, grads)

	for src := 0; src <= pos; src++ {
		addInPlace(dX[src], layerNormBackward(cache.X[src], block.LN1Gamma, dNorm1[src], grads.LN1Gamma, grads.LN1Beta))
	}
	return dX
}

func backpropAttention(cfg TransformerConfig, cache *transformerBlockForwardCache, pos int, dAttended []float64, dQ, dK, dV [][]float64) {
	headDim := cfg.EmbeddingDim / cfg.NumHeads
	scale := 1 / math.Sqrt(float64(headDim))
	for head := 0; head < cfg.NumHeads; head++ {
		headStart := head * headDim
		probs := cache.AttnProbs[pos][head]
		dProbs := make([]float64, len(probs))
		for src, prob := range probs {
			for dim := 0; dim < headDim; dim++ {
				globalDim := headStart + dim
				dProbs[src] += dAttended[globalDim] * cache.V[src][globalDim]
				dV[src][globalDim] += prob * dAttended[globalDim]
			}
		}
		weightedSum := 0.0
		for src, prob := range probs {
			weightedSum += dProbs[src] * prob
		}
		for src, prob := range probs {
			dScore := prob * (dProbs[src] - weightedSum)
			for dim := 0; dim < headDim; dim++ {
				globalDim := headStart + dim
				dQ[pos][globalDim] += dScore * scale * cache.K[src][globalDim]
				dK[src][globalDim] += dScore * scale * cache.Q[pos][globalDim]
			}
		}
	}
}

func backpropQKV(cfg TransformerConfig, block *TransformerBlock, cache *transformerBlockForwardCache, pos int, dQ, dK, dV, dNorm1 [][]float64, grads *transformerBlockGradients) {
	for src := 0; src <= pos; src++ {
		backpropLinear(cache.Norm1[src], block.KeyWeights, dK[src], grads.KeyWeights, dNorm1[src], cfg.EmbeddingDim)
		backpropLinear(cache.Norm1[src], block.ValueWeights, dV[src], grads.ValueWeights, dNorm1[src], cfg.EmbeddingDim)
	}
	backpropLinear(cache.Norm1[pos], block.QueryWeights, dQ[pos], grads.QueryWeights, dNorm1[pos], cfg.EmbeddingDim)
}

func backpropLinear(input, weights, dOutput, weightGrads, dInput []float64, outDim int) {
	for inIndex, value := range input {
		rowOffset := inIndex * outDim
		for outIndex, delta := range dOutput {
			weightGrads[rowOffset+outIndex] += value * delta
			dInput[inIndex] += weights[rowOffset+outIndex] * delta
		}
	}
}

func layerNormBackward(input, gamma, dOutput, gammaGrads, betaGrads []float64) []float64 {
	size := len(input)
	mean := 0.0
	for _, value := range input {
		mean += value
	}
	mean /= float64(size)
	variance := 0.0
	normalized := make([]float64, size)
	for i, value := range input {
		normalized[i] = value - mean
		variance += normalized[i] * normalized[i]
	}
	variance /= float64(size)
	invStd := 1 / math.Sqrt(variance+1e-5)
	for i := range normalized {
		normalized[i] *= invStd
	}

	dNormalized := make([]float64, size)
	sumDNorm := 0.0
	sumDNormNorm := 0.0
	for i, delta := range dOutput {
		gammaGrads[i] += delta * normalized[i]
		betaGrads[i] += delta
		dNormalized[i] = delta * gamma[i]
		sumDNorm += dNormalized[i]
		sumDNormNorm += dNormalized[i] * normalized[i]
	}

	dInput := make([]float64, size)
	scale := invStd / float64(size)
	for i := range dInput {
		dInput[i] = scale * (float64(size)*dNormalized[i] - sumDNorm - normalized[i]*sumDNormNorm)
	}
	return dInput
}

func zeroTransformerGradients(model *TransformerModel) *transformerGradients {
	model.ensureLayerBlocks()
	grads := &transformerGradients{
		TokenEmbeddings:    make([]float64, len(model.TokenEmbeddings)),
		PositionEmbeddings: make([]float64, len(model.PositionEmbeddings)),
		Layers:             make([]transformerBlockGradients, len(model.Layers)),
		FinalLNGamma:       make([]float64, len(model.FinalLNGamma)),
		FinalLNBeta:        make([]float64, len(model.FinalLNBeta)),
		OutputWeights:      make([]float64, len(model.OutputWeights)),
		OutputBias:         make([]float64, len(model.OutputBias)),
	}
	for layer := range model.Layers {
		grads.Layers[layer] = zeroTransformerBlockGradients(&model.Layers[layer])
	}
	grads.syncLegacyBlockFields()
	return grads
}

func zeroTransformerBlockGradients(block *TransformerBlock) transformerBlockGradients {
	return transformerBlockGradients{
		LN1Gamma:         make([]float64, len(block.LN1Gamma)),
		LN1Beta:          make([]float64, len(block.LN1Beta)),
		QueryWeights:     make([]float64, len(block.QueryWeights)),
		KeyWeights:       make([]float64, len(block.KeyWeights)),
		ValueWeights:     make([]float64, len(block.ValueWeights)),
		AttentionWeights: make([]float64, len(block.AttentionWeights)),
		LN2Gamma:         make([]float64, len(block.LN2Gamma)),
		LN2Beta:          make([]float64, len(block.LN2Beta)),
		MLPInputWeights:  make([]float64, len(block.MLPInputWeights)),
		MLPInputBias:     make([]float64, len(block.MLPInputBias)),
		MLPOutputWeights: make([]float64, len(block.MLPOutputWeights)),
		MLPOutputBias:    make([]float64, len(block.MLPOutputBias)),
	}
}

func (g *transformerGradients) syncLegacyBlockFields() {
	if len(g.Layers) == 0 {
		return
	}
	first := &g.Layers[0]
	g.LN1Gamma = first.LN1Gamma
	g.LN1Beta = first.LN1Beta
	g.QueryWeights = first.QueryWeights
	g.KeyWeights = first.KeyWeights
	g.ValueWeights = first.ValueWeights
	g.AttentionWeights = first.AttentionWeights
	g.LN2Gamma = first.LN2Gamma
	g.LN2Beta = first.LN2Beta
	g.MLPInputWeights = first.MLPInputWeights
	g.MLPInputBias = first.MLPInputBias
	g.MLPOutputWeights = first.MLPOutputWeights
	g.MLPOutputBias = first.MLPOutputBias
}

func (g *transformerBlockGradients) scale(scale float64) {
	scaleSlice(g.LN1Gamma, scale)
	scaleSlice(g.LN1Beta, scale)
	scaleSlice(g.QueryWeights, scale)
	scaleSlice(g.KeyWeights, scale)
	scaleSlice(g.ValueWeights, scale)
	scaleSlice(g.AttentionWeights, scale)
	scaleSlice(g.LN2Gamma, scale)
	scaleSlice(g.LN2Beta, scale)
	scaleSlice(g.MLPInputWeights, scale)
	scaleSlice(g.MLPInputBias, scale)
	scaleSlice(g.MLPOutputWeights, scale)
	scaleSlice(g.MLPOutputBias, scale)
}

func (g *transformerBlockGradients) slices() [][]float64 {
	return [][]float64{
		g.LN1Gamma,
		g.LN1Beta,
		g.QueryWeights,
		g.KeyWeights,
		g.ValueWeights,
		g.AttentionWeights,
		g.LN2Gamma,
		g.LN2Beta,
		g.MLPInputWeights,
		g.MLPInputBias,
		g.MLPOutputWeights,
		g.MLPOutputBias,
	}
}

func (g *transformerGradients) scale(scale float64) {
	scaleSlice(g.TokenEmbeddings, scale)
	scaleSlice(g.PositionEmbeddings, scale)
	for layer := range g.Layers {
		g.Layers[layer].scale(scale)
	}
	scaleSlice(g.FinalLNGamma, scale)
	scaleSlice(g.FinalLNBeta, scale)
	scaleSlice(g.OutputWeights, scale)
	scaleSlice(g.OutputBias, scale)
}

func (g *transformerGradients) clip(maxNorm float64) {
	slices := [][]float64{
		g.TokenEmbeddings,
		g.PositionEmbeddings,
		g.FinalLNGamma,
		g.FinalLNBeta,
		g.OutputWeights,
		g.OutputBias,
	}
	for layer := range g.Layers {
		slices = append(slices, g.Layers[layer].slices()...)
	}
	clipGradientSlices(maxNorm, slices...)
}

func (t *TransformerTrainer) applyGradients(grads *transformerGradients, cfg TrainingConfig) {
	t.Model.ensureLayerBlocks()
	applyAdam(t.Model.TokenEmbeddings, grads.TokenEmbeddings, t.Adam.TokenEmbeddingsM, t.Adam.TokenEmbeddingsV, t.Step, cfg)
	applyAdam(t.Model.PositionEmbeddings, grads.PositionEmbeddings, t.Adam.PositionEmbeddingsM, t.Adam.PositionEmbeddingsV, t.Step, cfg)
	for layer := range t.Model.Layers {
		applyTransformerBlockGradients(&t.Model.Layers[layer], &grads.Layers[layer], &t.Adam.Layers[layer], t.Step, cfg)
	}
	applyAdam(t.Model.FinalLNGamma, grads.FinalLNGamma, t.Adam.FinalLNGammaM, t.Adam.FinalLNGammaV, t.Step, cfg)
	applyAdam(t.Model.FinalLNBeta, grads.FinalLNBeta, t.Adam.FinalLNBetaM, t.Adam.FinalLNBetaV, t.Step, cfg)
	applyAdam(t.Model.OutputWeights, grads.OutputWeights, t.Adam.OutputWeightsM, t.Adam.OutputWeightsV, t.Step, cfg)
	applyAdam(t.Model.OutputBias, grads.OutputBias, t.Adam.OutputBiasM, t.Adam.OutputBiasV, t.Step, cfg)
	t.Model.syncLegacyBlockFields()
}

func newTransformerOptimizerState(model *TransformerModel) *TransformerOptimizerState {
	model.ensureLayerBlocks()
	state := &TransformerOptimizerState{
		Layers:              make([]TransformerBlockOptimizerState, len(model.Layers)),
		TokenEmbeddingsM:    make([]float64, len(model.TokenEmbeddings)),
		TokenEmbeddingsV:    make([]float64, len(model.TokenEmbeddings)),
		PositionEmbeddingsM: make([]float64, len(model.PositionEmbeddings)),
		PositionEmbeddingsV: make([]float64, len(model.PositionEmbeddings)),
		FinalLNGammaM:       make([]float64, len(model.FinalLNGamma)),
		FinalLNGammaV:       make([]float64, len(model.FinalLNGamma)),
		FinalLNBetaM:        make([]float64, len(model.FinalLNBeta)),
		FinalLNBetaV:        make([]float64, len(model.FinalLNBeta)),
		OutputWeightsM:      make([]float64, len(model.OutputWeights)),
		OutputWeightsV:      make([]float64, len(model.OutputWeights)),
		OutputBiasM:         make([]float64, len(model.OutputBias)),
		OutputBiasV:         make([]float64, len(model.OutputBias)),
	}
	for layer := range model.Layers {
		state.Layers[layer] = newTransformerBlockOptimizerState(&model.Layers[layer])
	}
	state.syncLegacyBlockFields()
	return state
}

func ensureTransformerOptimizerState(model *TransformerModel, state *TransformerOptimizerState) *TransformerOptimizerState {
	if state == nil {
		return newTransformerOptimizerState(model)
	}
	model.ensureLayerBlocks()
	fresh := newTransformerOptimizerState(model)
	copyIfSame(fresh.TokenEmbeddingsM, state.TokenEmbeddingsM)
	copyIfSame(fresh.TokenEmbeddingsV, state.TokenEmbeddingsV)
	copyIfSame(fresh.PositionEmbeddingsM, state.PositionEmbeddingsM)
	copyIfSame(fresh.PositionEmbeddingsV, state.PositionEmbeddingsV)
	copyIfSame(fresh.FinalLNGammaM, state.FinalLNGammaM)
	copyIfSame(fresh.FinalLNGammaV, state.FinalLNGammaV)
	copyIfSame(fresh.FinalLNBetaM, state.FinalLNBetaM)
	copyIfSame(fresh.FinalLNBetaV, state.FinalLNBetaV)
	if len(state.Layers) > 0 {
		for layer := range fresh.Layers {
			if layer < len(state.Layers) {
				copyTransformerBlockOptimizerState(&fresh.Layers[layer], &state.Layers[layer])
			}
		}
	} else if len(fresh.Layers) > 0 {
		copyLegacyTransformerBlockOptimizerState(&fresh.Layers[0], state)
	}
	copyIfSame(fresh.OutputWeightsM, state.OutputWeightsM)
	copyIfSame(fresh.OutputWeightsV, state.OutputWeightsV)
	copyIfSame(fresh.OutputBiasM, state.OutputBiasM)
	copyIfSame(fresh.OutputBiasV, state.OutputBiasV)
	fresh.syncLegacyBlockFields()
	return fresh
}

func newTransformerBlockOptimizerState(block *TransformerBlock) TransformerBlockOptimizerState {
	return TransformerBlockOptimizerState{
		LN1GammaM:         make([]float64, len(block.LN1Gamma)),
		LN1GammaV:         make([]float64, len(block.LN1Gamma)),
		LN1BetaM:          make([]float64, len(block.LN1Beta)),
		LN1BetaV:          make([]float64, len(block.LN1Beta)),
		QueryWeightsM:     make([]float64, len(block.QueryWeights)),
		QueryWeightsV:     make([]float64, len(block.QueryWeights)),
		KeyWeightsM:       make([]float64, len(block.KeyWeights)),
		KeyWeightsV:       make([]float64, len(block.KeyWeights)),
		ValueWeightsM:     make([]float64, len(block.ValueWeights)),
		ValueWeightsV:     make([]float64, len(block.ValueWeights)),
		AttentionWeightsM: make([]float64, len(block.AttentionWeights)),
		AttentionWeightsV: make([]float64, len(block.AttentionWeights)),
		LN2GammaM:         make([]float64, len(block.LN2Gamma)),
		LN2GammaV:         make([]float64, len(block.LN2Gamma)),
		LN2BetaM:          make([]float64, len(block.LN2Beta)),
		LN2BetaV:          make([]float64, len(block.LN2Beta)),
		MLPInputWeightsM:  make([]float64, len(block.MLPInputWeights)),
		MLPInputWeightsV:  make([]float64, len(block.MLPInputWeights)),
		MLPInputBiasM:     make([]float64, len(block.MLPInputBias)),
		MLPInputBiasV:     make([]float64, len(block.MLPInputBias)),
		MLPOutputWeightsM: make([]float64, len(block.MLPOutputWeights)),
		MLPOutputWeightsV: make([]float64, len(block.MLPOutputWeights)),
		MLPOutputBiasM:    make([]float64, len(block.MLPOutputBias)),
		MLPOutputBiasV:    make([]float64, len(block.MLPOutputBias)),
	}
}

func applyTransformerBlockGradients(block *TransformerBlock, grads *transformerBlockGradients, state *TransformerBlockOptimizerState, step int, cfg TrainingConfig) {
	applyAdam(block.LN1Gamma, grads.LN1Gamma, state.LN1GammaM, state.LN1GammaV, step, cfg)
	applyAdam(block.LN1Beta, grads.LN1Beta, state.LN1BetaM, state.LN1BetaV, step, cfg)
	applyAdam(block.QueryWeights, grads.QueryWeights, state.QueryWeightsM, state.QueryWeightsV, step, cfg)
	applyAdam(block.KeyWeights, grads.KeyWeights, state.KeyWeightsM, state.KeyWeightsV, step, cfg)
	applyAdam(block.ValueWeights, grads.ValueWeights, state.ValueWeightsM, state.ValueWeightsV, step, cfg)
	applyAdam(block.AttentionWeights, grads.AttentionWeights, state.AttentionWeightsM, state.AttentionWeightsV, step, cfg)
	applyAdam(block.LN2Gamma, grads.LN2Gamma, state.LN2GammaM, state.LN2GammaV, step, cfg)
	applyAdam(block.LN2Beta, grads.LN2Beta, state.LN2BetaM, state.LN2BetaV, step, cfg)
	applyAdam(block.MLPInputWeights, grads.MLPInputWeights, state.MLPInputWeightsM, state.MLPInputWeightsV, step, cfg)
	applyAdam(block.MLPInputBias, grads.MLPInputBias, state.MLPInputBiasM, state.MLPInputBiasV, step, cfg)
	applyAdam(block.MLPOutputWeights, grads.MLPOutputWeights, state.MLPOutputWeightsM, state.MLPOutputWeightsV, step, cfg)
	applyAdam(block.MLPOutputBias, grads.MLPOutputBias, state.MLPOutputBiasM, state.MLPOutputBiasV, step, cfg)
}

func copyTransformerBlockOptimizerState(dst, src *TransformerBlockOptimizerState) {
	copyIfSame(dst.LN1GammaM, src.LN1GammaM)
	copyIfSame(dst.LN1GammaV, src.LN1GammaV)
	copyIfSame(dst.LN1BetaM, src.LN1BetaM)
	copyIfSame(dst.LN1BetaV, src.LN1BetaV)
	copyIfSame(dst.QueryWeightsM, src.QueryWeightsM)
	copyIfSame(dst.QueryWeightsV, src.QueryWeightsV)
	copyIfSame(dst.KeyWeightsM, src.KeyWeightsM)
	copyIfSame(dst.KeyWeightsV, src.KeyWeightsV)
	copyIfSame(dst.ValueWeightsM, src.ValueWeightsM)
	copyIfSame(dst.ValueWeightsV, src.ValueWeightsV)
	copyIfSame(dst.AttentionWeightsM, src.AttentionWeightsM)
	copyIfSame(dst.AttentionWeightsV, src.AttentionWeightsV)
	copyIfSame(dst.LN2GammaM, src.LN2GammaM)
	copyIfSame(dst.LN2GammaV, src.LN2GammaV)
	copyIfSame(dst.LN2BetaM, src.LN2BetaM)
	copyIfSame(dst.LN2BetaV, src.LN2BetaV)
	copyIfSame(dst.MLPInputWeightsM, src.MLPInputWeightsM)
	copyIfSame(dst.MLPInputWeightsV, src.MLPInputWeightsV)
	copyIfSame(dst.MLPInputBiasM, src.MLPInputBiasM)
	copyIfSame(dst.MLPInputBiasV, src.MLPInputBiasV)
	copyIfSame(dst.MLPOutputWeightsM, src.MLPOutputWeightsM)
	copyIfSame(dst.MLPOutputWeightsV, src.MLPOutputWeightsV)
	copyIfSame(dst.MLPOutputBiasM, src.MLPOutputBiasM)
	copyIfSame(dst.MLPOutputBiasV, src.MLPOutputBiasV)
}

func copyLegacyTransformerBlockOptimizerState(dst *TransformerBlockOptimizerState, src *TransformerOptimizerState) {
	copyIfSame(dst.LN1GammaM, src.LN1GammaM)
	copyIfSame(dst.LN1GammaV, src.LN1GammaV)
	copyIfSame(dst.LN1BetaM, src.LN1BetaM)
	copyIfSame(dst.LN1BetaV, src.LN1BetaV)
	copyIfSame(dst.QueryWeightsM, src.QueryWeightsM)
	copyIfSame(dst.QueryWeightsV, src.QueryWeightsV)
	copyIfSame(dst.KeyWeightsM, src.KeyWeightsM)
	copyIfSame(dst.KeyWeightsV, src.KeyWeightsV)
	copyIfSame(dst.ValueWeightsM, src.ValueWeightsM)
	copyIfSame(dst.ValueWeightsV, src.ValueWeightsV)
	copyIfSame(dst.AttentionWeightsM, src.AttentionWeightsM)
	copyIfSame(dst.AttentionWeightsV, src.AttentionWeightsV)
	copyIfSame(dst.LN2GammaM, src.LN2GammaM)
	copyIfSame(dst.LN2GammaV, src.LN2GammaV)
	copyIfSame(dst.LN2BetaM, src.LN2BetaM)
	copyIfSame(dst.LN2BetaV, src.LN2BetaV)
	copyIfSame(dst.MLPInputWeightsM, src.MLPInputWeightsM)
	copyIfSame(dst.MLPInputWeightsV, src.MLPInputWeightsV)
	copyIfSame(dst.MLPInputBiasM, src.MLPInputBiasM)
	copyIfSame(dst.MLPInputBiasV, src.MLPInputBiasV)
	copyIfSame(dst.MLPOutputWeightsM, src.MLPOutputWeightsM)
	copyIfSame(dst.MLPOutputWeightsV, src.MLPOutputWeightsV)
	copyIfSame(dst.MLPOutputBiasM, src.MLPOutputBiasM)
	copyIfSame(dst.MLPOutputBiasV, src.MLPOutputBiasV)
}

func (s *TransformerOptimizerState) syncLegacyBlockFields() {
	if len(s.Layers) == 0 {
		return
	}
	first := &s.Layers[0]
	s.LN1GammaM = first.LN1GammaM
	s.LN1GammaV = first.LN1GammaV
	s.LN1BetaM = first.LN1BetaM
	s.LN1BetaV = first.LN1BetaV
	s.QueryWeightsM = first.QueryWeightsM
	s.QueryWeightsV = first.QueryWeightsV
	s.KeyWeightsM = first.KeyWeightsM
	s.KeyWeightsV = first.KeyWeightsV
	s.ValueWeightsM = first.ValueWeightsM
	s.ValueWeightsV = first.ValueWeightsV
	s.AttentionWeightsM = first.AttentionWeightsM
	s.AttentionWeightsV = first.AttentionWeightsV
	s.LN2GammaM = first.LN2GammaM
	s.LN2GammaV = first.LN2GammaV
	s.LN2BetaM = first.LN2BetaM
	s.LN2BetaV = first.LN2BetaV
	s.MLPInputWeightsM = first.MLPInputWeightsM
	s.MLPInputWeightsV = first.MLPInputWeightsV
	s.MLPInputBiasM = first.MLPInputBiasM
	s.MLPInputBiasV = first.MLPInputBiasV
	s.MLPOutputWeightsM = first.MLPOutputWeightsM
	s.MLPOutputWeightsV = first.MLPOutputWeightsV
	s.MLPOutputBiasM = first.MLPOutputBiasM
	s.MLPOutputBiasV = first.MLPOutputBiasV
}

func copyIfSame(dst, src []float64) {
	if len(dst) == len(src) {
		copy(dst, src)
	}
}

func zeros2D(rows, cols int) [][]float64 {
	values := make([][]float64, rows)
	for i := range values {
		values[i] = make([]float64, cols)
	}
	return values
}

func addInPlace(dst, src []float64) {
	for i, value := range src {
		dst[i] += value
	}
}

func add2DInPlace(dst, src [][]float64) {
	for row := range dst {
		addInPlace(dst[row], src[row])
	}
}

func isZeroSlice(values []float64) bool {
	for _, value := range values {
		if value != 0 {
			return false
		}
	}
	return true
}
