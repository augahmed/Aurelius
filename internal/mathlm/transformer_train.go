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
	TokenEmbeddingsM    []float64 `json:"token_embeddings_m"`
	TokenEmbeddingsV    []float64 `json:"token_embeddings_v"`
	PositionEmbeddingsM []float64 `json:"position_embeddings_m"`
	PositionEmbeddingsV []float64 `json:"position_embeddings_v"`
	LN1GammaM           []float64 `json:"ln1_gamma_m"`
	LN1GammaV           []float64 `json:"ln1_gamma_v"`
	LN1BetaM            []float64 `json:"ln1_beta_m"`
	LN1BetaV            []float64 `json:"ln1_beta_v"`
	QueryWeightsM       []float64 `json:"query_weights_m"`
	QueryWeightsV       []float64 `json:"query_weights_v"`
	KeyWeightsM         []float64 `json:"key_weights_m"`
	KeyWeightsV         []float64 `json:"key_weights_v"`
	ValueWeightsM       []float64 `json:"value_weights_m"`
	ValueWeightsV       []float64 `json:"value_weights_v"`
	AttentionWeightsM   []float64 `json:"attention_weights_m"`
	AttentionWeightsV   []float64 `json:"attention_weights_v"`
	LN2GammaM           []float64 `json:"ln2_gamma_m"`
	LN2GammaV           []float64 `json:"ln2_gamma_v"`
	LN2BetaM            []float64 `json:"ln2_beta_m"`
	LN2BetaV            []float64 `json:"ln2_beta_v"`
	MLPInputWeightsM    []float64 `json:"mlp_input_weights_m"`
	MLPInputWeightsV    []float64 `json:"mlp_input_weights_v"`
	MLPInputBiasM       []float64 `json:"mlp_input_bias_m"`
	MLPInputBiasV       []float64 `json:"mlp_input_bias_v"`
	MLPOutputWeightsM   []float64 `json:"mlp_output_weights_m"`
	MLPOutputWeightsV   []float64 `json:"mlp_output_weights_v"`
	MLPOutputBiasM      []float64 `json:"mlp_output_bias_m"`
	MLPOutputBiasV      []float64 `json:"mlp_output_bias_v"`
	OutputWeightsM      []float64 `json:"output_weights_m"`
	OutputWeightsV      []float64 `json:"output_weights_v"`
	OutputBiasM         []float64 `json:"output_bias_m"`
	OutputBiasV         []float64 `json:"output_bias_v"`
}

type transformerGradients struct {
	TokenEmbeddings    []float64
	PositionEmbeddings []float64
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
	OutputWeights      []float64
	OutputBias         []float64
}

func NewTransformerTrainer(model *TransformerModel) (*TransformerTrainer, error) {
	if model == nil {
		return nil, fmt.Errorf("model is required")
	}
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
	t.Adam = ensureTransformerOptimizerState(t.Model, t.Adam)

	rng := rand.New(rand.NewSource(cfg.Seed))
	indices := make([]int, len(train))
	for i := range indices {
		indices[i] = i
	}

	startStep := t.Step
	started := time.Now()
	report := TrainingReport{Steps: t.Step}
	for epoch := 0; epoch < cfg.Epochs && !reachedMaxSteps(cfg, t.Step, startStep); epoch++ {
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
		for start := 0; start < len(indices); start += cfg.BatchSize {
			if reachedMaxSteps(cfg, t.Step, startStep) {
				break
			}
			end := min(len(indices), start+cfg.BatchSize)
			batch := make([]arithmetic.SequenceExample, end-start)
			for i := range batch {
				batch[i] = train[indices[start+i]]
			}
			loss, err := t.trainBatch(batch, cfg)
			if err != nil {
				return TrainingReport{}, err
			}
			report.TrainLoss = loss
			report.Steps = t.Step
			if err := maybeReportProgress(cfg, t.Step, startStep, loss, started); err != nil {
				return TrainingReport{}, err
			}
			if err := maybeSaveCheckpoint(cfg, t.Step, startStep); err != nil {
				return TrainingReport{}, err
			}
		}
	}

	if cfg.MaxSteps == 0 {
		trainLoss, err := AverageTransformerLoss(t.Model, train)
		if err != nil {
			return TrainingReport{}, err
		}
		report.TrainLoss = trainLoss
	}
	if len(val) > 0 {
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
	logits := model.logitsForState(state)
	probs := softmax(logits)
	loss := -math.Log(maxFloat(probs[example.Target], 1e-12))

	dLogits := make([]float64, len(probs))
	copy(dLogits, probs)
	dLogits[example.Target] -= 1

	dState := make([]float64, model.LMConfig.EmbeddingDim)
	for dim, value := range state {
		rowOffset := dim * model.LMConfig.VocabSize
		for token, delta := range dLogits {
			grads.OutputWeights[rowOffset+token] += value * delta
			dState[dim] += model.OutputWeights[rowOffset+token] * delta
		}
	}
	for token, delta := range dLogits {
		grads.OutputBias[token] += delta
	}

	backpropTransformerState(model, cache, last, dState, grads)
	return loss, nil
}

func backpropTransformerState(model *TransformerModel, cache *transformerForwardCache, pos int, dState []float64, grads *transformerGradients) {
	cfg := model.LMConfig
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
			dHidden[hiddenIndex] += model.MLPOutputWeights[rowOffset+dim] * delta
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
			dNorm2[dim] += model.MLPInputWeights[rowOffset+hiddenIndex] * delta
		}
	}
	addInPlace(dResidual1[pos], layerNormBackward(cache.Residual1[pos], model.LN2Gamma, dNorm2, grads.LN2Gamma, grads.LN2Beta))

	addInPlace(dX[pos], dResidual1[pos])
	dAttnProj := dResidual1[pos]
	dAttended := make([]float64, cfg.EmbeddingDim)
	for dim, value := range cache.Attended[pos] {
		rowOffset := dim * cfg.EmbeddingDim
		for outDim, delta := range dAttnProj {
			grads.AttentionWeights[rowOffset+outDim] += value * delta
			dAttended[dim] += model.AttentionWeights[rowOffset+outDim] * delta
		}
	}

	backpropAttention(model, cache, pos, dAttended, dQ, dK, dV)
	backpropQKV(model, cache, pos, dQ, dK, dV, dNorm1, grads)

	for src := 0; src <= pos; src++ {
		addInPlace(dX[src], layerNormBackward(cache.X[src], model.LN1Gamma, dNorm1[src], grads.LN1Gamma, grads.LN1Beta))
	}
	for src := 0; src <= pos; src++ {
		tokenOffset := cache.Context[src] * cfg.EmbeddingDim
		positionOffset := src * cfg.EmbeddingDim
		for dim, delta := range dX[src] {
			grads.TokenEmbeddings[tokenOffset+dim] += delta
			grads.PositionEmbeddings[positionOffset+dim] += delta
		}
	}
}

func backpropAttention(model *TransformerModel, cache *transformerForwardCache, pos int, dAttended []float64, dQ, dK, dV [][]float64) {
	cfg := model.LMConfig
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

func backpropQKV(model *TransformerModel, cache *transformerForwardCache, pos int, dQ, dK, dV, dNorm1 [][]float64, grads *transformerGradients) {
	cfg := model.LMConfig
	for src := 0; src <= pos; src++ {
		backpropLinear(cache.Norm1[src], model.KeyWeights, dK[src], grads.KeyWeights, dNorm1[src], cfg.EmbeddingDim)
		backpropLinear(cache.Norm1[src], model.ValueWeights, dV[src], grads.ValueWeights, dNorm1[src], cfg.EmbeddingDim)
	}
	backpropLinear(cache.Norm1[pos], model.QueryWeights, dQ[pos], grads.QueryWeights, dNorm1[pos], cfg.EmbeddingDim)
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
	return &transformerGradients{
		TokenEmbeddings:    make([]float64, len(model.TokenEmbeddings)),
		PositionEmbeddings: make([]float64, len(model.PositionEmbeddings)),
		LN1Gamma:           make([]float64, len(model.LN1Gamma)),
		LN1Beta:            make([]float64, len(model.LN1Beta)),
		QueryWeights:       make([]float64, len(model.QueryWeights)),
		KeyWeights:         make([]float64, len(model.KeyWeights)),
		ValueWeights:       make([]float64, len(model.ValueWeights)),
		AttentionWeights:   make([]float64, len(model.AttentionWeights)),
		LN2Gamma:           make([]float64, len(model.LN2Gamma)),
		LN2Beta:            make([]float64, len(model.LN2Beta)),
		MLPInputWeights:    make([]float64, len(model.MLPInputWeights)),
		MLPInputBias:       make([]float64, len(model.MLPInputBias)),
		MLPOutputWeights:   make([]float64, len(model.MLPOutputWeights)),
		MLPOutputBias:      make([]float64, len(model.MLPOutputBias)),
		OutputWeights:      make([]float64, len(model.OutputWeights)),
		OutputBias:         make([]float64, len(model.OutputBias)),
	}
}

func (g *transformerGradients) scale(scale float64) {
	scaleSlice(g.TokenEmbeddings, scale)
	scaleSlice(g.PositionEmbeddings, scale)
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
	scaleSlice(g.OutputWeights, scale)
	scaleSlice(g.OutputBias, scale)
}

func (g *transformerGradients) clip(maxNorm float64) {
	clipGradientSlices(maxNorm,
		g.TokenEmbeddings,
		g.PositionEmbeddings,
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
		g.OutputWeights,
		g.OutputBias,
	)
}

func (t *TransformerTrainer) applyGradients(grads *transformerGradients, cfg TrainingConfig) {
	applyAdam(t.Model.TokenEmbeddings, grads.TokenEmbeddings, t.Adam.TokenEmbeddingsM, t.Adam.TokenEmbeddingsV, t.Step, cfg)
	applyAdam(t.Model.PositionEmbeddings, grads.PositionEmbeddings, t.Adam.PositionEmbeddingsM, t.Adam.PositionEmbeddingsV, t.Step, cfg)
	applyAdam(t.Model.LN1Gamma, grads.LN1Gamma, t.Adam.LN1GammaM, t.Adam.LN1GammaV, t.Step, cfg)
	applyAdam(t.Model.LN1Beta, grads.LN1Beta, t.Adam.LN1BetaM, t.Adam.LN1BetaV, t.Step, cfg)
	applyAdam(t.Model.QueryWeights, grads.QueryWeights, t.Adam.QueryWeightsM, t.Adam.QueryWeightsV, t.Step, cfg)
	applyAdam(t.Model.KeyWeights, grads.KeyWeights, t.Adam.KeyWeightsM, t.Adam.KeyWeightsV, t.Step, cfg)
	applyAdam(t.Model.ValueWeights, grads.ValueWeights, t.Adam.ValueWeightsM, t.Adam.ValueWeightsV, t.Step, cfg)
	applyAdam(t.Model.AttentionWeights, grads.AttentionWeights, t.Adam.AttentionWeightsM, t.Adam.AttentionWeightsV, t.Step, cfg)
	applyAdam(t.Model.LN2Gamma, grads.LN2Gamma, t.Adam.LN2GammaM, t.Adam.LN2GammaV, t.Step, cfg)
	applyAdam(t.Model.LN2Beta, grads.LN2Beta, t.Adam.LN2BetaM, t.Adam.LN2BetaV, t.Step, cfg)
	applyAdam(t.Model.MLPInputWeights, grads.MLPInputWeights, t.Adam.MLPInputWeightsM, t.Adam.MLPInputWeightsV, t.Step, cfg)
	applyAdam(t.Model.MLPInputBias, grads.MLPInputBias, t.Adam.MLPInputBiasM, t.Adam.MLPInputBiasV, t.Step, cfg)
	applyAdam(t.Model.MLPOutputWeights, grads.MLPOutputWeights, t.Adam.MLPOutputWeightsM, t.Adam.MLPOutputWeightsV, t.Step, cfg)
	applyAdam(t.Model.MLPOutputBias, grads.MLPOutputBias, t.Adam.MLPOutputBiasM, t.Adam.MLPOutputBiasV, t.Step, cfg)
	applyAdam(t.Model.OutputWeights, grads.OutputWeights, t.Adam.OutputWeightsM, t.Adam.OutputWeightsV, t.Step, cfg)
	applyAdam(t.Model.OutputBias, grads.OutputBias, t.Adam.OutputBiasM, t.Adam.OutputBiasV, t.Step, cfg)
}

func newTransformerOptimizerState(model *TransformerModel) *TransformerOptimizerState {
	return &TransformerOptimizerState{
		TokenEmbeddingsM:    make([]float64, len(model.TokenEmbeddings)),
		TokenEmbeddingsV:    make([]float64, len(model.TokenEmbeddings)),
		PositionEmbeddingsM: make([]float64, len(model.PositionEmbeddings)),
		PositionEmbeddingsV: make([]float64, len(model.PositionEmbeddings)),
		LN1GammaM:           make([]float64, len(model.LN1Gamma)),
		LN1GammaV:           make([]float64, len(model.LN1Gamma)),
		LN1BetaM:            make([]float64, len(model.LN1Beta)),
		LN1BetaV:            make([]float64, len(model.LN1Beta)),
		QueryWeightsM:       make([]float64, len(model.QueryWeights)),
		QueryWeightsV:       make([]float64, len(model.QueryWeights)),
		KeyWeightsM:         make([]float64, len(model.KeyWeights)),
		KeyWeightsV:         make([]float64, len(model.KeyWeights)),
		ValueWeightsM:       make([]float64, len(model.ValueWeights)),
		ValueWeightsV:       make([]float64, len(model.ValueWeights)),
		AttentionWeightsM:   make([]float64, len(model.AttentionWeights)),
		AttentionWeightsV:   make([]float64, len(model.AttentionWeights)),
		LN2GammaM:           make([]float64, len(model.LN2Gamma)),
		LN2GammaV:           make([]float64, len(model.LN2Gamma)),
		LN2BetaM:            make([]float64, len(model.LN2Beta)),
		LN2BetaV:            make([]float64, len(model.LN2Beta)),
		MLPInputWeightsM:    make([]float64, len(model.MLPInputWeights)),
		MLPInputWeightsV:    make([]float64, len(model.MLPInputWeights)),
		MLPInputBiasM:       make([]float64, len(model.MLPInputBias)),
		MLPInputBiasV:       make([]float64, len(model.MLPInputBias)),
		MLPOutputWeightsM:   make([]float64, len(model.MLPOutputWeights)),
		MLPOutputWeightsV:   make([]float64, len(model.MLPOutputWeights)),
		MLPOutputBiasM:      make([]float64, len(model.MLPOutputBias)),
		MLPOutputBiasV:      make([]float64, len(model.MLPOutputBias)),
		OutputWeightsM:      make([]float64, len(model.OutputWeights)),
		OutputWeightsV:      make([]float64, len(model.OutputWeights)),
		OutputBiasM:         make([]float64, len(model.OutputBias)),
		OutputBiasV:         make([]float64, len(model.OutputBias)),
	}
}

func ensureTransformerOptimizerState(model *TransformerModel, state *TransformerOptimizerState) *TransformerOptimizerState {
	if state == nil {
		return newTransformerOptimizerState(model)
	}
	fresh := newTransformerOptimizerState(model)
	copyIfSame(fresh.TokenEmbeddingsM, state.TokenEmbeddingsM)
	copyIfSame(fresh.TokenEmbeddingsV, state.TokenEmbeddingsV)
	copyIfSame(fresh.PositionEmbeddingsM, state.PositionEmbeddingsM)
	copyIfSame(fresh.PositionEmbeddingsV, state.PositionEmbeddingsV)
	copyIfSame(fresh.LN1GammaM, state.LN1GammaM)
	copyIfSame(fresh.LN1GammaV, state.LN1GammaV)
	copyIfSame(fresh.LN1BetaM, state.LN1BetaM)
	copyIfSame(fresh.LN1BetaV, state.LN1BetaV)
	copyIfSame(fresh.QueryWeightsM, state.QueryWeightsM)
	copyIfSame(fresh.QueryWeightsV, state.QueryWeightsV)
	copyIfSame(fresh.KeyWeightsM, state.KeyWeightsM)
	copyIfSame(fresh.KeyWeightsV, state.KeyWeightsV)
	copyIfSame(fresh.ValueWeightsM, state.ValueWeightsM)
	copyIfSame(fresh.ValueWeightsV, state.ValueWeightsV)
	copyIfSame(fresh.AttentionWeightsM, state.AttentionWeightsM)
	copyIfSame(fresh.AttentionWeightsV, state.AttentionWeightsV)
	copyIfSame(fresh.LN2GammaM, state.LN2GammaM)
	copyIfSame(fresh.LN2GammaV, state.LN2GammaV)
	copyIfSame(fresh.LN2BetaM, state.LN2BetaM)
	copyIfSame(fresh.LN2BetaV, state.LN2BetaV)
	copyIfSame(fresh.MLPInputWeightsM, state.MLPInputWeightsM)
	copyIfSame(fresh.MLPInputWeightsV, state.MLPInputWeightsV)
	copyIfSame(fresh.MLPInputBiasM, state.MLPInputBiasM)
	copyIfSame(fresh.MLPInputBiasV, state.MLPInputBiasV)
	copyIfSame(fresh.MLPOutputWeightsM, state.MLPOutputWeightsM)
	copyIfSame(fresh.MLPOutputWeightsV, state.MLPOutputWeightsV)
	copyIfSame(fresh.MLPOutputBiasM, state.MLPOutputBiasM)
	copyIfSame(fresh.MLPOutputBiasV, state.MLPOutputBiasV)
	copyIfSame(fresh.OutputWeightsM, state.OutputWeightsM)
	copyIfSame(fresh.OutputWeightsV, state.OutputWeightsV)
	copyIfSame(fresh.OutputBiasM, state.OutputBiasM)
	copyIfSame(fresh.OutputBiasV, state.OutputBiasV)
	return fresh
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
