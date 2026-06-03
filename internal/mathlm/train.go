package mathlm

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

type TrainingConfig struct {
	Epochs             int
	BatchSize          int
	LearningRate       float64
	WarmupSteps        int
	DecaySteps         int
	MinLearningRate    float64
	Beta1              float64
	Beta2              float64
	Epsilon            float64
	Seed               int64
	MaxSteps           int
	LogEvery           int
	SaveEvery          int
	GradClip           float64
	SkipFinalTrainLoss bool
	SkipValidationLoss bool
	OnProgress         func(TrainingProgress) error
	OnCheckpoint       func(step int) error

	scheduleStepOffset int
}

type TrainingReport struct {
	TrainLoss float64
	ValLoss   float64
	Steps     int
}

type TrainingProgress struct {
	Step           int
	Loss           float64
	LearningRate   float64
	Elapsed        time.Duration
	StepsPerSecond float64
}

type Trainer struct {
	Model *Model          `json:"model"`
	Adam  *OptimizerState `json:"adam"`
	Step  int             `json:"step"`
}

type OptimizerState struct {
	EmbeddingsM    []float64 `json:"embeddings_m"`
	EmbeddingsV    []float64 `json:"embeddings_v"`
	HiddenWeightsM []float64 `json:"hidden_weights_m"`
	HiddenWeightsV []float64 `json:"hidden_weights_v"`
	HiddenBiasM    []float64 `json:"hidden_bias_m"`
	HiddenBiasV    []float64 `json:"hidden_bias_v"`
	OutputWeightsM []float64 `json:"output_weights_m"`
	OutputWeightsV []float64 `json:"output_weights_v"`
	OutputBiasM    []float64 `json:"output_bias_m"`
	OutputBiasV    []float64 `json:"output_bias_v"`
}

func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		Epochs:       8,
		BatchSize:    64,
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Seed:         1,
	}
}

func NewTrainer(model *Model) (*Trainer, error) {
	if model == nil {
		return nil, fmt.Errorf("model is required")
	}
	return &Trainer{
		Model: model,
		Adam:  newOptimizerState(model),
	}, nil
}

func (c TrainingConfig) Validate() error {
	if c.Epochs <= 0 {
		return fmt.Errorf("epochs must be positive")
	}
	if c.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive")
	}
	if c.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive")
	}
	if c.WarmupSteps < 0 {
		return fmt.Errorf("warmup steps cannot be negative")
	}
	if c.DecaySteps < 0 {
		return fmt.Errorf("decay steps cannot be negative")
	}
	if c.MinLearningRate < 0 {
		return fmt.Errorf("min learning rate cannot be negative")
	}
	if c.MinLearningRate > c.LearningRate {
		return fmt.Errorf("min learning rate cannot exceed learning rate")
	}
	if c.Beta1 <= 0 || c.Beta1 >= 1 {
		return fmt.Errorf("beta1 must be between 0 and 1")
	}
	if c.Beta2 <= 0 || c.Beta2 >= 1 {
		return fmt.Errorf("beta2 must be between 0 and 1")
	}
	if c.Epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive")
	}
	if c.MaxSteps < 0 {
		return fmt.Errorf("max steps cannot be negative")
	}
	if c.LogEvery < 0 {
		return fmt.Errorf("log every cannot be negative")
	}
	if c.SaveEvery < 0 {
		return fmt.Errorf("save every cannot be negative")
	}
	if c.GradClip < 0 {
		return fmt.Errorf("grad clip cannot be negative")
	}
	return nil
}

func (t *Trainer) Train(train, val []arithmetic.SequenceExample, cfg TrainingConfig) (TrainingReport, error) {
	if t == nil || t.Model == nil {
		return TrainingReport{}, fmt.Errorf("trainer model is required")
	}
	if err := cfg.Validate(); err != nil {
		return TrainingReport{}, err
	}
	if len(train) == 0 {
		return TrainingReport{}, fmt.Errorf("training set cannot be empty")
	}

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
		trainLoss, err := AverageLoss(t.Model, train)
		if err != nil {
			return TrainingReport{}, err
		}
		report.TrainLoss = trainLoss
	}
	if len(val) > 0 && !cfg.SkipValidationLoss {
		valLoss, err := AverageLoss(t.Model, val)
		if err != nil {
			return TrainingReport{}, err
		}
		report.ValLoss = valLoss
	}
	return report, nil
}

func AverageLoss(model *Model, sequences []arithmetic.SequenceExample) (float64, error) {
	if model == nil {
		return 0, fmt.Errorf("model is required")
	}
	if len(sequences) == 0 {
		return 0, nil
	}
	total := 0.0
	for _, sequence := range sequences {
		_, _, logits, err := model.forwardContext(sequence.Context)
		if err != nil {
			return 0, err
		}
		total += crossEntropyLoss(logits, sequence.Target)
	}
	return total / float64(len(sequences)), nil
}

func (t *Trainer) trainBatch(batch []arithmetic.SequenceExample, cfg TrainingConfig) (float64, error) {
	grads := zeroGradients(t.Model)
	totalLoss := 0.0

	for _, example := range batch {
		loss, err := accumulateGradients(t.Model, example, grads)
		if err != nil {
			return 0, err
		}
		totalLoss += loss
	}

	scale := 1 / float64(len(batch))
	scaleSlice(grads.Embeddings, scale)
	scaleSlice(grads.HiddenWeights, scale)
	scaleSlice(grads.HiddenBias, scale)
	scaleSlice(grads.OutputWeights, scale)
	scaleSlice(grads.OutputBias, scale)
	clipGradientSlices(cfg.GradClip, grads.Embeddings, grads.HiddenWeights, grads.HiddenBias, grads.OutputWeights, grads.OutputBias)

	t.Step++
	applyAdam(t.Model.Embeddings, grads.Embeddings, t.Adam.EmbeddingsM, t.Adam.EmbeddingsV, t.Step, cfg)
	applyAdam(t.Model.HiddenWeights, grads.HiddenWeights, t.Adam.HiddenWeightsM, t.Adam.HiddenWeightsV, t.Step, cfg)
	applyAdam(t.Model.HiddenBias, grads.HiddenBias, t.Adam.HiddenBiasM, t.Adam.HiddenBiasV, t.Step, cfg)
	applyAdam(t.Model.OutputWeights, grads.OutputWeights, t.Adam.OutputWeightsM, t.Adam.OutputWeightsV, t.Step, cfg)
	applyAdam(t.Model.OutputBias, grads.OutputBias, t.Adam.OutputBiasM, t.Adam.OutputBiasV, t.Step, cfg)

	return totalLoss / float64(len(batch)), nil
}

type gradients struct {
	Embeddings    []float64
	HiddenWeights []float64
	HiddenBias    []float64
	OutputWeights []float64
	OutputBias    []float64
}

func zeroGradients(model *Model) *gradients {
	return &gradients{
		Embeddings:    make([]float64, len(model.Embeddings)),
		HiddenWeights: make([]float64, len(model.HiddenWeights)),
		HiddenBias:    make([]float64, len(model.HiddenBias)),
		OutputWeights: make([]float64, len(model.OutputWeights)),
		OutputBias:    make([]float64, len(model.OutputBias)),
	}
}

func accumulateGradients(model *Model, example arithmetic.SequenceExample, grads *gradients) (float64, error) {
	inputVector, hidden, logits, err := model.forwardContext(example.Context)
	if err != nil {
		return 0, err
	}
	probs := softmax(logits)
	loss := -math.Log(maxFloat(probs[example.Target], 1e-12))

	dLogits := make([]float64, len(probs))
	copy(dLogits, probs)
	dLogits[example.Target] -= 1

	for hiddenIndex, hiddenValue := range hidden {
		rowOffset := hiddenIndex * model.LMConfig.VocabSize
		for tokenIndex, delta := range dLogits {
			grads.OutputWeights[rowOffset+tokenIndex] += hiddenValue * delta
		}
	}
	for tokenIndex, delta := range dLogits {
		grads.OutputBias[tokenIndex] += delta
	}

	dHidden := make([]float64, len(hidden))
	for hiddenIndex := range hidden {
		sum := 0.0
		rowOffset := hiddenIndex * model.LMConfig.VocabSize
		for tokenIndex, delta := range dLogits {
			sum += model.OutputWeights[rowOffset+tokenIndex] * delta
		}
		dHidden[hiddenIndex] = sum * (1 - hidden[hiddenIndex]*hidden[hiddenIndex])
		grads.HiddenBias[hiddenIndex] += dHidden[hiddenIndex]
	}

	dInput := make([]float64, len(inputVector))
	for inputIndex, inputValue := range inputVector {
		rowOffset := inputIndex * model.LMConfig.HiddenDim
		sum := 0.0
		for hiddenIndex, delta := range dHidden {
			grads.HiddenWeights[rowOffset+hiddenIndex] += inputValue * delta
			sum += model.HiddenWeights[rowOffset+hiddenIndex] * delta
		}
		dInput[inputIndex] = sum
	}

	for pos, token := range example.Context {
		embedOffset := token * model.LMConfig.EmbeddingDim
		inputOffset := pos * model.LMConfig.EmbeddingDim
		for dim := 0; dim < model.LMConfig.EmbeddingDim; dim++ {
			grads.Embeddings[embedOffset+dim] += dInput[inputOffset+dim]
		}
	}

	return loss, nil
}

func softmax(logits []float64) []float64 {
	maxLogit := logits[0]
	for _, logit := range logits[1:] {
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, logit := range logits {
		probs[i] = math.Exp(logit - maxLogit)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func crossEntropyLoss(logits []float64, target int) float64 {
	probs := softmax(logits)
	return -math.Log(maxFloat(probs[target], 1e-12))
}

func tanh(value float64) float64 {
	return math.Tanh(value)
}

func newOptimizerState(model *Model) *OptimizerState {
	return &OptimizerState{
		EmbeddingsM:    make([]float64, len(model.Embeddings)),
		EmbeddingsV:    make([]float64, len(model.Embeddings)),
		HiddenWeightsM: make([]float64, len(model.HiddenWeights)),
		HiddenWeightsV: make([]float64, len(model.HiddenWeights)),
		HiddenBiasM:    make([]float64, len(model.HiddenBias)),
		HiddenBiasV:    make([]float64, len(model.HiddenBias)),
		OutputWeightsM: make([]float64, len(model.OutputWeights)),
		OutputWeightsV: make([]float64, len(model.OutputWeights)),
		OutputBiasM:    make([]float64, len(model.OutputBias)),
		OutputBiasV:    make([]float64, len(model.OutputBias)),
	}
}

func applyAdam(params, grads, m, v []float64, step int, cfg TrainingConfig) {
	beta1Pow := math.Pow(cfg.Beta1, float64(step))
	beta2Pow := math.Pow(cfg.Beta2, float64(step))
	learningRate := effectiveLearningRate(cfg, step)
	for i, grad := range grads {
		m[i] = cfg.Beta1*m[i] + (1-cfg.Beta1)*grad
		v[i] = cfg.Beta2*v[i] + (1-cfg.Beta2)*grad*grad
		mHat := m[i] / (1 - beta1Pow)
		vHat := v[i] / (1 - beta2Pow)
		params[i] -= learningRate * mHat / (math.Sqrt(vHat) + cfg.Epsilon)
	}
}

func effectiveLearningRate(cfg TrainingConfig, step int) float64 {
	scheduleStep := step - cfg.scheduleStepOffset
	if scheduleStep < 1 {
		scheduleStep = 1
	}
	learningRate := cfg.LearningRate
	if cfg.WarmupSteps > 0 && scheduleStep < cfg.WarmupSteps {
		learningRate *= float64(scheduleStep) / float64(cfg.WarmupSteps)
	}
	if cfg.DecaySteps > 0 && scheduleStep > cfg.WarmupSteps {
		decayStep := min(scheduleStep-cfg.WarmupSteps, cfg.DecaySteps)
		progress := float64(decayStep) / float64(cfg.DecaySteps)
		learningRate = cfg.LearningRate - (cfg.LearningRate-cfg.MinLearningRate)*progress
	}
	if learningRate < cfg.MinLearningRate {
		return cfg.MinLearningRate
	}
	return learningRate
}

func scaleSlice(values []float64, scale float64) {
	for i := range values {
		values[i] *= scale
	}
}

func clipGradientSlices(maxNorm float64, slices ...[]float64) {
	if maxNorm <= 0 {
		return
	}
	sumSquares := 0.0
	for _, values := range slices {
		for _, value := range values {
			sumSquares += value * value
		}
	}
	norm := math.Sqrt(sumSquares)
	if norm == 0 || norm <= maxNorm {
		return
	}
	scale := maxNorm / norm
	for _, values := range slices {
		scaleSlice(values, scale)
	}
}

func reachedMaxSteps(cfg TrainingConfig, currentStep, startStep int) bool {
	return cfg.MaxSteps > 0 && currentStep-startStep >= cfg.MaxSteps
}

func maybeReportProgress(cfg TrainingConfig, step, startStep int, loss float64, started time.Time) error {
	stepsThisRun := step - startStep
	if cfg.LogEvery <= 0 || cfg.OnProgress == nil || stepsThisRun%cfg.LogEvery != 0 {
		return nil
	}
	elapsed := time.Since(started)
	stepsPerSecond := 0.0
	if elapsed > 0 {
		stepsPerSecond = float64(stepsThisRun) / elapsed.Seconds()
	}
	return cfg.OnProgress(TrainingProgress{
		Step:           step,
		Loss:           loss,
		LearningRate:   effectiveLearningRate(cfg, step),
		Elapsed:        elapsed,
		StepsPerSecond: stepsPerSecond,
	})
}

func maybeSaveCheckpoint(cfg TrainingConfig, step, startStep int) error {
	if cfg.SaveEvery <= 0 || cfg.OnCheckpoint == nil || (step-startStep)%cfg.SaveEvery != 0 {
		return nil
	}
	return cfg.OnCheckpoint(step)
}

func maxFloat(left, right float64) float64 {
	if left > right {
		return left
	}
	return right
}
