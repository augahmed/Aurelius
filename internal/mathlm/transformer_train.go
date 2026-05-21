package mathlm

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

type TransformerTrainer struct {
	Model *TransformerModel          `json:"model"`
	Adam  *TransformerOptimizerState `json:"adam"`
	Step  int                        `json:"step"`
}

type TransformerOptimizerState struct {
	OutputWeightsM []float64 `json:"output_weights_m"`
	OutputWeightsV []float64 `json:"output_weights_v"`
	OutputBiasM    []float64 `json:"output_bias_m"`
	OutputBiasV    []float64 `json:"output_bias_v"`
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

	rng := rand.New(rand.NewSource(cfg.Seed))
	indices := make([]int, len(train))
	for i := range indices {
		indices[i] = i
	}

	report := TrainingReport{}
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
		for start := 0; start < len(indices); start += cfg.BatchSize {
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
		}
	}

	trainLoss, err := AverageTransformerLoss(t.Model, train)
	if err != nil {
		return TrainingReport{}, err
	}
	report.TrainLoss = trainLoss
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
	weightGrads := make([]float64, len(t.Model.OutputWeights))
	biasGrads := make([]float64, len(t.Model.OutputBias))
	totalLoss := 0.0

	for _, example := range batch {
		loss, err := accumulateTransformerHeadGradients(t.Model, example, weightGrads, biasGrads)
		if err != nil {
			return 0, err
		}
		totalLoss += loss
	}

	scale := 1 / float64(len(batch))
	scaleSlice(weightGrads, scale)
	scaleSlice(biasGrads, scale)

	t.Step++
	applyAdam(t.Model.OutputWeights, weightGrads, t.Adam.OutputWeightsM, t.Adam.OutputWeightsV, t.Step, cfg)
	applyAdam(t.Model.OutputBias, biasGrads, t.Adam.OutputBiasM, t.Adam.OutputBiasV, t.Step, cfg)

	return totalLoss / float64(len(batch)), nil
}

func accumulateTransformerHeadGradients(model *TransformerModel, example arithmetic.SequenceExample, weightGrads, biasGrads []float64) (float64, error) {
	states, err := model.forwardContext(example.Context)
	if err != nil {
		return 0, err
	}
	state := states[len(states)-1]
	logits := model.logitsForState(state)
	probs := softmax(logits)
	loss := -math.Log(maxFloat(probs[example.Target], 1e-12))

	dLogits := make([]float64, len(probs))
	copy(dLogits, probs)
	dLogits[example.Target] -= 1

	for dim, value := range state {
		rowOffset := dim * model.LMConfig.VocabSize
		for token, delta := range dLogits {
			weightGrads[rowOffset+token] += value * delta
		}
	}
	for token, delta := range dLogits {
		biasGrads[token] += delta
	}
	return loss, nil
}

func newTransformerOptimizerState(model *TransformerModel) *TransformerOptimizerState {
	return &TransformerOptimizerState{
		OutputWeightsM: make([]float64, len(model.OutputWeights)),
		OutputWeightsV: make([]float64, len(model.OutputWeights)),
		OutputBiasM:    make([]float64, len(model.OutputBias)),
		OutputBiasV:    make([]float64, len(model.OutputBias)),
	}
}
