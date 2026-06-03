package mathlm

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

func TestModelForwardShape(t *testing.T) {
	model, err := NewModel(Config{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 4,
		HiddenDim:    16,
		Seed:         1,
	})
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}
	logits, err := model.Forward([]int{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if len(logits) != 256 {
		t.Fatalf("len(logits) = %d, want 256", len(logits))
	}
}

func TestTrainingReducesLoss(t *testing.T) {
	model, err := NewModel(Config{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 8,
		HiddenDim:    32,
		Seed:         2,
	})
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}
	trainer, err := NewTrainer(model)
	if err != nil {
		t.Fatalf("NewTrainer error: %v", err)
	}

	examples := []arithmetic.Example{
		{Prompt: "2 + 2 = ", Completion: "4", Operation: "add"},
		{Prompt: "3 + 1 = ", Completion: "4", Operation: "add"},
		{Prompt: "5 - 2 = ", Completion: "3", Operation: "sub"},
		{Prompt: "6 - 1 = ", Completion: "5", Operation: "sub"},
	}
	sequences, err := arithmetic.BuildTrainingSequences(examples, byteTok(), 8)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}

	before, err := AverageLoss(model, sequences)
	if err != nil {
		t.Fatalf("AverageLoss(before) error: %v", err)
	}
	if _, err := trainer.Train(sequences, sequences, TrainingConfig{
		Epochs:       20,
		BatchSize:    8,
		LearningRate: 0.02,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Seed:         3,
	}); err != nil {
		t.Fatalf("Train error: %v", err)
	}
	after, err := AverageLoss(model, sequences)
	if err != nil {
		t.Fatalf("AverageLoss(after) error: %v", err)
	}
	if after >= before {
		t.Fatalf("loss after training = %f, want less than %f", after, before)
	}
}

func TestTrainingControlsMaxStepsAndProgress(t *testing.T) {
	model, err := NewModel(Config{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 8,
		HiddenDim:    16,
		Seed:         22,
	})
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}
	trainer, err := NewTrainer(model)
	if err != nil {
		t.Fatalf("NewTrainer error: %v", err)
	}
	examples := []arithmetic.Example{
		{Prompt: "2 + 2 = ", Completion: "4", Operation: "add", Level: 1},
		{Prompt: "5 - 2 = ", Completion: "3", Operation: "sub", Level: 1},
	}
	sequences, err := arithmetic.BuildTrainingSequences(examples, byteTok(), 8)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}
	progressSteps := []int{}
	learningRates := []float64{}
	report, err := trainer.Train(sequences, sequences, TrainingConfig{
		Epochs:       20,
		BatchSize:    1,
		LearningRate: 0.01,
		WarmupSteps:  2,
		DecaySteps:   2,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Seed:         23,
		MaxSteps:     3,
		LogEvery:     1,
		GradClip:     0.5,
		OnProgress: func(progress TrainingProgress) error {
			progressSteps = append(progressSteps, progress.Step)
			learningRates = append(learningRates, progress.LearningRate)
			if progress.StepsPerSecond <= 0 {
				t.Fatalf("steps/sec = %f, want positive", progress.StepsPerSecond)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Train error: %v", err)
	}
	if report.Steps != 3 || trainer.Step != 3 {
		t.Fatalf("steps = report %d trainer %d, want 3", report.Steps, trainer.Step)
	}
	if len(progressSteps) != 3 {
		t.Fatalf("progress callback count = %d, want 3", len(progressSteps))
	}
	if learningRates[0] >= learningRates[1] {
		t.Fatalf("warmup learning rates = %v, want step 2 greater than step 1", learningRates)
	}
	if learningRates[2] >= learningRates[1] {
		t.Fatalf("decay learning rates = %v, want step 3 less than step 2", learningRates)
	}
}

func TestEffectiveLearningRateIsRunRelative(t *testing.T) {
	cfg := TrainingConfig{
		LearningRate:       0.01,
		WarmupSteps:        2,
		DecaySteps:         2,
		MinLearningRate:    0.001,
		scheduleStepOffset: 10,
	}
	want := map[int]float64{
		11: 0.005,
		12: 0.01,
		13: 0.0055,
		14: 0.001,
		20: 0.001,
	}
	for step, expected := range want {
		if got := effectiveLearningRate(cfg, step); math.Abs(got-expected) > 1e-12 {
			t.Fatalf("effectiveLearningRate(step=%d) = %g, want %g", step, got, expected)
		}
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	model, err := NewModel(Config{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 4,
		HiddenDim:    16,
		Seed:         4,
	})
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}
	trainer, err := NewTrainer(model)
	if err != nil {
		t.Fatalf("NewTrainer error: %v", err)
	}
	path := filepath.Join(t.TempDir(), "checkpoint.json")
	if err := SaveCheckpoint(path, trainer); err != nil {
		t.Fatalf("SaveCheckpoint error: %v", err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint error: %v", err)
	}
	logitsA, err := trainer.Model.Forward([]int{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Forward(original) error: %v", err)
	}
	logitsB, err := loaded.Model.Forward([]int{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Forward(loaded) error: %v", err)
	}
	if len(logitsA) != len(logitsB) {
		t.Fatalf("loaded logits len = %d, want %d", len(logitsB), len(logitsA))
	}
	for i := range logitsA {
		if logitsA[i] != logitsB[i] {
			t.Fatalf("logit[%d] = %f, want %f", i, logitsB[i], logitsA[i])
		}
	}
}

func byteTok() *byteTokenizer {
	return &byteTokenizer{}
}

type byteTokenizer struct{}

func (b *byteTokenizer) Encode(text string) ([]int, error) {
	tokens := make([]int, len(text))
	for i := range text {
		tokens[i] = int(text[i])
	}
	return tokens, nil
}

func (b *byteTokenizer) Decode(tokens []int) (string, error) {
	bytes := make([]byte, len(tokens))
	for i, token := range tokens {
		bytes[i] = byte(token)
	}
	return string(bytes), nil
}

func (b *byteTokenizer) VocabSize() int {
	return 256
}
