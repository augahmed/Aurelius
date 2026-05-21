package mathlm

import (
	"path/filepath"
	"testing"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

func TestTransformerForwardShape(t *testing.T) {
	model, err := NewTransformerModel(TransformerConfig{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 8,
		NumHeads:     2,
		MLPDim:       16,
		Seed:         1,
	})
	if err != nil {
		t.Fatalf("NewTransformerModel error: %v", err)
	}
	logits, err := model.Forward([]int{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if len(logits) != 256 {
		t.Fatalf("len(logits) = %d, want 256", len(logits))
	}
}

func TestTransformerCausalMasking(t *testing.T) {
	model, err := NewTransformerModel(TransformerConfig{
		VocabSize:    256,
		ContextSize:  4,
		EmbeddingDim: 8,
		NumHeads:     2,
		MLPDim:       16,
		Seed:         2,
	})
	if err != nil {
		t.Fatalf("NewTransformerModel error: %v", err)
	}
	left, err := model.ForwardAll([]int{10, 11, 12, 13})
	if err != nil {
		t.Fatalf("ForwardAll(left) error: %v", err)
	}
	right, err := model.ForwardAll([]int{10, 11, 12, 99})
	if err != nil {
		t.Fatalf("ForwardAll(right) error: %v", err)
	}
	for token := range left[2] {
		if left[2][token] != right[2][token] {
			t.Fatalf("position 2 logit[%d] changed after future token changed: %f vs %f", token, left[2][token], right[2][token])
		}
	}
}

func TestTransformerTrainingReducesLoss(t *testing.T) {
	model, err := NewTransformerModel(TransformerConfig{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 16,
		NumHeads:     2,
		MLPDim:       32,
		Seed:         3,
	})
	if err != nil {
		t.Fatalf("NewTransformerModel error: %v", err)
	}
	trainer, err := NewTransformerTrainer(model)
	if err != nil {
		t.Fatalf("NewTransformerTrainer error: %v", err)
	}
	examples := []arithmetic.Example{
		{Prompt: "2 + 2 = ", Completion: "4", Operation: "add", Level: 1},
		{Prompt: "3 + 1 = ", Completion: "4", Operation: "add", Level: 1},
	}
	sequences, err := arithmetic.BuildTrainingSequences(examples, byteTok(), 8)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}
	before, err := AverageTransformerLoss(model, sequences)
	if err != nil {
		t.Fatalf("AverageTransformerLoss(before) error: %v", err)
	}
	if _, err := trainer.Train(sequences, sequences, TrainingConfig{
		Epochs:       25,
		BatchSize:    4,
		LearningRate: 0.05,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Seed:         4,
	}); err != nil {
		t.Fatalf("Train error: %v", err)
	}
	after, err := AverageTransformerLoss(model, sequences)
	if err != nil {
		t.Fatalf("AverageTransformerLoss(after) error: %v", err)
	}
	if after >= before {
		t.Fatalf("loss after training = %f, want less than %f", after, before)
	}
}

func TestTransformerTrainingUpdatesBlockWeights(t *testing.T) {
	model, err := NewTransformerModel(TransformerConfig{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 16,
		NumHeads:     2,
		MLPDim:       32,
		Seed:         13,
	})
	if err != nil {
		t.Fatalf("NewTransformerModel error: %v", err)
	}
	trainer, err := NewTransformerTrainer(model)
	if err != nil {
		t.Fatalf("NewTransformerTrainer error: %v", err)
	}
	originalTokenEmbedding := append([]float64(nil), model.TokenEmbeddings...)
	originalQueryWeights := append([]float64(nil), model.QueryWeights...)
	originalMLPInputWeights := append([]float64(nil), model.MLPInputWeights...)

	examples := []arithmetic.Example{
		{Prompt: "2 + 2 = ", Completion: "4", Operation: "add", Level: 1},
		{Prompt: "5 - 2 = ", Completion: "3", Operation: "sub", Level: 1},
	}
	sequences, err := arithmetic.BuildTrainingSequences(examples, byteTok(), 8)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}
	if _, err := trainer.Train(sequences, sequences, TrainingConfig{
		Epochs:       5,
		BatchSize:    4,
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Seed:         14,
	}); err != nil {
		t.Fatalf("Train error: %v", err)
	}
	if slicesEqual(originalTokenEmbedding, model.TokenEmbeddings) {
		t.Fatal("token embeddings did not change")
	}
	if slicesEqual(originalQueryWeights, model.QueryWeights) {
		t.Fatal("query weights did not change")
	}
	if slicesEqual(originalMLPInputWeights, model.MLPInputWeights) {
		t.Fatal("MLP input weights did not change")
	}
}

func TestTransformerCheckpointRoundTrip(t *testing.T) {
	model, err := NewTransformerModel(TransformerConfig{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 8,
		NumHeads:     2,
		MLPDim:       16,
		Seed:         5,
	})
	if err != nil {
		t.Fatalf("NewTransformerModel error: %v", err)
	}
	trainer, err := NewTransformerTrainer(model)
	if err != nil {
		t.Fatalf("NewTransformerTrainer error: %v", err)
	}
	anyTrainer, err := NewTransformerAnyTrainer(trainer)
	if err != nil {
		t.Fatalf("NewTransformerAnyTrainer error: %v", err)
	}
	path := filepath.Join(t.TempDir(), "transformer-checkpoint.json")
	if err := SaveAnyCheckpoint(path, anyTrainer); err != nil {
		t.Fatalf("SaveAnyCheckpoint error: %v", err)
	}
	loaded, err := LoadAnyCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadAnyCheckpoint error: %v", err)
	}
	if loaded.ModelType != "transformer" {
		t.Fatalf("loaded model type = %q, want transformer", loaded.ModelType)
	}
	logitsA, err := anyTrainer.Model().Forward([]int{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Forward(original) error: %v", err)
	}
	logitsB, err := loaded.Model().Forward([]int{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Forward(loaded) error: %v", err)
	}
	for i := range logitsA {
		if logitsA[i] != logitsB[i] {
			t.Fatalf("logit[%d] = %f, want %f", i, logitsB[i], logitsA[i])
		}
	}
}

func slicesEqual(left, right []float64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
