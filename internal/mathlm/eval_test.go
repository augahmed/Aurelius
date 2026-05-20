package mathlm

import (
	"testing"

	"github.com/augahmed/aurelius/internal/arithmetic"
)

func TestEvaluateExamples(t *testing.T) {
	model, err := NewModel(Config{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 8,
		HiddenDim:    32,
		Seed:         5,
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
		{Prompt: "3 + 1 = ", Completion: "4", Operation: "add", Level: 1},
	}
	sequences, err := arithmetic.BuildTrainingSequences(examples, byteTok(), 8)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}
	if _, err := trainer.Train(sequences, sequences, TrainingConfig{
		Epochs:       30,
		BatchSize:    4,
		LearningRate: 0.02,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Seed:         6,
	}); err != nil {
		t.Fatalf("Train error: %v", err)
	}
	report, err := EvaluateExamples(model, examples, 2)
	if err != nil {
		t.Fatalf("EvaluateExamples error: %v", err)
	}
	if report.Total != 2 {
		t.Fatalf("report.Total = %d, want 2", report.Total)
	}
	if report.ByOperation["add"].Total != 2 {
		t.Fatalf("operation add total = %d, want 2", report.ByOperation["add"].Total)
	}
	if report.ByLevel[1].Total != 2 {
		t.Fatalf("level 1 total = %d, want 2", report.ByLevel[1].Total)
	}
}
