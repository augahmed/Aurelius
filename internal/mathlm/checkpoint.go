package mathlm

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/augahmed/aurelius/internal/arithmetic"
	sharedmodel "github.com/augahmed/aurelius/internal/model"
)

type Checkpoint struct {
	ModelType   string              `json:"model_type,omitempty"`
	Trainer     *Trainer            `json:"trainer,omitempty"`
	Transformer *TransformerTrainer `json:"transformer,omitempty"`
}

type AnyTrainer struct {
	ModelType   string
	MLP         *Trainer
	Transformer *TransformerTrainer
}

func NewMLPAnyTrainer(trainer *Trainer) (*AnyTrainer, error) {
	if trainer == nil || trainer.Model == nil {
		return nil, fmt.Errorf("mlp trainer with model is required")
	}
	return &AnyTrainer{ModelType: "mlp", MLP: trainer}, nil
}

func NewTransformerAnyTrainer(trainer *TransformerTrainer) (*AnyTrainer, error) {
	if trainer == nil || trainer.Model == nil {
		return nil, fmt.Errorf("transformer trainer with model is required")
	}
	return &AnyTrainer{ModelType: "transformer", Transformer: trainer}, nil
}

func (t *AnyTrainer) Model() sharedmodel.Model {
	if t == nil {
		return nil
	}
	switch t.ModelType {
	case "mlp":
		if t.MLP == nil {
			return nil
		}
		return t.MLP.Model
	case "transformer":
		if t.Transformer == nil {
			return nil
		}
		return t.Transformer.Model
	default:
		return nil
	}
}

func (t *AnyTrainer) Train(train, val []arithmetic.SequenceExample, cfg TrainingConfig) (TrainingReport, error) {
	switch t.ModelType {
	case "mlp":
		return t.MLP.Train(train, val, cfg)
	case "transformer":
		return t.Transformer.Train(train, val, cfg)
	default:
		return TrainingReport{}, fmt.Errorf("unsupported model type %q", t.ModelType)
	}
}

func SaveCheckpoint(path string, trainer *Trainer) error {
	if trainer == nil || trainer.Model == nil {
		return fmt.Errorf("trainer with model is required")
	}
	data, err := json.MarshalIndent(Checkpoint{ModelType: "mlp", Trainer: trainer}, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal checkpoint: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write checkpoint: %w", err)
	}
	return nil
}

func SaveAnyCheckpoint(path string, trainer *AnyTrainer) error {
	if trainer == nil {
		return fmt.Errorf("trainer is required")
	}
	checkpoint := Checkpoint{ModelType: trainer.ModelType}
	switch trainer.ModelType {
	case "mlp":
		if trainer.MLP == nil || trainer.MLP.Model == nil {
			return fmt.Errorf("mlp trainer with model is required")
		}
		checkpoint.Trainer = trainer.MLP
	case "transformer":
		if trainer.Transformer == nil || trainer.Transformer.Model == nil {
			return fmt.Errorf("transformer trainer with model is required")
		}
		checkpoint.Transformer = trainer.Transformer
	default:
		return fmt.Errorf("unsupported model type %q", trainer.ModelType)
	}
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal checkpoint: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write checkpoint: %w", err)
	}
	return nil
}

func LoadCheckpoint(path string) (*Trainer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read checkpoint: %w", err)
	}
	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("parse checkpoint: %w", err)
	}
	if checkpoint.Trainer == nil || checkpoint.Trainer.Model == nil {
		return nil, fmt.Errorf("checkpoint missing trainer model")
	}
	if err := checkpoint.Trainer.Model.ModelConfig().Validate(); err != nil {
		return nil, fmt.Errorf("invalid checkpoint model config: %w", err)
	}
	if checkpoint.Trainer.Adam == nil {
		checkpoint.Trainer.Adam = newOptimizerState(checkpoint.Trainer.Model)
	}
	return checkpoint.Trainer, nil
}

func LoadAnyCheckpoint(path string) (*AnyTrainer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read checkpoint: %w", err)
	}
	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("parse checkpoint: %w", err)
	}
	modelType := checkpoint.ModelType
	if modelType == "" {
		modelType = "mlp"
	}
	switch modelType {
	case "mlp":
		if checkpoint.Trainer == nil || checkpoint.Trainer.Model == nil {
			return nil, fmt.Errorf("checkpoint missing mlp trainer model")
		}
		if err := checkpoint.Trainer.Model.ModelConfig().Validate(); err != nil {
			return nil, fmt.Errorf("invalid checkpoint model config: %w", err)
		}
		if checkpoint.Trainer.Adam == nil {
			checkpoint.Trainer.Adam = newOptimizerState(checkpoint.Trainer.Model)
		}
		return NewMLPAnyTrainer(checkpoint.Trainer)
	case "transformer":
		if checkpoint.Transformer == nil || checkpoint.Transformer.Model == nil {
			return nil, fmt.Errorf("checkpoint missing transformer trainer model")
		}
		checkpoint.Transformer.Model.ensureLayerBlocks()
		if err := checkpoint.Transformer.Model.LMConfig.Validate(); err != nil {
			return nil, fmt.Errorf("invalid checkpoint transformer config: %w", err)
		}
		checkpoint.Transformer.Adam = ensureTransformerOptimizerState(checkpoint.Transformer.Model, checkpoint.Transformer.Adam)
		return NewTransformerAnyTrainer(checkpoint.Transformer)
	default:
		return nil, fmt.Errorf("unsupported model type %q", modelType)
	}
}
