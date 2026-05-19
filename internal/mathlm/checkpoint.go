package mathlm

import (
	"encoding/json"
	"fmt"
	"os"
)

type Checkpoint struct {
	Trainer *Trainer `json:"trainer"`
}

func SaveCheckpoint(path string, trainer *Trainer) error {
	if trainer == nil || trainer.Model == nil {
		return fmt.Errorf("trainer with model is required")
	}
	data, err := json.MarshalIndent(Checkpoint{Trainer: trainer}, "", "  ")
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
