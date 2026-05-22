package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/arithmetic"
	"github.com/augahmed/aurelius/internal/gpt2"
	"github.com/augahmed/aurelius/internal/mathlm"
	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/server"
)

func TestRunTokenize(t *testing.T) {
	assets := writeGPT2TestAssets(t)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"tokenize",
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
		"-text", "hello!",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "tokens: [7 8]") {
		t.Fatalf("stdout = %q, want token ids", stdout.String())
	}
	if !strings.Contains(stdout.String(), "decoded: hello!") {
		t.Fatalf("stdout = %q, want decoded text", stdout.String())
	}
}

func TestRunInspectModel(t *testing.T) {
	assets := writeGPT2TestAssets(t)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"inspect-model",
		"-model-config", assets.configPath,
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "model_type: gpt2") {
		t.Fatalf("stdout = %q, want model type", stdout.String())
	}
	if !strings.Contains(stdout.String(), "tokenizer_vocab_size: 11") {
		t.Fatalf("stdout = %q, want tokenizer vocab size", stdout.String())
	}
}

func TestRunGenerateGPT2(t *testing.T) {
	assets := writeGPT2ModelAssets(t)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"generate-gpt2",
		"-model-config", assets.configPath,
		"-weights", assets.weightsPath,
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
		"-prompt", "hello!",
		"-max-tokens", "1",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.HasPrefix(stdout.String(), "hello!") {
		t.Fatalf("stdout = %q, want GPT-2 generation output", stdout.String())
	}
}

func TestRunGenerateGPT2WithCache(t *testing.T) {
	assets := writeGPT2ModelAssets(t)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"generate-gpt2",
		"-model-config", assets.configPath,
		"-weights", assets.weightsPath,
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
		"-prompt", "hello!",
		"-max-tokens", "1",
		"-use-cache",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.HasPrefix(stdout.String(), "hello!") {
		t.Fatalf("stdout = %q, want GPT-2 generation output", stdout.String())
	}
}

func TestRunEmitGPT2Observation(t *testing.T) {
	assets := writeGPT2ModelAssets(t)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"emit-gpt2-observation",
		"-model-config", assets.configPath,
		"-weights", assets.weightsPath,
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
		"-prompt", "hello!",
		"-top-k", "3",
		"-name", "fixture-name",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	var fixture gpt2.ParityFixture
	if err := json.Unmarshal(stdout.Bytes(), &fixture); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if fixture.Name != "fixture-name" {
		t.Fatalf("fixture.Name = %q, want %q", fixture.Name, "fixture-name")
	}
	if fixture.Prompt != "hello!" {
		t.Fatalf("fixture.Prompt = %q, want %q", fixture.Prompt, "hello!")
	}
	if len(fixture.ExpectedTopTokens) != 3 {
		t.Fatalf("len(fixture.ExpectedTopTokens) = %d, want %d", len(fixture.ExpectedTopTokens), 3)
	}
}

func TestRunInspectGPT2Next(t *testing.T) {
	assets := writeGPT2ModelAssets(t)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"inspect-gpt2-next",
		"-model-config", assets.configPath,
		"-weights", assets.weightsPath,
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
		"-prompt", "hello!",
		"-top-k", "3",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "prompt_tokens: [7 8]") {
		t.Fatalf("stdout = %q, want prompt tokens", stdout.String())
	}
	if !strings.Contains(stdout.String(), "1: token=") {
		t.Fatalf("stdout = %q, want ranked token output", stdout.String())
	}
}

func TestRunValidateGPT2(t *testing.T) {
	assets := writeGPT2ModelAssets(t)
	fixturePath := filepath.Join(filepath.Dir(assets.configPath), "fixture.json")
	if err := os.WriteFile(fixturePath, []byte(`{
  "name": "tiny-gpt2-hello",
  "prompt": "hello!",
  "expected_input_tokens": [7, 8],
  "expected_top_tokens": [
    {"token": 8, "logit": 0.81598435},
    {"token": 5, "logit": 0.73998573},
    {"token": 3, "logit": 0.27599468}
  ],
  "logit_tolerance": 0.00001
}`), 0o644); err != nil {
		t.Fatalf("WriteFile fixture error: %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"validate-gpt2",
		"-model-config", assets.configPath,
		"-weights", assets.weightsPath,
		"-vocab", assets.vocabPath,
		"-merges", assets.mergesPath,
		"-fixture", fixturePath,
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `validated fixture "tiny-gpt2-hello"`) {
		t.Fatalf("stdout = %q, want validation success", stdout.String())
	}
}

func TestResolveServeBackendAutoUsesGPT2WhenAssetsExist(t *testing.T) {
	baseDir := t.TempDir()
	writeGPT2ModelAssetsAt(t, filepath.Join(baseDir, "artifacts", "gpt2"))

	got, err := resolveServeBackend("auto", baseDir, gpt2AssetPaths{})
	if err != nil {
		t.Fatalf("resolveServeBackend error: %v", err)
	}
	if got != "gpt2" {
		t.Fatalf("resolveServeBackend() = %q, want %q", got, "gpt2")
	}
}

func TestResolveServeBackendAutoFallsBackToToy(t *testing.T) {
	got, err := resolveServeBackend("auto", t.TempDir(), gpt2AssetPaths{})
	if err != nil {
		t.Fatalf("resolveServeBackend error: %v", err)
	}
	if got != "toy" {
		t.Fatalf("resolveServeBackend() = %q, want %q", got, "toy")
	}
}

func TestBuildServeGeneratorUsesStopTokenDefaults(t *testing.T) {
	fake := &recordingGenerator{}
	wrapped := generatorWithDefaults{
		underlying: fake,
		stopTokens: []int{50256},
	}

	_, err := wrapped.GenerateWithOptions("prompt", runtime.GenerateOptions{
		MaxTokens:  1,
		StopTokens: []int{7},
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if !reflect.DeepEqual(fake.options.StopTokens, []int{7, 50256}) {
		t.Fatalf("stop tokens = %v, want %v", fake.options.StopTokens, []int{7, 50256})
	}
}

func TestServeGeneratePolicyForGPT2(t *testing.T) {
	got := serveGeneratePolicy("gpt2")
	want := server.GeneratePolicy{
		DefaultMaxTokens:   8,
		MaxTokensCap:       12,
		DefaultTemperature: 0.8,
		MinTemperature:     0.2,
		MaxTemperature:     1.2,
		DefaultTopK:        40,
		MaxTopK:            80,
		MaxMessages:        6,
		MaxMessageRunes:    240,
		MaxPromptRunes:     480,
		AssistantPreamble:  "You are a helpful assistant. Answer directly and completely.",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("serveGeneratePolicy() = %+v, want %+v", got, want)
	}
}

func TestServeGeneratePolicyForToy(t *testing.T) {
	got := serveGeneratePolicy("toy")
	if !reflect.DeepEqual(got, server.GeneratePolicy{}) {
		t.Fatalf("serveGeneratePolicy() = %+v, want zero policy", got)
	}
}

func TestRunEvalMathDebugErrors(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := filepath.Join(dir, "math.json")
	dataPath := filepath.Join(dir, "val.jsonl")
	errorsPath := filepath.Join(dir, "errors.json")

	model, err := mathlm.NewModel(mathlm.Config{
		VocabSize:    256,
		ContextSize:  8,
		EmbeddingDim: 8,
		HiddenDim:    16,
		Seed:         21,
	})
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}
	trainer, err := mathlm.NewTrainer(model)
	if err != nil {
		t.Fatalf("NewTrainer error: %v", err)
	}
	if err := mathlm.SaveCheckpoint(checkpointPath, trainer); err != nil {
		t.Fatalf("SaveCheckpoint error: %v", err)
	}

	example := arithmetic.Example{
		Prompt:          "2 + 2 = ",
		Completion:      "4",
		Operation:       "add",
		Level:           1,
		Template:        "equation",
		AnswerDigits:    1,
		SmallDifference: true,
		RequiresCarry:   false,
		RequiresBorrow:  false,
		MinOperand:      0,
		MaxOperand:      9,
	}
	data, err := json.Marshal(example)
	if err != nil {
		t.Fatalf("Marshal example error: %v", err)
	}
	if err := os.WriteFile(dataPath, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("WriteFile dataset error: %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"eval-math",
		"-checkpoint", checkpointPath,
		"-data", dataPath,
		"-show-errors", "1",
		"-errors-out", errorsPath,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "template[equation]=") {
		t.Fatalf("stdout = %q, want template debug output", stdout.String())
	}
	if !strings.Contains(stdout.String(), "answer_digits[1]=") {
		t.Fatalf("stdout = %q, want answer digit debug output", stdout.String())
	}
	if !strings.Contains(stdout.String(), "small_difference[true]=") {
		t.Fatalf("stdout = %q, want small difference debug output", stdout.String())
	}
	if !strings.Contains(stdout.String(), "error[1]") {
		t.Fatalf("stdout = %q, want shown error", stdout.String())
	}
	var payload struct {
		ByTemplate        map[string]mathlm.EvalGroup `json:"by_template"`
		ByAnswerDigits    map[int]mathlm.EvalGroup    `json:"by_answer_digits"`
		BySmallDifference map[string]mathlm.EvalGroup `json:"by_small_difference"`
		Errors            []mathlm.EvalError          `json:"errors"`
	}
	raw, err := os.ReadFile(errorsPath)
	if err != nil {
		t.Fatalf("ReadFile errors error: %v", err)
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("Unmarshal errors error: %v", err)
	}
	if payload.ByTemplate["equation"].Total != 1 {
		t.Fatalf("template total = %d, want 1", payload.ByTemplate["equation"].Total)
	}
	if payload.ByAnswerDigits[1].Total != 1 {
		t.Fatalf("answer digit total = %d, want 1", payload.ByAnswerDigits[1].Total)
	}
	if payload.BySmallDifference["true"].Total != 1 {
		t.Fatalf("small difference total = %d, want 1", payload.BySmallDifference["true"].Total)
	}
	if len(payload.Errors) != 1 || payload.Errors[0].Expected != "4" || payload.Errors[0].Prompt != "2 + 2 = " {
		t.Fatalf("unexpected errors payload: %+v", payload.Errors)
	}
}

func TestRunMixMathData(t *testing.T) {
	root := t.TempDir()
	sourceA := filepath.Join(root, "a")
	sourceB := filepath.Join(root, "b")
	output := filepath.Join(root, "mixed")
	if err := arithmetic.GenerateDataset(sourceA, arithmetic.GenerateConfig{
		TrainCount: 2,
		ValCount:   1,
		Operations: []string{"add"},
		Levels:     []int{1},
		Seed:       1,
	}); err != nil {
		t.Fatalf("GenerateDataset(sourceA) error: %v", err)
	}
	if err := arithmetic.GenerateDataset(sourceB, arithmetic.GenerateConfig{
		TrainCount:          2,
		ValCount:            1,
		Operations:          []string{"sub"},
		Levels:              []int{2},
		AnswerDigits:        []int{1},
		SmallDifferenceOnly: true,
		Seed:                2,
	}); err != nil {
		t.Fatalf("GenerateDataset(sourceB) error: %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"mix-math-data",
		"-output-dir", output,
		"-inputs", sourceA + ":1," + sourceB + ":2",
		"-seed", "3",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "wrote mixed arithmetic dataset") {
		t.Fatalf("stdout = %q, want success message", stdout.String())
	}
	train, err := arithmetic.LoadExamples(filepath.Join(output, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples train error: %v", err)
	}
	if len(train) != 6 {
		t.Fatalf("len(train) = %d, want 6", len(train))
	}
}

func TestRunTrainMathWithTrainingControls(t *testing.T) {
	root := t.TempDir()
	dataDir := filepath.Join(root, "data")
	checkpointPath := filepath.Join(root, "math-transformer.json")
	if err := arithmetic.GenerateDataset(dataDir, arithmetic.GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		Operations: []string{"add"},
		Levels:     []int{1},
		Templates:  []string{"equation"},
		MinOperand: 0,
		MaxOperand: 9,
		Seed:       31,
	}); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"train-math",
		"-model", "transformer",
		"-data-dir", dataDir,
		"-checkpoint", checkpointPath,
		"-context-size", "8",
		"-embedding-dim", "8",
		"-hidden-dim", "16",
		"-num-heads", "2",
		"-epochs", "10",
		"-batch-size", "2",
		"-learning-rate", "0.01",
		"-max-steps", "1",
		"-log-every", "1",
		"-save-every", "1",
		"-grad-clip", "1",
		"-seed", "32",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run() exit code = %d, stderr = %q", code, stderr.String())
	}
	output := stdout.String()
	if !strings.Contains(output, "step=1 train_loss=") {
		t.Fatalf("stdout = %q, want progress line", output)
	}
	if !strings.Contains(output, "checkpoint_step=1 path=") {
		t.Fatalf("stdout = %q, want periodic checkpoint line", output)
	}
	if _, err := os.Stat(periodicCheckpointPath(checkpointPath, 1)); err != nil {
		t.Fatalf("periodic checkpoint missing: %v", err)
	}
	loaded, err := mathlm.LoadAnyCheckpoint(checkpointPath)
	if err != nil {
		t.Fatalf("LoadAnyCheckpoint final error: %v", err)
	}
	if loaded.ModelType != "transformer" || loaded.Transformer.Step != 1 {
		t.Fatalf("loaded checkpoint type=%q step=%d, want transformer step 1", loaded.ModelType, loaded.Transformer.Step)
	}

	var resumeStdout bytes.Buffer
	var resumeStderr bytes.Buffer
	code = run([]string{
		"train-math",
		"-model", "transformer",
		"-data-dir", dataDir,
		"-checkpoint", checkpointPath,
		"-resume", checkpointPath,
		"-epochs", "10",
		"-batch-size", "2",
		"-learning-rate", "0.01",
		"-max-steps", "1",
		"-seed", "33",
	}, &resumeStdout, &resumeStderr)
	if code != 0 {
		t.Fatalf("resume run() exit code = %d, stderr = %q", code, resumeStderr.String())
	}
	resumed, err := mathlm.LoadAnyCheckpoint(checkpointPath)
	if err != nil {
		t.Fatalf("LoadAnyCheckpoint resumed error: %v", err)
	}
	if resumed.Transformer.Step != 2 {
		t.Fatalf("resumed transformer step = %d, want 2", resumed.Transformer.Step)
	}
}

type gpt2TestAssets struct {
	vocabPath   string
	mergesPath  string
	configPath  string
	weightsPath string
}

func writeGPT2TestAssets(t *testing.T) gpt2TestAssets {
	t.Helper()

	dir := t.TempDir()
	assets := gpt2TestAssets{
		vocabPath:  filepath.Join(dir, "vocab.json"),
		mergesPath: filepath.Join(dir, "merges.txt"),
		configPath: filepath.Join(dir, "config.json"),
	}

	if err := os.WriteFile(assets.vocabPath, []byte(`{"h":0,"e":1,"l":2,"o":3,"he":4,"hel":5,"hell":6,"hello":7,"!":8,"Ã":9,"©":10}`), 0o644); err != nil {
		t.Fatalf("WriteFile vocab error: %v", err)
	}
	if err := os.WriteFile(assets.mergesPath, []byte("#version: 0.2\nh e\nhe l\nhel l\nhell o\n"), 0o644); err != nil {
		t.Fatalf("WriteFile merges error: %v", err)
	}
	if err := os.WriteFile(assets.configPath, []byte(`{"model_type":"gpt2","vocab_size":11,"n_ctx":32,"n_embd":8,"n_layer":2,"n_head":2}`), 0o644); err != nil {
		t.Fatalf("WriteFile config error: %v", err)
	}

	return assets
}

func writeGPT2ModelAssets(t *testing.T) gpt2TestAssets {
	t.Helper()

	assets := writeGPT2TestAssets(t)
	assets.weightsPath = filepath.Join(filepath.Dir(assets.configPath), "model.safetensors")
	if err := writeSafeTensors(assets.weightsPath, tinyStateDict()); err != nil {
		t.Fatalf("writeSafeTensors error: %v", err)
	}
	if err := os.WriteFile(assets.configPath, []byte(`{"model_type":"gpt2","vocab_size":11,"n_ctx":8,"n_embd":2,"n_layer":1,"n_head":1,"n_inner":3}`), 0o644); err != nil {
		t.Fatalf("WriteFile config error: %v", err)
	}
	return assets
}

func writeGPT2ModelAssetsAt(t *testing.T, dir string) gpt2TestAssets {
	t.Helper()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll error: %v", err)
	}
	assets := gpt2TestAssets{
		vocabPath:   filepath.Join(dir, "vocab.json"),
		mergesPath:  filepath.Join(dir, "merges.txt"),
		configPath:  filepath.Join(dir, "config.json"),
		weightsPath: filepath.Join(dir, "model.safetensors"),
	}
	if err := os.WriteFile(assets.vocabPath, []byte(`{"h":0,"e":1,"l":2,"o":3,"he":4,"hel":5,"hell":6,"hello":7,"!":8,"Ã":9,"©":10}`), 0o644); err != nil {
		t.Fatalf("WriteFile vocab error: %v", err)
	}
	if err := os.WriteFile(assets.mergesPath, []byte("#version: 0.2\nh e\nhe l\nhel l\nhell o\n"), 0o644); err != nil {
		t.Fatalf("WriteFile merges error: %v", err)
	}
	if err := os.WriteFile(assets.configPath, []byte(`{"model_type":"gpt2","vocab_size":11,"n_ctx":8,"n_embd":2,"n_layer":1,"n_head":1,"n_inner":3,"eos_token_id":10}`), 0o644); err != nil {
		t.Fatalf("WriteFile config error: %v", err)
	}
	if err := writeSafeTensors(assets.weightsPath, tinyStateDict()); err != nil {
		t.Fatalf("writeSafeTensors error: %v", err)
	}
	return assets
}

func tinyStateDict() map[string]gpt2.Tensor {
	return map[string]gpt2.Tensor{
		"transformer.wte.weight": tensor2D(11, 2, []float64{
			0.10, 0.00,
			0.00, 0.10,
			0.20, -0.10,
			-0.10, 0.20,
			0.30, 0.40,
			-0.30, 0.50,
			0.40, -0.20,
			0.60, 0.10,
			-0.20, 0.70,
			0.50, -0.40,
			-0.50, -0.30,
		}),
		"transformer.wpe.weight": tensor2D(8, 2, []float64{
			0.01, -0.02,
			0.03, 0.04,
			0.05, -0.01,
			-0.04, 0.02,
			0.02, 0.03,
			-0.02, -0.03,
			0.04, 0.01,
			-0.01, 0.05,
		}),
		"transformer.h.0.ln_1.weight": tensor1D([]float64{1.10, 0.90}),
		"transformer.h.0.ln_1.bias":   tensor1D([]float64{0.01, -0.02}),
		"transformer.h.0.attn.c_attn.weight": tensor2D(2, 6, []float64{
			0.20, -0.10, 0.05, 0.30, -0.20, 0.10,
			0.10, 0.15, -0.05, -0.25, 0.20, 0.05,
		}),
		"transformer.h.0.attn.c_attn.bias": tensor1D([]float64{0.01, -0.02, 0.03, 0.02, -0.01, 0.04}),
		"transformer.h.0.attn.c_proj.weight": tensor2D(2, 2, []float64{
			0.20, -0.30,
			0.40, 0.10,
		}),
		"transformer.h.0.attn.c_proj.bias": tensor1D([]float64{0.01, -0.02}),
		"transformer.h.0.ln_2.weight":      tensor1D([]float64{0.95, 1.05}),
		"transformer.h.0.ln_2.bias":        tensor1D([]float64{-0.03, 0.02}),
		"transformer.h.0.mlp.c_fc.weight": tensor2D(2, 3, []float64{
			0.30, -0.20, 0.10,
			-0.10, 0.25, 0.20,
		}),
		"transformer.h.0.mlp.c_fc.bias": tensor1D([]float64{0.01, -0.02, 0.03}),
		"transformer.h.0.mlp.c_proj.weight": tensor2D(3, 2, []float64{
			0.20, -0.10,
			0.05, 0.30,
			-0.25, 0.15,
		}),
		"transformer.h.0.mlp.c_proj.bias": tensor1D([]float64{0.02, -0.01}),
		"transformer.ln_f.weight":         tensor1D([]float64{1.00, 0.85}),
		"transformer.ln_f.bias":           tensor1D([]float64{0.00, 0.03}),
	}
}

func tensor1D(data []float64) gpt2.Tensor {
	return gpt2.Tensor{Shape: []int{len(data)}, Data: append([]float64(nil), data...)}
}

func tensor2D(rows, cols int, data []float64) gpt2.Tensor {
	return gpt2.Tensor{Shape: []int{rows, cols}, Data: append([]float64(nil), data...)}
}

func writeSafeTensors(path string, tensors map[string]gpt2.Tensor) error {
	type headerEntry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}

	header := make(map[string]headerEntry, len(tensors))
	payload := make([]byte, 0)
	offset := 0
	for name, tensor := range tensors {
		start := offset
		for _, value := range tensor.Data {
			var bytes [4]byte
			binary.LittleEndian.PutUint32(bytes[:], math.Float32bits(float32(value)))
			payload = append(payload, bytes[:]...)
			offset += 4
		}
		header[name] = headerEntry{
			DType:       "F32",
			Shape:       append([]int(nil), tensor.Shape...),
			DataOffsets: []int{start, offset},
		}
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}

	file := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(file[:8], uint64(len(headerBytes)))
	copy(file[8:], headerBytes)
	copy(file[8+len(headerBytes):], payload)
	return os.WriteFile(path, file, 0o644)
}

type recordingGenerator struct {
	options runtime.GenerateOptions
}

func (g *recordingGenerator) GenerateWithOptions(_ string, options runtime.GenerateOptions) (string, error) {
	g.options = options
	return "ok", nil
}
