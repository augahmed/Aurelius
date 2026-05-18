package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

type gpt2TestAssets struct {
	vocabPath  string
	mergesPath string
	configPath string
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
