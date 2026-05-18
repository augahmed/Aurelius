package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/augahmed/aurelius/internal/gpt2"
	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/server"
	"github.com/augahmed/aurelius/internal/tokenizer"
	"github.com/augahmed/aurelius/internal/transformer"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout io.Writer, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "generate":
			return runGenerate(args[1:], stdout, stderr)
		case "generate-gpt2":
			return runGenerateGPT2(args[1:], stdout, stderr)
		case "emit-gpt2-observation":
			return runEmitGPT2Observation(args[1:], stdout, stderr)
		case "inspect-gpt2-next":
			return runInspectGPT2Next(args[1:], stdout, stderr)
		case "validate-gpt2":
			return runValidateGPT2(args[1:], stdout, stderr)
		case "serve":
			return runServe(args[1:], stderr)
		case "tokenize":
			return runTokenize(args[1:], stdout, stderr)
		case "inspect-model":
			return runInspectModel(args[1:], stdout, stderr)
		}
	}
	return runGenerate(args, stdout, stderr)
}

func runGenerate(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("generate", flag.ContinueOnError)
	flags.SetOutput(stderr)

	prompt := flags.String("prompt", "", "prompt text to generate from")
	maxTokens := flags.Int("max-tokens", 10, "number of tokens to generate")
	useCache := flags.Bool("use-cache", false, "use model KV cache when the selected model supports it")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *prompt == "" {
		fmt.Fprintln(stderr, "prompt is required")
		flags.Usage()
		return 1
	}

	engine, err := buildEngine()
	if err != nil {
		fmt.Fprintf(stderr, "create engine: %v\n", err)
		return 1
	}
	output, err := engine.GenerateWithOptions(*prompt, runtime.GenerateOptions{
		MaxTokens: *maxTokens,
		UseCache:  *useCache,
	})
	if err != nil {
		fmt.Fprintf(stderr, "generate: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, output)
	return 0
}

func runGenerateGPT2(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("generate-gpt2", flag.ContinueOnError)
	flags.SetOutput(stderr)

	configPath := flags.String("model-config", "", "path to GPT-2 config.json")
	weightsPath := flags.String("weights", "", "path to GPT-2 model.safetensors")
	vocabPath := flags.String("vocab", "", "path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "path to GPT-2 merges.txt")
	prompt := flags.String("prompt", "", "prompt text to generate from")
	maxTokens := flags.Int("max-tokens", 1, "number of tokens to generate")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *configPath == "" || *weightsPath == "" || *vocabPath == "" || *mergesPath == "" {
		fmt.Fprintln(stderr, "model-config, weights, vocab, and merges are required")
		flags.Usage()
		return 1
	}
	if *prompt == "" {
		fmt.Fprintln(stderr, "prompt is required")
		flags.Usage()
		return 1
	}

	assets, err := loadGPT2Assets(*configPath, *weightsPath, *vocabPath, *mergesPath)
	if err != nil {
		fmt.Fprintf(stderr, "load GPT-2 assets: %v\n", err)
		return 1
	}
	engine, err := runtime.NewEngine(assets.tokenizer, assets.model, sampler.NewGreedySampler())
	if err != nil {
		fmt.Fprintf(stderr, "create GPT-2 engine: %v\n", err)
		return 1
	}

	stopTokens := gpt2StopTokens(assets.config)
	output, err := engine.GenerateWithOptions(*prompt, runtime.GenerateOptions{
		MaxTokens:  *maxTokens,
		StopTokens: stopTokens,
	})
	if err != nil {
		fmt.Fprintf(stderr, "generate-gpt2: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, output)
	return 0
}

func runEmitGPT2Observation(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("emit-gpt2-observation", flag.ContinueOnError)
	flags.SetOutput(stderr)

	configPath := flags.String("model-config", "", "path to GPT-2 config.json")
	weightsPath := flags.String("weights", "", "path to GPT-2 model.safetensors")
	vocabPath := flags.String("vocab", "", "path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "path to GPT-2 merges.txt")
	prompt := flags.String("prompt", "", "prompt text to inspect")
	topK := flags.Int("top-k", 5, "number of top tokens to include")
	name := flags.String("name", "", "optional fixture name")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *configPath == "" || *weightsPath == "" || *vocabPath == "" || *mergesPath == "" {
		fmt.Fprintln(stderr, "model-config, weights, vocab, and merges are required")
		flags.Usage()
		return 1
	}
	if *prompt == "" {
		fmt.Fprintln(stderr, "prompt is required")
		flags.Usage()
		return 1
	}
	if *topK <= 0 {
		fmt.Fprintln(stderr, "top-k must be positive")
		flags.Usage()
		return 1
	}

	assets, err := loadGPT2Assets(*configPath, *weightsPath, *vocabPath, *mergesPath)
	if err != nil {
		fmt.Fprintf(stderr, "load GPT-2 assets: %v\n", err)
		return 1
	}
	observation, err := gpt2.BuildObservation(*prompt, assets.tokenizer, assets.model, *topK)
	if err != nil {
		fmt.Fprintf(stderr, "emit-gpt2-observation: %v\n", err)
		return 1
	}
	observation.Name = *name
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(observation); err != nil {
		fmt.Fprintf(stderr, "encode observation: %v\n", err)
		return 1
	}
	return 0
}

func runInspectGPT2Next(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("inspect-gpt2-next", flag.ContinueOnError)
	flags.SetOutput(stderr)

	configPath := flags.String("model-config", "", "path to GPT-2 config.json")
	weightsPath := flags.String("weights", "", "path to GPT-2 model.safetensors")
	vocabPath := flags.String("vocab", "", "path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "path to GPT-2 merges.txt")
	prompt := flags.String("prompt", "", "prompt text to inspect")
	topK := flags.Int("top-k", 5, "number of next-token candidates to print")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *configPath == "" || *weightsPath == "" || *vocabPath == "" || *mergesPath == "" {
		fmt.Fprintln(stderr, "model-config, weights, vocab, and merges are required")
		flags.Usage()
		return 1
	}
	if *prompt == "" {
		fmt.Fprintln(stderr, "prompt is required")
		flags.Usage()
		return 1
	}
	if *topK <= 0 {
		fmt.Fprintln(stderr, "top-k must be positive")
		flags.Usage()
		return 1
	}

	assets, err := loadGPT2Assets(*configPath, *weightsPath, *vocabPath, *mergesPath)
	if err != nil {
		fmt.Fprintf(stderr, "load GPT-2 assets: %v\n", err)
		return 1
	}

	input, err := assets.tokenizer.Encode(*prompt)
	if err != nil {
		fmt.Fprintf(stderr, "encode prompt: %v\n", err)
		return 1
	}
	logits, err := assets.model.Forward(input, nil)
	if err != nil {
		fmt.Fprintf(stderr, "forward prompt: %v\n", err)
		return 1
	}

	top := gpt2.TopTokenScores(logits, *topK)
	fmt.Fprintf(stdout, "prompt_tokens: %v\n", input)
	for i, score := range top {
		tokenText, err := assets.tokenizer.Decode([]int{score.Token})
		if err != nil {
			tokenText = "<decode-error>"
		}
		fmt.Fprintf(stdout, "%d: token=%d logit=%.8f text=%q\n", i+1, score.Token, score.Logit, tokenText)
	}
	return 0
}

func runValidateGPT2(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("validate-gpt2", flag.ContinueOnError)
	flags.SetOutput(stderr)

	configPath := flags.String("model-config", "", "path to GPT-2 config.json")
	weightsPath := flags.String("weights", "", "path to GPT-2 model.safetensors")
	vocabPath := flags.String("vocab", "", "path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "path to GPT-2 merges.txt")
	fixturePath := flags.String("fixture", "", "path to GPT-2 parity fixture json")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *configPath == "" || *weightsPath == "" || *vocabPath == "" || *mergesPath == "" || *fixturePath == "" {
		fmt.Fprintln(stderr, "model-config, weights, vocab, merges, and fixture are required")
		flags.Usage()
		return 1
	}

	assets, err := loadGPT2Assets(*configPath, *weightsPath, *vocabPath, *mergesPath)
	if err != nil {
		fmt.Fprintf(stderr, "load GPT-2 assets: %v\n", err)
		return 1
	}
	fixture, err := gpt2.LoadParityFixture(*fixturePath)
	if err != nil {
		fmt.Fprintf(stderr, "load GPT-2 fixture: %v\n", err)
		return 1
	}
	if err := gpt2.ValidateParityFixture(fixture, assets.tokenizer, assets.model); err != nil {
		fmt.Fprintf(stderr, "validate-gpt2: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "validated fixture %q\n", fixture.Name)
	return 0
}

func runServe(args []string, stderr io.Writer) int {
	flags := flag.NewFlagSet("serve", flag.ContinueOnError)
	flags.SetOutput(stderr)

	addr := flags.String("addr", "localhost:8080", "address for the Aurelius web server")
	backend := flags.String("backend", "auto", "model backend: auto, toy, or gpt2")
	configPath := flags.String("model-config", "", "path to GPT-2 config.json")
	weightsPath := flags.String("weights", "", "path to GPT-2 model.safetensors")
	vocabPath := flags.String("vocab", "", "path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "path to GPT-2 merges.txt")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(stderr, "get working directory: %v\n", err)
		return 1
	}

	generator, selectedBackend, err := buildServeGenerator(*backend, cwd, gpt2AssetPaths{
		configPath:  *configPath,
		weightsPath: *weightsPath,
		vocabPath:   *vocabPath,
		mergesPath:  *mergesPath,
	})
	if err != nil {
		fmt.Fprintf(stderr, "create engine: %v\n", err)
		return 1
	}

	app := server.New(generator, server.WithGeneratePolicy(serveGeneratePolicy(selectedBackend)))
	log.Printf("Aurelius server listening on http://%s using %s backend", *addr, selectedBackend)
	if err := http.ListenAndServe(*addr, app.Handler()); err != nil {
		fmt.Fprintf(stderr, "serve: %v\n", err)
		return 1
	}
	return 0
}

func runTokenize(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("tokenize", flag.ContinueOnError)
	flags.SetOutput(stderr)

	vocabPath := flags.String("vocab", "", "path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "path to GPT-2 merges.txt")
	text := flags.String("text", "", "text to tokenize")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *vocabPath == "" || *mergesPath == "" {
		fmt.Fprintln(stderr, "vocab and merges are required")
		flags.Usage()
		return 1
	}
	if *text == "" {
		fmt.Fprintln(stderr, "text is required")
		flags.Usage()
		return 1
	}

	tok, err := tokenizer.LoadBPETokenizer(*vocabPath, *mergesPath)
	if err != nil {
		fmt.Fprintf(stderr, "load tokenizer: %v\n", err)
		return 1
	}

	tokens, err := tok.Encode(*text)
	if err != nil {
		fmt.Fprintf(stderr, "encode: %v\n", err)
		return 1
	}
	decoded, err := tok.Decode(tokens)
	if err != nil {
		fmt.Fprintf(stderr, "decode: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "tokens: %v\n", tokens)
	fmt.Fprintf(stdout, "decoded: %s\n", decoded)
	return 0
}

func runInspectModel(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("inspect-model", flag.ContinueOnError)
	flags.SetOutput(stderr)

	configPath := flags.String("model-config", "", "path to GPT-2 config.json")
	vocabPath := flags.String("vocab", "", "optional path to GPT-2 vocab.json")
	mergesPath := flags.String("merges", "", "optional path to GPT-2 merges.txt")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	if *configPath == "" {
		fmt.Fprintln(stderr, "model-config is required")
		flags.Usage()
		return 1
	}

	cfg, err := gpt2.LoadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(stderr, "load model config: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "model_type: %s\n", cfg.ModelType)
	fmt.Fprintf(stdout, "vocab_size: %d\n", cfg.VocabSize)
	fmt.Fprintf(stdout, "context_length: %d\n", cfg.ResolvedContextLength())
	fmt.Fprintf(stdout, "embedding_dim: %d\n", cfg.EmbeddingDim)
	fmt.Fprintf(stdout, "num_layers: %d\n", cfg.NumLayers)
	fmt.Fprintf(stdout, "num_heads: %d\n", cfg.NumHeads)
	fmt.Fprintf(stdout, "feed_forward_dim: %d\n", cfg.ResolvedFeedForwardDim())
	fmt.Fprintf(stdout, "bos_token_id: %d\n", cfg.BOSTokenID)
	fmt.Fprintf(stdout, "eos_token_id: %d\n", cfg.EOSTokenID)

	if *vocabPath != "" || *mergesPath != "" {
		if *vocabPath == "" || *mergesPath == "" {
			fmt.Fprintln(stderr, "vocab and merges must be provided together")
			return 1
		}

		tok, err := tokenizer.LoadBPETokenizer(*vocabPath, *mergesPath)
		if err != nil {
			fmt.Fprintf(stderr, "load tokenizer: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "tokenizer_vocab_size: %d\n", tok.VocabSize())
		fmt.Fprintf(stdout, "tokenizer_matches_model_vocab: %t\n", tok.VocabSize() == cfg.VocabSize)
	}

	return 0
}

func buildEngine() (*runtime.Engine, error) {
	tok := tokenizer.NewByteTokenizer()
	model, err := transformer.NewTinyTransformer(transformer.DefaultTinyConfig(tok.VocabSize()))
	if err != nil {
		return nil, err
	}
	return runtime.NewEngine(tok, model, sampler.NewGreedySampler())
}

func buildGPT2Engine(configPath, weightsPath, vocabPath, mergesPath string) (*runtime.Engine, error) {
	assets, err := loadGPT2Assets(configPath, weightsPath, vocabPath, mergesPath)
	if err != nil {
		return nil, err
	}
	return runtime.NewEngine(assets.tokenizer, assets.model, sampler.NewGreedySampler())
}

type gpt2Assets struct {
	config    gpt2.Config
	tokenizer tokenizer.Tokenizer
	model     *gpt2.Model
}

func loadGPT2Assets(configPath, weightsPath, vocabPath, mergesPath string) (gpt2Assets, error) {
	cfg, err := gpt2.LoadConfig(configPath)
	if err != nil {
		return gpt2Assets{}, err
	}
	tok, err := tokenizer.LoadBPETokenizer(vocabPath, mergesPath)
	if err != nil {
		return gpt2Assets{}, err
	}
	model, err := gpt2.LoadModel(configPath, weightsPath)
	if err != nil {
		return gpt2Assets{}, err
	}
	return gpt2Assets{
		config:    cfg,
		tokenizer: tok,
		model:     model,
	}, nil
}

func gpt2StopTokens(cfg gpt2.Config) []int {
	if cfg.EOSTokenID <= 0 || cfg.EOSTokenID >= cfg.VocabSize {
		return nil
	}
	return []int{cfg.EOSTokenID}
}

type gpt2AssetPaths struct {
	configPath  string
	weightsPath string
	vocabPath   string
	mergesPath  string
}

type generateWithOptions interface {
	GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error)
}

type generatorWithDefaults struct {
	underlying generateWithOptions
	stopTokens []int
}

func (g generatorWithDefaults) GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error) {
	options.StopTokens = mergeStopTokens(options.StopTokens, g.stopTokens)
	return g.underlying.GenerateWithOptions(prompt, options)
}

func buildServeGenerator(backend, baseDir string, paths gpt2AssetPaths) (server.Generator, string, error) {
	selectedBackend, err := resolveServeBackend(backend, baseDir, paths)
	if err != nil {
		return nil, "", err
	}

	switch selectedBackend {
	case "toy":
		engine, err := buildEngine()
		if err != nil {
			return nil, "", err
		}
		return engine, selectedBackend, nil
	case "gpt2":
		resolved := resolveGPT2AssetPaths(baseDir, paths)
		assets, err := loadGPT2Assets(resolved.configPath, resolved.weightsPath, resolved.vocabPath, resolved.mergesPath)
		if err != nil {
			return nil, "", err
		}
		engine, err := runtime.NewEngine(assets.tokenizer, assets.model, sampler.NewGreedySampler())
		if err != nil {
			return nil, "", err
		}
		return generatorWithDefaults{
			underlying: engine,
			stopTokens: gpt2StopTokens(assets.config),
		}, selectedBackend, nil
	default:
		return nil, "", fmt.Errorf("unsupported backend %q", selectedBackend)
	}
}

func resolveServeBackend(requested, baseDir string, paths gpt2AssetPaths) (string, error) {
	switch requested {
	case "toy", "gpt2":
		return requested, nil
	case "auto", "":
		resolved := resolveGPT2AssetPaths(baseDir, paths)
		if hasGPT2Assets(resolved) {
			return "gpt2", nil
		}
		return "toy", nil
	default:
		return "", fmt.Errorf("backend must be one of auto, toy, or gpt2")
	}
}

func resolveGPT2AssetPaths(baseDir string, paths gpt2AssetPaths) gpt2AssetPaths {
	defaultDir := filepath.Join(baseDir, "artifacts", "gpt2")
	if paths.configPath == "" {
		paths.configPath = filepath.Join(defaultDir, "config.json")
	}
	if paths.weightsPath == "" {
		paths.weightsPath = filepath.Join(defaultDir, "model.safetensors")
	}
	if paths.vocabPath == "" {
		paths.vocabPath = filepath.Join(defaultDir, "vocab.json")
	}
	if paths.mergesPath == "" {
		paths.mergesPath = filepath.Join(defaultDir, "merges.txt")
	}
	return paths
}

func hasGPT2Assets(paths gpt2AssetPaths) bool {
	for _, path := range []string{paths.configPath, paths.weightsPath, paths.vocabPath, paths.mergesPath} {
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			return false
		}
	}
	return true
}

func mergeStopTokens(existing, defaults []int) []int {
	if len(defaults) == 0 {
		return existing
	}
	merged := append([]int(nil), existing...)
	for _, candidate := range defaults {
		seen := false
		for _, current := range merged {
			if current == candidate {
				seen = true
				break
			}
		}
		if !seen {
			merged = append(merged, candidate)
		}
	}
	return merged
}

func serveGeneratePolicy(backend string) server.GeneratePolicy {
	switch backend {
	case "gpt2":
		return server.GeneratePolicy{
			DefaultMaxTokens: 2,
			MaxTokensCap:     2,
			MaxMessages:      6,
			MaxMessageRunes:  240,
			MaxPromptRunes:   480,
			DisableCache:     true,
		}
	default:
		return server.GeneratePolicy{}
	}
}
