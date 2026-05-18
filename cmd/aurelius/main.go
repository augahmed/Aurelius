package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

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

	engine, err := buildGPT2Engine(*configPath, *weightsPath, *vocabPath, *mergesPath)
	if err != nil {
		fmt.Fprintf(stderr, "create GPT-2 engine: %v\n", err)
		return 1
	}

	output, err := engine.Generate(*prompt, *maxTokens)
	if err != nil {
		fmt.Fprintf(stderr, "generate-gpt2: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, output)
	return 0
}

func runServe(args []string, stderr io.Writer) int {
	flags := flag.NewFlagSet("serve", flag.ContinueOnError)
	flags.SetOutput(stderr)

	addr := flags.String("addr", "localhost:8080", "address for the Aurelius web server")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}

	engine, err := buildEngine()
	if err != nil {
		fmt.Fprintf(stderr, "create engine: %v\n", err)
		return 1
	}

	app := server.New(engine)
	log.Printf("Aurelius server listening on http://%s", *addr)
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
	tok, err := tokenizer.LoadBPETokenizer(vocabPath, mergesPath)
	if err != nil {
		return nil, err
	}
	model, err := gpt2.LoadModel(configPath, weightsPath)
	if err != nil {
		return nil, err
	}
	return runtime.NewEngine(tok, model, sampler.NewGreedySampler())
}
