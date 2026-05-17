package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/tokenizer"
	"github.com/augahmed/aurelius/internal/transformer"
)

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "generate" {
		args = args[1:]
	}

	prompt := flag.String("prompt", "", "prompt text to generate from")
	maxTokens := flag.Int("max-tokens", 10, "number of tokens to generate")
	useCache := flag.Bool("use-cache", false, "use model KV cache when the selected model supports it")
	if err := flag.CommandLine.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "parse flags: %v\n", err)
		os.Exit(2)
	}

	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "prompt is required")
		flag.Usage()
		os.Exit(1)
	}

	tok := tokenizer.NewByteTokenizer()
	model, err := transformer.NewTinyTransformer(transformer.DefaultTinyConfig(tok.VocabSize()))
	if err != nil {
		fmt.Fprintf(os.Stderr, "create model: %v\n", err)
		os.Exit(1)
	}
	engine, err := runtime.NewEngine(tok, model, sampler.NewGreedySampler())
	if err != nil {
		fmt.Fprintf(os.Stderr, "create engine: %v\n", err)
		os.Exit(1)
	}
	output, err := engine.GenerateWithOptions(*prompt, runtime.GenerateOptions{
		MaxTokens: *maxTokens,
		UseCache:  *useCache,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(output)
}
