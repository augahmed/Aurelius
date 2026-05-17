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
	prompt := flag.String("prompt", "", "prompt text to generate from")
	maxTokens := flag.Int("max-tokens", 10, "number of tokens to generate")
	flag.Parse()

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
	output, err := engine.Generate(*prompt, *maxTokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(output)
}
