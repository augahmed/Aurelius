package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

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
		case "serve":
			return runServe(args[1:], stderr)
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

func buildEngine() (*runtime.Engine, error) {
	tok := tokenizer.NewByteTokenizer()
	model, err := transformer.NewTinyTransformer(transformer.DefaultTinyConfig(tok.VocabSize()))
	if err != nil {
		return nil, err
	}
	return runtime.NewEngine(tok, model, sampler.NewGreedySampler())
}
