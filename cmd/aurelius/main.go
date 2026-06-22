package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/augahmed/aurelius/internal/arithmetic"
	"github.com/augahmed/aurelius/internal/gpt2"
	"github.com/augahmed/aurelius/internal/mathlm"
	"github.com/augahmed/aurelius/internal/mathrouter"
	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/sampler"
	"github.com/augahmed/aurelius/internal/server"
	"github.com/augahmed/aurelius/internal/textdata"
	"github.com/augahmed/aurelius/internal/tokenizer"
	"github.com/augahmed/aurelius/internal/transformer"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout io.Writer, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "gen-math-data":
			return runGenerateMathData(args[1:], stdout, stderr)
		case "gen-math-error-replay":
			return runGenerateMathErrorReplay(args[1:], stdout, stderr)
		case "gen-math-instructions":
			return runGenerateMathInstructions(args[1:], stdout, stderr)
		case "mix-math-data":
			return runMixMathData(args[1:], stdout, stderr)
		case "fetch-text-data":
			return runFetchTextData(args[1:], stdout, stderr)
		case "inspect-text-data":
			return runInspectTextData(args[1:], stdout, stderr)
		case "dedupe-text-data":
			return runDedupeTextData(args[1:], stdout, stderr)
		case "split-text-data":
			return runSplitTextData(args[1:], stdout, stderr)
		case "train-math":
			return runTrainMath(args[1:], stdout, stderr)
		case "train-text":
			return runTrainText(args[1:], stdout, stderr)
		case "eval-math":
			return runEvalMath(args[1:], stdout, stderr)
		case "eval-instructions":
			return runEvalInstructions(args[1:], stdout, stderr)
		case "generate-math":
			return runGenerateMath(args[1:], stdout, stderr)
		case "generate-checkpoint":
			return runGenerateCheckpoint(args[1:], stdout, stderr)
		case "export-checkpoint":
			return runExportCheckpoint(args[1:], stdout, stderr)
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
	temperature := flags.Float64("temperature", 0, "sampling temperature; 0 keeps the default sampler")
	topK := flags.Int("top-k", 0, "limit token sampling to the top-k logits; 0 disables top-k sampling")
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
		MaxTokens:   *maxTokens,
		TopK:        *topK,
		UseCache:    *useCache,
		Temperature: *temperature,
	})
	if err != nil {
		fmt.Fprintf(stderr, "generate: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, output)
	return 0
}

func runGenerateMathData(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("gen-math-data", flag.ContinueOnError)
	flags.SetOutput(stderr)

	outputDir := flags.String("output-dir", "", "directory to write train.jsonl, val.jsonl, and meta.json")
	trainCount := flags.Int("train-count", 4000, "number of training examples")
	valCount := flags.Int("val-count", 500, "number of validation examples")
	minOperand := flags.Int("min-operand", 0, "minimum operand value")
	maxOperand := flags.Int("max-operand", 20, "maximum operand value")
	operations := flags.String("operations", "add,sub,mul,div", "comma-separated operations: add,sub,mul,div,word,derivative")
	levels := flags.String("levels", "1,2,3,4,5", "comma-separated curriculum levels: 1,2,3,4,5,6,7")
	templates := flags.String("templates", "all", "comma-separated templates: equation,question,solve, or all")
	reasoningStyle := flags.String("reasoning-style", "direct", "completion style: direct, worked, compact, or coefficients")
	answerDigits := flags.String("answer-digits", "", "optional comma-separated answer digit buckets, for example 1,2")
	smallDifferenceOnly := flags.Bool("small-difference-only", false, "only generate subtraction examples with one-digit differences")
	seed := flags.Int64("seed", 1, "random seed")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *outputDir == "" {
		fmt.Fprintln(stderr, "output-dir is required")
		flags.Usage()
		return 1
	}

	parsedLevels, err := splitInts(*levels)
	if err != nil {
		fmt.Fprintf(stderr, "parse levels: %v\n", err)
		return 1
	}
	parsedAnswerDigits, err := splitInts(*answerDigits)
	if err != nil {
		fmt.Fprintf(stderr, "parse answer-digits: %v\n", err)
		return 1
	}
	parsedTemplates := splitCSV(*templates)
	if len(parsedTemplates) == 1 && parsedTemplates[0] == "all" {
		parsedTemplates = nil
	}

	cfg := arithmetic.GenerateConfig{
		TrainCount:          *trainCount,
		ValCount:            *valCount,
		MinOperand:          *minOperand,
		MaxOperand:          *maxOperand,
		Operations:          splitCSV(*operations),
		Levels:              parsedLevels,
		Templates:           parsedTemplates,
		ReasoningStyle:      *reasoningStyle,
		AnswerDigits:        parsedAnswerDigits,
		SmallDifferenceOnly: *smallDifferenceOnly,
		Seed:                *seed,
	}
	if err := arithmetic.GenerateDataset(*outputDir, cfg); err != nil {
		fmt.Fprintf(stderr, "generate dataset: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote arithmetic dataset to %s\n", *outputDir)
	return 0
}

func runGenerateMathErrorReplay(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("gen-math-error-replay", flag.ContinueOnError)
	flags.SetOutput(stderr)

	errorsPath := flags.String("errors", "", "eval-math errors JSON file produced by -errors-out")
	outputDir := flags.String("output-dir", "", "directory to write replay train.jsonl, val.jsonl, and meta.json")
	repeat := flags.Int("repeat", 3, "number of times to repeat each training error example")
	valRatio := flags.Float64("val-ratio", 0.1, "fraction of unique error examples held out for validation")
	limit := flags.Int("limit", 0, "optional cap on loaded error examples before dedupe; 0 disables")
	seed := flags.Int64("seed", 1, "random seed")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *errorsPath == "" || *outputDir == "" {
		fmt.Fprintln(stderr, "errors and output-dir are required")
		flags.Usage()
		return 1
	}
	if *repeat <= 0 {
		fmt.Fprintln(stderr, "repeat must be positive")
		return 1
	}
	if *valRatio < 0 || *valRatio >= 1 {
		fmt.Fprintln(stderr, "val-ratio must be >= 0 and < 1")
		return 1
	}
	if *limit < 0 {
		fmt.Fprintln(stderr, "limit cannot be negative")
		return 1
	}

	report, err := loadMathEvalErrorReplayReport(*errorsPath)
	if err != nil {
		fmt.Fprintf(stderr, "load errors: %v\n", err)
		return 1
	}
	errors := report.Errors
	if *limit > 0 && *limit < len(errors) {
		errors = errors[:*limit]
	}
	unique := buildErrorReplayExamples(errors)
	if len(unique) == 0 {
		fmt.Fprintln(stderr, "errors file did not contain any replayable examples")
		return 1
	}

	rng := rand.New(rand.NewSource(*seed))
	rng.Shuffle(len(unique), func(i, j int) {
		unique[i], unique[j] = unique[j], unique[i]
	})
	trainUnique, val := splitReplayExamples(unique, *valRatio)
	train := repeatExamples(trainUnique, *repeat)
	rng.Shuffle(len(train), func(i, j int) {
		train[i], train[j] = train[j], train[i]
	})
	if len(val) == 0 {
		val = append([]arithmetic.Example(nil), trainUnique...)
	}

	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "create output dir: %v\n", err)
		return 1
	}
	if err := writeArithmeticJSONL(filepath.Join(*outputDir, "train.jsonl"), train); err != nil {
		fmt.Fprintf(stderr, "write train: %v\n", err)
		return 1
	}
	if err := writeArithmeticJSONL(filepath.Join(*outputDir, "val.jsonl"), val); err != nil {
		fmt.Fprintf(stderr, "write val: %v\n", err)
		return 1
	}
	meta := mathErrorReplayMetadata{
		SourceErrors:     *errorsPath,
		SourceTotal:      report.Total,
		SourceCorrect:    report.Correct,
		SourceAccuracy:   report.Accuracy,
		InputErrors:      len(errors),
		UniqueErrors:     len(unique),
		TrainCount:       len(train),
		ValCount:         len(val),
		Repeat:           *repeat,
		ValRatio:         *valRatio,
		Seed:             *seed,
		AnswerSource:     "router_when_supported_else_expected",
		ReplayCompletion: "direct_final_answer",
	}
	if err := writeJSONFile(filepath.Join(*outputDir, "meta.json"), meta); err != nil {
		fmt.Fprintf(stderr, "write meta: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "wrote math error replay dataset to %s unique_errors=%d train=%d val=%d repeat=%d\n",
		*outputDir,
		len(unique),
		len(train),
		len(val),
		*repeat,
	)
	return 0
}

func runGenerateMathInstructions(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("gen-math-instructions", flag.ContinueOnError)
	flags.SetOutput(stderr)

	dataDir := flags.String("data-dir", "", "directory containing arithmetic train.jsonl and val.jsonl")
	outputDir := flags.String("output-dir", "", "directory to write instruction train.jsonl, val.jsonl, and meta.json")
	systemPrompt := flags.String("system", textdata.DefaultMathInstructionSystem, "system prompt for instruction-format examples")
	format := flags.String("format", textdata.MathInstructionFormatInstruction, "output format: instruction, chat, or compact")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *dataDir == "" || *outputDir == "" {
		fmt.Fprintln(stderr, "data-dir and output-dir are required")
		flags.Usage()
		return 1
	}
	report, err := textdata.GenerateMathInstructionDataset(textdata.MathInstructionConfig{
		DataDir:   *dataDir,
		OutputDir: *outputDir,
		System:    *systemPrompt,
		Format:    *format,
	})
	if err != nil {
		fmt.Fprintf(stderr, "generate math instructions: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote math instruction dataset to %s train=%d val=%d format=%s\n",
		*outputDir,
		report.TrainCount,
		report.ValCount,
		report.Format,
	)
	return 0
}

func runMixMathData(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("mix-math-data", flag.ContinueOnError)
	flags.SetOutput(stderr)

	outputDir := flags.String("output-dir", "", "directory to write mixed train.jsonl, val.jsonl, and meta.json")
	inputs := flags.String("inputs", "", "comma-separated dataset directories with integer weights, for example ./data/base:1,./data/targeted:2")
	seed := flags.Int64("seed", 1, "random seed")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *outputDir == "" || *inputs == "" {
		fmt.Fprintln(stderr, "output-dir and inputs are required")
		flags.Usage()
		return 1
	}
	sources, err := parseMixSources(*inputs)
	if err != nil {
		fmt.Fprintf(stderr, "parse inputs: %v\n", err)
		return 1
	}
	if err := arithmetic.MixDatasets(*outputDir, arithmetic.MixConfig{
		Sources: sources,
		Seed:    *seed,
	}); err != nil {
		fmt.Fprintf(stderr, "mix dataset: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote mixed arithmetic dataset to %s\n", *outputDir)
	return 0
}

func runFetchTextData(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("fetch-text-data", flag.ContinueOnError)
	flags.SetOutput(stderr)

	outputDir := flags.String("output-dir", "", "directory to write cleaned .txt files and sources.jsonl")
	urlsFlag := flags.String("urls", "", "comma-separated http(s) URLs to fetch")
	urlFile := flags.String("url-file", "", "optional file containing one URL per line; # comments allowed")
	maxPages := flags.Int("max-pages", 0, "optional cap on pages fetched; 0 fetches all URLs")
	maxBytes := flags.Int64("max-bytes", 2*1024*1024, "maximum fetched bytes per page")
	timeout := flags.Duration("timeout", 15*time.Second, "per-request timeout, for example 10s")
	userAgent := flags.String("user-agent", "AureliusTextIngest/0.1", "HTTP user agent for page fetches")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *outputDir == "" {
		fmt.Fprintln(stderr, "output-dir is required")
		flags.Usage()
		return 1
	}
	if *maxPages < 0 || *maxBytes <= 0 {
		fmt.Fprintln(stderr, "max-pages must be non-negative and max-bytes must be positive")
		return 1
	}

	urls := splitCSV(*urlsFlag)
	if *urlFile != "" {
		fileURLs, err := textdata.LoadURLList(*urlFile)
		if err != nil {
			fmt.Fprintf(stderr, "load URL file: %v\n", err)
			return 1
		}
		urls = append(urls, fileURLs...)
	}
	if len(urls) == 0 {
		fmt.Fprintln(stderr, "urls or url-file is required")
		flags.Usage()
		return 1
	}

	ctx := context.Background()
	results, err := textdata.IngestWebText(ctx, urls, textdata.WebIngestConfig{
		OutputDir: *outputDir,
		MaxPages:  *maxPages,
		MaxBytes:  *maxBytes,
		Timeout:   *timeout,
		UserAgent: *userAgent,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fetch text data: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote_text_pages=%d output_dir=%s metadata=%s\n", len(results), *outputDir, filepath.Join(*outputDir, "sources.jsonl"))
	for _, result := range results {
		fmt.Fprintf(stdout, "source=%q path=%q text_bytes=%d\n", result.Source, result.Path, result.TextBytes)
	}
	return 0
}

func runInspectTextData(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("inspect-text-data", flag.ContinueOnError)
	flags.SetOutput(stderr)

	textPaths := flags.String("text", "", "comma-separated text files or directories to inspect")
	shortBytes := flags.Int("short-bytes", 256, "flag files below this cleaned byte size as short")
	jsonOut := flags.Bool("json", false, "print full JSON report")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if strings.TrimSpace(*textPaths) == "" {
		fmt.Fprintln(stderr, "text is required")
		flags.Usage()
		return 1
	}
	report, err := textdata.InspectTextDataset(splitCSV(*textPaths), *shortBytes)
	if err != nil {
		fmt.Fprintf(stderr, "inspect text data: %v\n", err)
		return 1
	}
	if *jsonOut {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			fmt.Fprintf(stderr, "write report: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "files=%d total_bytes=%d paragraphs=%d duplicate_paragraphs=%d empty_files=%d short_files=%d\n",
		report.FileCount,
		report.TotalBytes,
		report.TotalParagraphs,
		report.DuplicateParagraphs,
		report.EmptyFiles,
		len(report.ShortFiles),
	)
	for _, file := range report.LargestFiles {
		fmt.Fprintf(stdout, "largest path=%q bytes=%d paragraphs=%d duplicates=%d\n", file.Path, file.Bytes, file.Paragraphs, file.DuplicateParagraphs)
	}
	return 0
}

func runDedupeTextData(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("dedupe-text-data", flag.ContinueOnError)
	flags.SetOutput(stderr)

	textPaths := flags.String("text", "", "comma-separated text files or directories to dedupe")
	outputDir := flags.String("output-dir", "", "directory to write deduped .txt files")
	minParagraphRunes := flags.Int("min-paragraph-runes", 1, "drop paragraphs shorter than this many runes")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if strings.TrimSpace(*textPaths) == "" || *outputDir == "" {
		fmt.Fprintln(stderr, "text and output-dir are required")
		flags.Usage()
		return 1
	}
	if *minParagraphRunes < 0 {
		fmt.Fprintln(stderr, "min-paragraph-runes cannot be negative")
		return 1
	}
	report, err := textdata.DedupeTextDataset(splitCSV(*textPaths), textdata.DedupeConfig{
		OutputDir:         *outputDir,
		MinParagraphRunes: *minParagraphRunes,
	})
	if err != nil {
		fmt.Fprintf(stderr, "dedupe text data: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "input_files=%d output_files=%d input_paragraphs=%d output_paragraphs=%d duplicate_paragraphs=%d too_short_paragraphs=%d empty_output_files=%d output_dir=%s\n",
		report.InputFiles,
		report.OutputFiles,
		report.InputParagraphs,
		report.OutputParagraphs,
		report.DuplicateParagraphs,
		report.TooShortParagraphs,
		report.EmptyOutputFiles,
		*outputDir,
	)
	return 0
}

func runSplitTextData(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("split-text-data", flag.ContinueOnError)
	flags.SetOutput(stderr)

	textPaths := flags.String("text", "", "comma-separated text files or directories to split")
	outputDir := flags.String("output-dir", "", "directory to write train/ and val/")
	valRatio := flags.Float64("val-ratio", 0.1, "fraction of files assigned to validation")
	seed := flags.Int64("seed", 1, "random seed for deterministic split")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if strings.TrimSpace(*textPaths) == "" || *outputDir == "" {
		fmt.Fprintln(stderr, "text and output-dir are required")
		flags.Usage()
		return 1
	}
	report, err := textdata.SplitTextDataset(splitCSV(*textPaths), textdata.SplitConfig{
		OutputDir: *outputDir,
		ValRatio:  *valRatio,
		Seed:      *seed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "split text data: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "input_files=%d train_files=%d val_files=%d train_dir=%s val_dir=%s\n",
		report.InputFiles,
		report.TrainFiles,
		report.ValFiles,
		filepath.Join(*outputDir, "train"),
		filepath.Join(*outputDir, "val"),
	)
	return 0
}

func runTrainMath(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("train-math", flag.ContinueOnError)
	flags.SetOutput(stderr)

	dataDir := flags.String("data-dir", "", "directory containing train.jsonl and val.jsonl")
	checkpointPath := flags.String("checkpoint", "", "path to write model checkpoint")
	resumePath := flags.String("resume", "", "optional checkpoint to resume from")
	modelType := flags.String("model", "mlp", "math model backend: mlp or transformer")
	contextSize := flags.Int("context-size", 32, "autoregressive context size")
	embeddingDim := flags.Int("embedding-dim", 32, "embedding size")
	hiddenDim := flags.Int("hidden-dim", 128, "hidden size")
	numHeads := flags.Int("num-heads", 1, "attention heads for -model transformer")
	numLayers := flags.Int("num-layers", 1, "decoder layers for -model transformer")
	epochs := flags.Int("epochs", 10, "number of training epochs")
	batchSize := flags.Int("batch-size", 64, "batch size")
	learningRate := flags.Float64("learning-rate", 0.01, "optimizer learning rate")
	warmupSteps := flags.Int("warmup-steps", 0, "linearly warm learning rate over this many global optimizer steps; 0 disables")
	decaySteps := flags.Int("decay-steps", 0, "linearly decay learning rate after warmup over this many global optimizer steps; 0 disables")
	minLearningRate := flags.Float64("min-learning-rate", 0, "minimum learning rate after decay")
	maxSteps := flags.Int("max-steps", 0, "optional maximum optimizer steps for this run; 0 disables")
	logEvery := flags.Int("log-every", 0, "print training progress every N optimizer steps; 0 disables")
	saveEvery := flags.Int("save-every", 0, "write periodic checkpoints every N optimizer steps; 0 disables")
	gradClip := flags.Float64("grad-clip", 0, "clip gradients by global norm before Adam; 0 disables")
	trainLimit := flags.Int("train-limit", 0, "optional cap on loaded training examples before sequence expansion; 0 disables")
	valLimit := flags.Int("val-limit", 0, "optional cap on loaded validation examples for loss and accuracy checks; 0 disables")
	skipAccuracyEval := flags.Bool("skip-accuracy-eval", false, "skip before/after generative validation accuracy checks")
	skipLossEval := flags.Bool("skip-loss-eval", false, "skip final full-dataset train and validation loss passes")
	seed := flags.Int64("seed", 1, "random seed")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *dataDir == "" || *checkpointPath == "" {
		fmt.Fprintln(stderr, "data-dir and checkpoint are required")
		flags.Usage()
		return 1
	}
	if *trainLimit < 0 || *valLimit < 0 {
		fmt.Fprintln(stderr, "train-limit and val-limit cannot be negative")
		return 1
	}
	contextSizeSet := false
	flags.Visit(func(flag *flag.Flag) {
		if flag.Name == "context-size" {
			contextSizeSet = true
		}
	})

	trainExamples, valExamples, err := loadArithmeticDatasets(*dataDir)
	if err != nil {
		fmt.Fprintf(stderr, "load dataset: %v\n", err)
		return 1
	}
	trainExamples = limitExamples(trainExamples, *trainLimit)
	valExamples = limitExamples(valExamples, *valLimit)
	tok := tokenizer.NewByteTokenizer()
	trainer, err := loadOrCreateAnyMathTrainer(*resumePath, *modelType, mathlm.Config{
		VocabSize:    tok.VocabSize(),
		ContextSize:  *contextSize,
		EmbeddingDim: *embeddingDim,
		HiddenDim:    *hiddenDim,
		Seed:         *seed,
	}, mathlm.TransformerConfig{
		VocabSize:    tok.VocabSize(),
		ContextSize:  *contextSize,
		EmbeddingDim: *embeddingDim,
		NumHeads:     *numHeads,
		NumLayers:    *numLayers,
		MLPDim:       *hiddenDim,
		Seed:         *seed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "load trainer: %v\n", err)
		return 1
	}
	trainingContextSize := *contextSize
	if *resumePath != "" {
		modelConfig := trainer.Model().Config()
		if contextSizeSet && *contextSize != modelConfig.ContextLength {
			fmt.Fprintf(stderr, "context-size %d does not match resumed checkpoint context size %d\n", *contextSize, modelConfig.ContextLength)
			return 1
		}
		trainingContextSize = modelConfig.ContextLength
	}
	trainSeq, err := arithmetic.BuildTrainingSequences(trainExamples, tok, trainingContextSize)
	if err != nil {
		fmt.Fprintf(stderr, "build train sequences: %v\n", err)
		return 1
	}
	valSeq, err := arithmetic.BuildTrainingSequences(valExamples, tok, trainingContextSize)
	if err != nil {
		fmt.Fprintf(stderr, "build val sequences: %v\n", err)
		return 1
	}

	var before mathlm.EvalReport
	if !*skipAccuracyEval {
		before, err = mathlm.EvaluateExamples(trainer.Model(), valExamples, maxCompletionTokens(valExamples))
		if err != nil {
			fmt.Fprintf(stderr, "evaluate before training: %v\n", err)
			return 1
		}
	}
	report, err := trainer.Train(trainSeq, valSeq, mathlm.TrainingConfig{
		Epochs:             *epochs,
		BatchSize:          *batchSize,
		LearningRate:       *learningRate,
		WarmupSteps:        *warmupSteps,
		DecaySteps:         *decaySteps,
		MinLearningRate:    *minLearningRate,
		Beta1:              0.9,
		Beta2:              0.999,
		Epsilon:            1e-8,
		Seed:               *seed,
		MaxSteps:           *maxSteps,
		LogEvery:           *logEvery,
		SaveEvery:          *saveEvery,
		GradClip:           *gradClip,
		SkipFinalTrainLoss: *skipLossEval,
		SkipValidationLoss: *skipLossEval,
		OnProgress: func(progress mathlm.TrainingProgress) error {
			fmt.Fprintf(stdout, "step=%d train_loss=%.4f learning_rate=%.6g elapsed=%s steps_per_sec=%.2f\n",
				progress.Step,
				progress.Loss,
				progress.LearningRate,
				progress.Elapsed.Truncate(time.Millisecond),
				progress.StepsPerSecond,
			)
			return nil
		},
		OnCheckpoint: func(step int) error {
			path := periodicCheckpointPath(*checkpointPath, step)
			if err := mathlm.SaveAnyCheckpoint(path, trainer); err != nil {
				return err
			}
			fmt.Fprintf(stdout, "checkpoint_step=%d path=%s\n", step, path)
			return nil
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "train model: %v\n", err)
		return 1
	}
	var after mathlm.EvalReport
	if !*skipAccuracyEval {
		after, err = mathlm.EvaluateExamples(trainer.Model(), valExamples, maxCompletionTokens(valExamples))
		if err != nil {
			fmt.Fprintf(stderr, "evaluate after training: %v\n", err)
			return 1
		}
	}
	if err := mathlm.SaveAnyCheckpoint(*checkpointPath, trainer); err != nil {
		fmt.Fprintf(stderr, "save checkpoint: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "model=%s train_loss=%.4f val_loss=%.4f steps=%d\n", trainer.ModelType, report.TrainLoss, report.ValLoss, report.Steps)
	if *skipAccuracyEval {
		fmt.Fprintln(stdout, "val_accuracy=skipped")
	} else {
		fmt.Fprintf(stdout, "val_accuracy_before=%.4f val_accuracy_after=%.4f\n", before.Accuracy, after.Accuracy)
	}
	fmt.Fprintf(stdout, "checkpoint=%s\n", *checkpointPath)
	return 0
}

func runTrainText(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("train-text", flag.ContinueOnError)
	flags.SetOutput(stderr)

	textPaths := flags.String("text", "", "comma-separated text files or directories for pretraining")
	valTextPaths := flags.String("val-text", "", "optional comma-separated validation text files or directories")
	instructionPaths := flags.String("instructions", "", "comma-separated instruction JSONL files")
	valInstructionPaths := flags.String("val-instructions", "", "optional comma-separated validation instruction JSONL files")
	checkpointPath := flags.String("checkpoint", "", "path to write model checkpoint")
	resumePath := flags.String("resume", "", "optional checkpoint to resume from")
	contextSize := flags.Int("context-size", 128, "autoregressive context size")
	embeddingDim := flags.Int("embedding-dim", 128, "embedding size")
	hiddenDim := flags.Int("hidden-dim", 512, "transformer MLP hidden size")
	numHeads := flags.Int("num-heads", 4, "attention heads")
	numLayers := flags.Int("num-layers", 4, "decoder layers")
	epochs := flags.Int("epochs", 1, "number of training epochs")
	batchSize := flags.Int("batch-size", 16, "batch size")
	learningRate := flags.Float64("learning-rate", 0.0003, "optimizer learning rate")
	warmupSteps := flags.Int("warmup-steps", 100, "linearly warm learning rate over this many current-run steps; 0 disables")
	decaySteps := flags.Int("decay-steps", 10000, "linearly decay learning rate after warmup over this many current-run steps; 0 disables")
	minLearningRate := flags.Float64("min-learning-rate", 0.00003, "minimum learning rate after decay")
	maxSteps := flags.Int("max-steps", 0, "optional maximum optimizer steps for this run; 0 disables")
	logEvery := flags.Int("log-every", 100, "print training progress every N optimizer steps; 0 disables")
	saveEvery := flags.Int("save-every", 1000, "write periodic checkpoints every N optimizer steps; 0 disables")
	gradClip := flags.Float64("grad-clip", 1, "clip gradients by global norm before Adam; 0 disables")
	stride := flags.Int("stride", 1, "stride between raw-text next-token training positions")
	trainLimit := flags.Int("train-limit", 0, "optional cap on built training sequences; 0 disables")
	valLimit := flags.Int("val-limit", 0, "optional cap on built validation sequences; 0 disables")
	skipLossEval := flags.Bool("skip-loss-eval", false, "skip final full-dataset train and validation loss passes")
	seed := flags.Int64("seed", 1, "random seed")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *checkpointPath == "" {
		fmt.Fprintln(stderr, "checkpoint is required")
		flags.Usage()
		return 1
	}
	if strings.TrimSpace(*textPaths) == "" && strings.TrimSpace(*instructionPaths) == "" {
		fmt.Fprintln(stderr, "text or instructions is required")
		flags.Usage()
		return 1
	}
	if *trainLimit < 0 || *valLimit < 0 {
		fmt.Fprintln(stderr, "train-limit and val-limit cannot be negative")
		return 1
	}

	tok := tokenizer.NewByteTokenizer()
	trainer, err := loadOrCreateAnyMathTrainer(*resumePath, "transformer", mathlm.Config{}, mathlm.TransformerConfig{
		VocabSize:    tok.VocabSize(),
		ContextSize:  *contextSize,
		EmbeddingDim: *embeddingDim,
		NumHeads:     *numHeads,
		NumLayers:    *numLayers,
		MLPDim:       *hiddenDim,
		Seed:         *seed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "load trainer: %v\n", err)
		return 1
	}
	if trainer.ModelType != "transformer" {
		fmt.Fprintln(stderr, "train-text requires a transformer checkpoint")
		return 1
	}
	trainingContextSize := *contextSize
	if *resumePath != "" {
		trainingContextSize = trainer.Model().Config().ContextLength
	}

	trainSeq, err := buildTextTrainingSequences(tok, trainingContextSize, *stride, splitCSV(*textPaths), splitCSV(*instructionPaths))
	if err != nil {
		fmt.Fprintf(stderr, "build training sequences: %v\n", err)
		return 1
	}
	valSeq, err := buildTextTrainingSequences(tok, trainingContextSize, *stride, splitCSV(*valTextPaths), splitCSV(*valInstructionPaths))
	if err != nil && (strings.TrimSpace(*valTextPaths) != "" || strings.TrimSpace(*valInstructionPaths) != "") {
		fmt.Fprintf(stderr, "build validation sequences: %v\n", err)
		return 1
	}
	if strings.TrimSpace(*valTextPaths) == "" && strings.TrimSpace(*valInstructionPaths) == "" {
		valSeq = nil
	}
	trainSeq = limitSequences(trainSeq, *trainLimit)
	valSeq = limitSequences(valSeq, *valLimit)

	report, err := trainer.Train(trainSeq, valSeq, mathlm.TrainingConfig{
		Epochs:             *epochs,
		BatchSize:          *batchSize,
		LearningRate:       *learningRate,
		WarmupSteps:        *warmupSteps,
		DecaySteps:         *decaySteps,
		MinLearningRate:    *minLearningRate,
		Beta1:              0.9,
		Beta2:              0.999,
		Epsilon:            1e-8,
		Seed:               *seed,
		MaxSteps:           *maxSteps,
		LogEvery:           *logEvery,
		SaveEvery:          *saveEvery,
		GradClip:           *gradClip,
		SkipFinalTrainLoss: *skipLossEval,
		SkipValidationLoss: *skipLossEval,
		OnProgress: func(progress mathlm.TrainingProgress) error {
			fmt.Fprintf(stdout, "step=%d train_loss=%.4f learning_rate=%.6g elapsed=%s steps_per_sec=%.2f\n",
				progress.Step,
				progress.Loss,
				progress.LearningRate,
				progress.Elapsed.Truncate(time.Millisecond),
				progress.StepsPerSecond,
			)
			return nil
		},
		OnCheckpoint: func(step int) error {
			path := periodicCheckpointPath(*checkpointPath, step)
			if err := mathlm.SaveAnyCheckpoint(path, trainer); err != nil {
				return err
			}
			fmt.Fprintf(stdout, "checkpoint_step=%d path=%s\n", step, path)
			return nil
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "train text: %v\n", err)
		return 1
	}
	if err := mathlm.SaveAnyCheckpoint(*checkpointPath, trainer); err != nil {
		fmt.Fprintf(stderr, "save checkpoint: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "model=transformer train_loss=%.4f val_loss=%.4f steps=%d train_sequences=%d val_sequences=%d\n",
		report.TrainLoss,
		report.ValLoss,
		report.Steps,
		len(trainSeq),
		len(valSeq),
	)
	fmt.Fprintf(stdout, "checkpoint=%s\n", *checkpointPath)
	return 0
}

func runEvalMath(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("eval-math", flag.ContinueOnError)
	flags.SetOutput(stderr)

	checkpointPath := flags.String("checkpoint", "", "path to model checkpoint")
	dataPath := flags.String("data", "", "path to validation jsonl file")
	maxTokens := flags.Int("max-tokens", 0, "max completion tokens; defaults to dataset-driven value")
	showErrors := flags.Int("show-errors", 0, "print up to N incorrect examples with metadata")
	errorsOut := flags.String("errors-out", "", "write all incorrect examples and grouped template counts as JSON")
	exampleLimit := flags.Int("limit", 0, "optional cap on loaded evaluation examples; 0 disables")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *checkpointPath == "" || *dataPath == "" {
		fmt.Fprintln(stderr, "checkpoint and data are required")
		flags.Usage()
		return 1
	}
	if *exampleLimit < 0 {
		fmt.Fprintln(stderr, "limit cannot be negative")
		return 1
	}

	trainer, err := mathlm.LoadAnyCheckpoint(*checkpointPath)
	if err != nil {
		fmt.Fprintf(stderr, "load checkpoint: %v\n", err)
		return 1
	}
	examples, err := arithmetic.LoadExamples(*dataPath)
	if err != nil {
		fmt.Fprintf(stderr, "load examples: %v\n", err)
		return 1
	}
	examples = limitExamples(examples, *exampleLimit)
	tokenLimit := *maxTokens
	if tokenLimit <= 0 {
		tokenLimit = maxCompletionTokens(examples)
	}
	collectErrors := *showErrors > 0 || *errorsOut != ""
	report, err := mathlm.EvaluateExamplesWithOptions(trainer.Model(), examples, tokenLimit, mathlm.EvalOptions{
		CollectErrors: collectErrors,
	})
	if err != nil {
		fmt.Fprintf(stderr, "evaluate model: %v\n", err)
		return 1
	}
	writeEvalReport(stdout, report)
	if collectErrors {
		writeEvalDebugReport(stdout, report, *showErrors)
	}
	if *errorsOut != "" {
		if err := writeEvalErrorsFile(*errorsOut, report); err != nil {
			fmt.Fprintf(stderr, "write errors: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "errors_out=%s\n", *errorsOut)
	}
	return 0
}

func runEvalInstructions(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("eval-instructions", flag.ContinueOnError)
	flags.SetOutput(stderr)

	checkpointPath := flags.String("checkpoint", "", "path to JSON model checkpoint")
	instructionPath := flags.String("instructions", "", "path to instruction validation jsonl file")
	maxTokens := flags.Int("max-tokens", 0, "max generated tokens; defaults to expected completion length plus slack")
	showErrors := flags.Int("show-errors", 0, "print up to N incorrect examples")
	errorsOut := flags.String("errors-out", "", "write all incorrect examples as JSON")
	exampleLimit := flags.Int("limit", 0, "optional cap on loaded evaluation examples; 0 disables")
	stop := flags.String("stop", "\\nUser:,\\n\\nUser:,\\nSystem:,\\n\\nSystem:", "comma-separated stop strings; supports \\n and \\t escapes")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *checkpointPath == "" || *instructionPath == "" {
		fmt.Fprintln(stderr, "checkpoint and instructions are required")
		flags.Usage()
		return 1
	}
	if *exampleLimit < 0 {
		fmt.Fprintln(stderr, "limit cannot be negative")
		return 1
	}

	examples, err := textdata.LoadInstructionExamples([]string{*instructionPath})
	if err != nil {
		fmt.Fprintf(stderr, "load instruction examples: %v\n", err)
		return 1
	}
	examples = limitInstructionExamples(examples, *exampleLimit)

	engine, err := buildMathLMEngine(*checkpointPath)
	if err != nil {
		fmt.Fprintf(stderr, "create checkpoint engine: %v\n", err)
		return 1
	}
	tok := tokenizer.NewByteTokenizer()
	tokenLimit := *maxTokens
	if tokenLimit <= 0 {
		tokenLimit = maxInstructionCompletionTokens(examples, tok) + 8
	}
	report, err := evaluateInstructionExamples(engine, examples, tok, instructionEvalOptions{
		MaxTokens:     tokenLimit,
		StopStrings:   parseStopStrings(*stop),
		CollectErrors: *showErrors > 0 || *errorsOut != "",
	})
	if err != nil {
		fmt.Fprintf(stderr, "evaluate instructions: %v\n", err)
		return 1
	}

	writeInstructionEvalReport(stdout, report)
	if report.Errors != nil {
		writeInstructionEvalDebugReport(stdout, report, *showErrors)
	}
	if *errorsOut != "" {
		if err := writeInstructionEvalErrorsFile(*errorsOut, report); err != nil {
			fmt.Fprintf(stderr, "write errors: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "errors_out=%s\n", *errorsOut)
	}
	return 0
}

func runGenerateMath(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("generate-math", flag.ContinueOnError)
	flags.SetOutput(stderr)

	checkpointPath := flags.String("checkpoint", "", "path to model checkpoint")
	prompt := flags.String("prompt", "", "prompt text to complete")
	maxTokens := flags.Int("max-tokens", 8, "number of tokens to generate")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *checkpointPath == "" || *prompt == "" {
		fmt.Fprintln(stderr, "checkpoint and prompt are required")
		flags.Usage()
		return 1
	}

	trainer, err := mathlm.LoadAnyCheckpoint(*checkpointPath)
	if err != nil {
		fmt.Fprintf(stderr, "load checkpoint: %v\n", err)
		return 1
	}
	engine, err := runtime.NewEngine(tokenizer.NewByteTokenizer(), trainer.Model(), sampler.NewGreedySampler())
	if err != nil {
		fmt.Fprintf(stderr, "create engine: %v\n", err)
		return 1
	}
	output, err := engine.GenerateWithOptions(*prompt, runtime.GenerateOptions{
		MaxTokens:  *maxTokens,
		StopTokens: []int{int('\n')},
	})
	if err != nil {
		fmt.Fprintf(stderr, "generate math: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, output)
	return 0
}

func runGenerateCheckpoint(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("generate-checkpoint", flag.ContinueOnError)
	flags.SetOutput(stderr)

	checkpointPath := flags.String("checkpoint", "", "path to JSON model checkpoint")
	prompt := flags.String("prompt", "", "prompt text to generate from")
	maxTokens := flags.Int("max-tokens", 64, "number of tokens to generate")
	temperature := flags.Float64("temperature", 0, "sampling temperature; 0 keeps greedy decoding")
	topK := flags.Int("top-k", 0, "limit token sampling to the top-k logits; 0 disables top-k sampling")
	stop := flags.String("stop", "\\nUser:,\\n\\nUser:", "comma-separated stop strings; supports \\n and \\t escapes")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *checkpointPath == "" || *prompt == "" {
		fmt.Fprintln(stderr, "checkpoint and prompt are required")
		flags.Usage()
		return 1
	}
	engine, err := buildMathLMEngine(*checkpointPath)
	if err != nil {
		fmt.Fprintf(stderr, "create checkpoint engine: %v\n", err)
		return 1
	}
	output, err := engine.GenerateWithOptions(*prompt, runtime.GenerateOptions{
		MaxTokens:   *maxTokens,
		TopK:        *topK,
		StopStrings: parseStopStrings(*stop),
		Temperature: *temperature,
	})
	if err != nil {
		fmt.Fprintf(stderr, "generate-checkpoint: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, output)
	return 0
}

func runExportCheckpoint(args []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("export-checkpoint", flag.ContinueOnError)
	flags.SetOutput(stderr)

	checkpointPath := flags.String("checkpoint", "", "path to source JSON model checkpoint")
	outputPath := flags.String("output", "", "path to write exported JSON model checkpoint")
	stripOptimizer := flags.Bool("strip-optimizer", true, "remove Adam optimizer state for inference-only release artifacts")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintf(stderr, "parse flags: %v\n", err)
		return 2
	}
	if *checkpointPath == "" || *outputPath == "" {
		fmt.Fprintln(stderr, "checkpoint and output are required")
		flags.Usage()
		return 1
	}

	trainer, err := mathlm.LoadAnyCheckpoint(*checkpointPath)
	if err != nil {
		fmt.Fprintf(stderr, "load checkpoint: %v\n", err)
		return 1
	}
	if *stripOptimizer {
		stripAnyTrainerOptimizer(trainer)
	}
	if err := os.MkdirAll(filepath.Dir(*outputPath), 0o755); err != nil {
		fmt.Fprintf(stderr, "create output dir: %v\n", err)
		return 1
	}
	if err := mathlm.SaveAnyCheckpoint(*outputPath, trainer); err != nil {
		fmt.Fprintf(stderr, "write checkpoint: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "exported checkpoint input=%s output=%s model=%s strip_optimizer=%t\n", *checkpointPath, *outputPath, trainer.ModelType, *stripOptimizer)
	return 0
}

func stripAnyTrainerOptimizer(trainer *mathlm.AnyTrainer) {
	if trainer == nil {
		return
	}
	if trainer.MLP != nil {
		trainer.MLP.Adam = nil
	}
	if trainer.Transformer != nil {
		trainer.Transformer.Adam = nil
	}
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
	temperature := flags.Float64("temperature", 0, "sampling temperature; 0 keeps greedy decoding")
	topK := flags.Int("top-k", 0, "limit token sampling to the top-k logits; 0 disables top-k sampling")
	useCache := flags.Bool("use-cache", false, "use model KV cache when supported")
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
		MaxTokens:   *maxTokens,
		TopK:        *topK,
		StopTokens:  stopTokens,
		Temperature: *temperature,
		UseCache:    *useCache,
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
	backend := flags.String("backend", "auto", "model backend: auto, toy, gpt2, mathlm, or math-router")
	checkpointPath := flags.String("checkpoint", "", "path to mathlm JSON checkpoint for -backend mathlm or arithmetic checkpoint for -backend math-router")
	derivativeCheckpointPath := flags.String("derivative-checkpoint", "", "path to derivative mathlm JSON checkpoint for -backend math-router")
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
	}, *checkpointPath, *derivativeCheckpointPath)
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

func buildMathLMEngine(checkpointPath string) (*runtime.Engine, error) {
	trainer, err := mathlm.LoadAnyCheckpoint(checkpointPath)
	if err != nil {
		return nil, err
	}
	return runtime.NewEngine(tokenizer.NewByteTokenizer(), trainer.Model(), sampler.NewGreedySampler())
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
	underlying  generateWithOptions
	stopTokens  []int
	stopStrings []string
}

func (g generatorWithDefaults) GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error) {
	options.StopTokens = mergeStopTokens(options.StopTokens, g.stopTokens)
	options.StopStrings = mergeStopStrings(options.StopStrings, g.stopStrings)
	return g.underlying.GenerateWithOptions(prompt, options)
}

func buildServeGenerator(backend, baseDir string, paths gpt2AssetPaths, checkpointPath string, derivativeCheckpointPath string) (server.Generator, string, error) {
	selectedBackend, err := resolveServeBackend(backend, baseDir, paths, checkpointPath, derivativeCheckpointPath)
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
	case "mathlm":
		if checkpointPath == "" {
			return nil, "", fmt.Errorf("checkpoint is required for mathlm backend")
		}
		engine, err := buildMathLMEngine(checkpointPath)
		if err != nil {
			return nil, "", err
		}
		return generatorWithDefaults{
			underlying:  engine,
			stopStrings: []string{"\nUser:", "\n\nUser:"},
		}, selectedBackend, nil
	case "math-router":
		if checkpointPath == "" {
			return nil, "", fmt.Errorf("checkpoint is required for math-router backend")
		}
		if derivativeCheckpointPath == "" {
			return nil, "", fmt.Errorf("derivative-checkpoint is required for math-router backend")
		}
		arithmeticEngine, err := buildMathLMEngine(checkpointPath)
		if err != nil {
			return nil, "", fmt.Errorf("create arithmetic engine: %w", err)
		}
		derivativeEngine, err := buildMathLMEngine(derivativeCheckpointPath)
		if err != nil {
			return nil, "", fmt.Errorf("create derivative engine: %w", err)
		}
		return mathrouter.Router{
			Arithmetic:  arithmeticEngine,
			Derivative:  derivativeEngine,
			PreferModel: true,
		}, selectedBackend, nil
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

func resolveServeBackend(requested, baseDir string, paths gpt2AssetPaths, checkpointPath string, derivativeCheckpointPath string) (string, error) {
	switch requested {
	case "toy", "gpt2", "mathlm", "math-router":
		return requested, nil
	case "auto", "":
		if checkpointPath != "" && derivativeCheckpointPath != "" {
			return "math-router", nil
		}
		if checkpointPath != "" {
			return "mathlm", nil
		}
		resolved := resolveGPT2AssetPaths(baseDir, paths)
		if hasGPT2Assets(resolved) {
			return "gpt2", nil
		}
		return "toy", nil
	default:
		return "", fmt.Errorf("backend must be one of auto, toy, gpt2, mathlm, or math-router")
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

func mergeStopStrings(existing, defaults []string) []string {
	if len(defaults) == 0 {
		return existing
	}
	merged := append([]string(nil), existing...)
	for _, candidate := range defaults {
		if candidate == "" {
			continue
		}
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
	case "math-router":
		return server.GeneratePolicy{
			DefaultMaxTokens:   24,
			MaxTokensCap:       64,
			DefaultTemperature: 0,
			DefaultTopK:        1,
			MaxTopK:            1,
			MaxMessages:        4,
			MaxMessageRunes:    512,
			MaxPromptRunes:     1024,
			DefaultStopStrings: []string{"\nUser:", "\n\nUser:", "\nAssistant:", "\n\nAssistant:"},
			MaxStopStrings:     8,
			MaxStopRunes:       64,
		}
	case "gpt2":
		return server.GeneratePolicy{
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
			DefaultStopStrings: []string{"\nUser:", "\n\nUser:"},
			MaxStopStrings:     8,
			MaxStopRunes:       64,
		}
	case "mathlm":
		return server.GeneratePolicy{
			DefaultMaxTokens:   64,
			MaxTokensCap:       256,
			DefaultTemperature: 0.7,
			MinTemperature:     0.1,
			MaxTemperature:     1.2,
			DefaultTopK:        40,
			MaxTopK:            100,
			MaxMessages:        8,
			MaxMessageRunes:    512,
			MaxPromptRunes:     2048,
			AssistantPreamble:  "You are Aurelius, a concise helpful assistant.",
			DefaultStopStrings: []string{"\nUser:", "\n\nUser:"},
			MaxStopStrings:     8,
			MaxStopRunes:       64,
		}
	default:
		return server.GeneratePolicy{}
	}
}

func splitCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseStopStrings(value string) []string {
	parts := splitCSV(value)
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.ReplaceAll(part, `\n`, "\n")
		part = strings.ReplaceAll(part, `\r`, "\r")
		part = strings.ReplaceAll(part, `\t`, "\t")
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func splitInts(value string) ([]int, error) {
	parts := strings.Split(value, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q", part)
		}
		out = append(out, parsed)
	}
	return out, nil
}

func parseMixSources(value string) ([]arithmetic.MixSource, error) {
	parts := strings.Split(value, ",")
	sources := make([]arithmetic.MixSource, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		path, weightText, ok := strings.Cut(part, ":")
		if !ok {
			return nil, fmt.Errorf("input %q must use path:weight", part)
		}
		path = strings.TrimSpace(path)
		weightText = strings.TrimSpace(weightText)
		weight, err := strconv.Atoi(weightText)
		if err != nil {
			return nil, fmt.Errorf("invalid weight %q", weightText)
		}
		sources = append(sources, arithmetic.MixSource{
			Path:   path,
			Weight: weight,
		})
	}
	if len(sources) == 0 {
		return nil, fmt.Errorf("at least one input is required")
	}
	return sources, nil
}

func writeEvalReport(stdout io.Writer, report mathlm.EvalReport) {
	fmt.Fprintf(stdout, "accuracy=%.4f correct=%d total=%d max_tokens=%d\n", report.Accuracy, report.Correct, report.Total, report.MaxTokens)
	for _, operation := range sortedStringKeys(report.ByOperation) {
		group := report.ByOperation[operation]
		fmt.Fprintf(stdout, "operation[%s]=%.4f correct=%d total=%d\n", operation, group.Accuracy, group.Correct, group.Total)
	}
	for _, level := range sortedIntKeys(report.ByLevel) {
		group := report.ByLevel[level]
		fmt.Fprintf(stdout, "level[%d]=%.4f correct=%d total=%d\n", level, group.Accuracy, group.Correct, group.Total)
	}
}

type mathEvalErrorReplayReport struct {
	Total    int                `json:"total"`
	Correct  int                `json:"correct"`
	Accuracy float64            `json:"accuracy"`
	Errors   []mathlm.EvalError `json:"errors"`
}

type mathErrorReplayMetadata struct {
	SourceErrors     string  `json:"source_errors"`
	SourceTotal      int     `json:"source_total"`
	SourceCorrect    int     `json:"source_correct"`
	SourceAccuracy   float64 `json:"source_accuracy"`
	InputErrors      int     `json:"input_errors"`
	UniqueErrors     int     `json:"unique_errors"`
	TrainCount       int     `json:"train_count"`
	ValCount         int     `json:"val_count"`
	Repeat           int     `json:"repeat"`
	ValRatio         float64 `json:"val_ratio"`
	Seed             int64   `json:"seed"`
	AnswerSource     string  `json:"answer_source"`
	ReplayCompletion string  `json:"replay_completion"`
}

func loadMathEvalErrorReplayReport(path string) (mathEvalErrorReplayReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return mathEvalErrorReplayReport{}, fmt.Errorf("read errors file %q: %w", path, err)
	}
	var report mathEvalErrorReplayReport
	if err := json.Unmarshal(data, &report); err != nil {
		return mathEvalErrorReplayReport{}, fmt.Errorf("parse errors file %q: %w", path, err)
	}
	return report, nil
}

func buildErrorReplayExamples(errors []mathlm.EvalError) []arithmetic.Example {
	seen := make(map[string]bool)
	examples := make([]arithmetic.Example, 0, len(errors))
	for _, evalError := range errors {
		prompt := strings.TrimSpace(evalError.Prompt)
		expected := strings.TrimSpace(evalError.Expected)
		if prompt == "" || expected == "" {
			continue
		}
		answer := expected
		if task, ok := mathrouter.Normalize(evalError.Prompt); ok && task.Solved {
			answer = task.Answer
		}
		key := prompt + "\x00" + answer
		if seen[key] {
			continue
		}
		seen[key] = true
		examples = append(examples, arithmetic.Example{
			Prompt:          evalError.Prompt,
			Completion:      answer,
			Answer:          answer,
			Operation:       evalError.Operation,
			Level:           evalError.Level,
			MinOperand:      evalError.MinOperand,
			MaxOperand:      evalError.MaxOperand,
			AnswerDigits:    evalError.AnswerDigits,
			SmallDifference: evalError.SmallDifference,
			RequiresCarry:   evalError.RequiresCarry,
			RequiresBorrow:  evalError.RequiresBorrow,
			Template:        evalError.Template,
			ReasoningStyle:  replayReasoningStyle(evalError.ReasoningStyle),
		})
	}
	return examples
}

func replayReasoningStyle(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "direct"
	}
	return value
}

func splitReplayExamples(examples []arithmetic.Example, valRatio float64) ([]arithmetic.Example, []arithmetic.Example) {
	if len(examples) <= 1 || valRatio == 0 {
		return append([]arithmetic.Example(nil), examples...), nil
	}
	valCount := int(math.Round(float64(len(examples)) * valRatio))
	if valCount < 1 {
		valCount = 1
	}
	if valCount >= len(examples) {
		valCount = len(examples) - 1
	}
	val := append([]arithmetic.Example(nil), examples[:valCount]...)
	train := append([]arithmetic.Example(nil), examples[valCount:]...)
	return train, val
}

func repeatExamples(examples []arithmetic.Example, repeat int) []arithmetic.Example {
	repeated := make([]arithmetic.Example, 0, len(examples)*repeat)
	for i := 0; i < repeat; i++ {
		repeated = append(repeated, examples...)
	}
	return repeated
}

func writeArithmeticJSONL(path string, examples []arithmetic.Example) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	for _, example := range examples {
		if err := encoder.Encode(example); err != nil {
			return fmt.Errorf("write %q: %w", path, err)
		}
	}
	return nil
}

func writeJSONFile(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal %q: %w", path, err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %q: %w", path, err)
	}
	return nil
}

func writeEvalDebugReport(stdout io.Writer, report mathlm.EvalReport, showErrors int) {
	for _, template := range sortedStringKeys(report.ByTemplate) {
		group := report.ByTemplate[template]
		errors := group.Total - group.Correct
		fmt.Fprintf(stdout, "template[%s]=%.4f correct=%d total=%d errors=%d\n", template, group.Accuracy, group.Correct, group.Total, errors)
	}
	for _, digits := range sortedIntKeys(report.ByAnswerDigits) {
		group := report.ByAnswerDigits[digits]
		errors := group.Total - group.Correct
		fmt.Fprintf(stdout, "answer_digits[%d]=%.4f correct=%d total=%d errors=%d\n", digits, group.Accuracy, group.Correct, group.Total, errors)
	}
	for _, key := range sortedStringKeys(report.BySmallDifference) {
		group := report.BySmallDifference[key]
		errors := group.Total - group.Correct
		fmt.Fprintf(stdout, "small_difference[%s]=%.4f correct=%d total=%d errors=%d\n", key, group.Accuracy, group.Correct, group.Total, errors)
	}
	if showErrors <= 0 {
		return
	}
	limit := min(showErrors, len(report.Errors))
	for i := 0; i < limit; i++ {
		err := report.Errors[i]
		fmt.Fprintf(stdout, "error[%d] prompt=%q expected=%q generated=%q operation=%s level=%d template=%s reasoning_style=%s answer_digits=%d small_difference=%t carry=%t borrow=%t operands=%d..%d\n",
			i+1,
			err.Prompt,
			err.Expected,
			err.Generated,
			err.Operation,
			err.Level,
			err.Template,
			err.ReasoningStyle,
			err.AnswerDigits,
			err.SmallDifference,
			err.RequiresCarry,
			err.RequiresBorrow,
			err.MinOperand,
			err.MaxOperand,
		)
	}
}

func writeEvalErrorsFile(path string, report mathlm.EvalReport) error {
	payload := struct {
		Total             int                         `json:"total"`
		Correct           int                         `json:"correct"`
		Accuracy          float64                     `json:"accuracy"`
		MaxTokens         int                         `json:"max_tokens"`
		ByTemplate        map[string]mathlm.EvalGroup `json:"by_template"`
		ByAnswerDigits    map[int]mathlm.EvalGroup    `json:"by_answer_digits"`
		BySmallDifference map[string]mathlm.EvalGroup `json:"by_small_difference"`
		Errors            []mathlm.EvalError          `json:"errors"`
	}{
		Total:             report.Total,
		Correct:           report.Correct,
		Accuracy:          report.Accuracy,
		MaxTokens:         report.MaxTokens,
		ByTemplate:        report.ByTemplate,
		ByAnswerDigits:    report.ByAnswerDigits,
		BySmallDifference: report.BySmallDifference,
		Errors:            report.Errors,
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal eval errors: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write eval errors file: %w", err)
	}
	return nil
}

type instructionEvalOptions struct {
	MaxTokens     int
	StopStrings   []string
	CollectErrors bool
}

type instructionEvalReport struct {
	Total     int                    `json:"total"`
	Correct   int                    `json:"correct"`
	Accuracy  float64                `json:"accuracy"`
	MaxTokens int                    `json:"max_tokens"`
	Errors    []instructionEvalError `json:"errors,omitempty"`
}

type instructionEvalError struct {
	Index     int    `json:"index"`
	Prompt    string `json:"prompt"`
	Expected  string `json:"expected"`
	Generated string `json:"generated"`
}

func evaluateInstructionExamples(engine generateWithOptions, examples []textdata.InstructionExample, tok tokenizer.Tokenizer, options instructionEvalOptions) (instructionEvalReport, error) {
	if engine == nil {
		return instructionEvalReport{}, fmt.Errorf("generator is required")
	}
	if tok == nil {
		return instructionEvalReport{}, fmt.Errorf("tokenizer is required")
	}
	if options.MaxTokens <= 0 {
		return instructionEvalReport{}, fmt.Errorf("max tokens must be positive")
	}
	report := instructionEvalReport{
		Total:     len(examples),
		MaxTokens: options.MaxTokens,
	}
	for i, example := range examples {
		prompt, expected := example.PromptCompletion()
		output, err := engine.GenerateWithOptions(prompt, runtime.GenerateOptions{
			MaxTokens:   options.MaxTokens,
			TopK:        1,
			StopStrings: options.StopStrings,
		})
		if err != nil {
			return instructionEvalReport{}, fmt.Errorf("generate instruction %d: %w", i+1, err)
		}
		generated := normalizeInstructionGenerated(output, prompt)
		expected = strings.TrimSpace(expected)
		if generated == expected {
			report.Correct++
			continue
		}
		if options.CollectErrors {
			report.Errors = append(report.Errors, instructionEvalError{
				Index:     i + 1,
				Prompt:    prompt,
				Expected:  expected,
				Generated: generated,
			})
		}
	}
	if report.Total > 0 {
		report.Accuracy = float64(report.Correct) / float64(report.Total)
	}
	return report, nil
}

func normalizeInstructionGenerated(output string, prompt string) string {
	generated := strings.TrimPrefix(output, prompt)
	for _, marker := range []string{"\nUser:", "\n\nUser:", "\nSystem:", "\n\nSystem:"} {
		if idx := strings.Index(generated, marker); idx >= 0 {
			generated = generated[:idx]
		}
	}
	if idx := strings.Index(generated, "\n"); idx >= 0 {
		generated = generated[:idx]
	}
	return strings.TrimSpace(generated)
}

func writeInstructionEvalReport(stdout io.Writer, report instructionEvalReport) {
	fmt.Fprintf(stdout, "accuracy=%.4f correct=%d total=%d max_tokens=%d\n", report.Accuracy, report.Correct, report.Total, report.MaxTokens)
}

func writeInstructionEvalDebugReport(stdout io.Writer, report instructionEvalReport, showErrors int) {
	if showErrors <= 0 {
		return
	}
	limit := min(showErrors, len(report.Errors))
	for i := 0; i < limit; i++ {
		err := report.Errors[i]
		fmt.Fprintf(stdout, "error[%d] index=%d prompt=%q expected=%q generated=%q\n",
			i+1,
			err.Index,
			err.Prompt,
			err.Expected,
			err.Generated,
		)
	}
}

func writeInstructionEvalErrorsFile(path string, report instructionEvalReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal instruction eval errors: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write instruction eval errors file: %w", err)
	}
	return nil
}

func periodicCheckpointPath(basePath string, step int) string {
	ext := filepath.Ext(basePath)
	stem := strings.TrimSuffix(basePath, ext)
	if ext == "" {
		return fmt.Sprintf("%s-step%d", basePath, step)
	}
	return fmt.Sprintf("%s-step%d%s", stem, step, ext)
}

func sortedStringKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func sortedIntKeys[V any](values map[int]V) []int {
	keys := make([]int, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func loadArithmeticDatasets(dataDir string) ([]arithmetic.Example, []arithmetic.Example, error) {
	trainPath := filepath.Join(dataDir, "train.jsonl")
	valPath := filepath.Join(dataDir, "val.jsonl")
	train, err := arithmetic.LoadExamples(trainPath)
	if err != nil {
		return nil, nil, err
	}
	val, err := arithmetic.LoadExamples(valPath)
	if err != nil {
		return nil, nil, err
	}
	return train, val, nil
}

func limitExamples(examples []arithmetic.Example, limit int) []arithmetic.Example {
	if limit <= 0 || limit >= len(examples) {
		return examples
	}
	return examples[:limit]
}

func limitInstructionExamples(examples []textdata.InstructionExample, limit int) []textdata.InstructionExample {
	if limit <= 0 || limit >= len(examples) {
		return examples
	}
	return examples[:limit]
}

func buildTextTrainingSequences(tok tokenizer.Tokenizer, contextSize int, stride int, textPaths []string, instructionPaths []string) ([]arithmetic.SequenceExample, error) {
	sequences := make([]arithmetic.SequenceExample, 0)
	if len(textPaths) > 0 {
		text, err := textdata.LoadText(textPaths)
		if err != nil {
			return nil, err
		}
		textSequences, err := textdata.BuildPretrainingSequences(text, tok, textdata.BuildConfig{
			ContextSize: contextSize,
			Stride:      stride,
		})
		if err != nil {
			return nil, err
		}
		sequences = append(sequences, textSequences...)
	}
	if len(instructionPaths) > 0 {
		examples, err := textdata.LoadInstructionExamples(instructionPaths)
		if err != nil {
			return nil, err
		}
		instructionSequences, err := textdata.BuildInstructionSequences(examples, tok, contextSize)
		if err != nil {
			return nil, err
		}
		sequences = append(sequences, instructionSequences...)
	}
	if len(sequences) == 0 {
		return nil, fmt.Errorf("no text training sequences built")
	}
	return sequences, nil
}

func limitSequences(sequences []arithmetic.SequenceExample, limit int) []arithmetic.SequenceExample {
	if limit <= 0 || limit >= len(sequences) {
		return sequences
	}
	return sequences[:limit]
}

func loadOrCreateMathTrainer(resumePath string, cfg mathlm.Config) (*mathlm.Trainer, error) {
	if resumePath != "" {
		return mathlm.LoadCheckpoint(resumePath)
	}
	model, err := mathlm.NewModel(cfg)
	if err != nil {
		return nil, err
	}
	return mathlm.NewTrainer(model)
}

func loadOrCreateAnyMathTrainer(resumePath string, modelType string, mlpCfg mathlm.Config, transformerCfg mathlm.TransformerConfig) (*mathlm.AnyTrainer, error) {
	if resumePath != "" {
		return mathlm.LoadAnyCheckpoint(resumePath)
	}
	switch modelType {
	case "mlp":
		model, err := mathlm.NewModel(mlpCfg)
		if err != nil {
			return nil, err
		}
		trainer, err := mathlm.NewTrainer(model)
		if err != nil {
			return nil, err
		}
		return mathlm.NewMLPAnyTrainer(trainer)
	case "transformer":
		model, err := mathlm.NewTransformerModel(transformerCfg)
		if err != nil {
			return nil, err
		}
		trainer, err := mathlm.NewTransformerTrainer(model)
		if err != nil {
			return nil, err
		}
		return mathlm.NewTransformerAnyTrainer(trainer)
	default:
		return nil, fmt.Errorf("unsupported math model %q", modelType)
	}
}

func maxCompletionTokens(examples []arithmetic.Example) int {
	maxTokens := 1
	for _, example := range examples {
		length := len(example.Completion) + 1
		if length > maxTokens {
			maxTokens = length
		}
	}
	return maxTokens
}

func maxInstructionCompletionTokens(examples []textdata.InstructionExample, tok tokenizer.Tokenizer) int {
	maxTokens := 1
	for _, example := range examples {
		_, completion := example.PromptCompletion()
		tokens, err := tok.Encode(completion + "\n")
		if err != nil {
			continue
		}
		if len(tokens) > maxTokens {
			maxTokens = len(tokens)
		}
	}
	return maxTokens
}
