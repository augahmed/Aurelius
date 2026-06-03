package textdata

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/augahmed/aurelius/internal/arithmetic"
	"github.com/augahmed/aurelius/internal/tokenizer"
)

type InstructionExample struct {
	Prompt      string `json:"prompt,omitempty"`
	Completion  string `json:"completion,omitempty"`
	Instruction string `json:"instruction,omitempty"`
	Input       string `json:"input,omitempty"`
	Output      string `json:"output,omitempty"`
	System      string `json:"system,omitempty"`
}

type BuildConfig struct {
	ContextSize int
	Stride      int
}

func LoadText(paths []string) (string, error) {
	if len(paths) == 0 {
		return "", fmt.Errorf("at least one text path is required")
	}
	files, err := expandTextPaths(paths)
	if err != nil {
		return "", err
	}
	var builder strings.Builder
	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read text %q: %w", path, err)
		}
		if builder.Len() > 0 {
			builder.WriteString("\n\n")
		}
		builder.Write(data)
	}
	return builder.String(), nil
}

func BuildPretrainingSequences(text string, tok tokenizer.Tokenizer, cfg BuildConfig) ([]arithmetic.SequenceExample, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	if cfg.ContextSize <= 0 {
		return nil, fmt.Errorf("context size must be positive")
	}
	stride := cfg.Stride
	if stride <= 0 {
		stride = 1
	}
	tokens, err := tok.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("encode text: %w", err)
	}
	if len(tokens) < 2 {
		return nil, fmt.Errorf("text must produce at least two tokens")
	}
	sequences := make([]arithmetic.SequenceExample, 0, (len(tokens)-1+stride-1)/stride)
	for targetIndex := 1; targetIndex < len(tokens); targetIndex += stride {
		context := make([]int, cfg.ContextSize)
		start := max(0, targetIndex-cfg.ContextSize)
		window := tokens[start:targetIndex]
		copy(context[cfg.ContextSize-len(window):], window)
		sequences = append(sequences, arithmetic.SequenceExample{
			Context: context,
			Target:  tokens[targetIndex],
		})
	}
	return sequences, nil
}

func LoadInstructionExamples(paths []string) ([]InstructionExample, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("at least one instruction path is required")
	}
	examples := make([]InstructionExample, 0)
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("open instruction data %q: %w", path, err)
		}
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		line := 0
		for scanner.Scan() {
			line++
			raw := strings.TrimSpace(scanner.Text())
			if raw == "" {
				continue
			}
			var example InstructionExample
			if err := json.Unmarshal([]byte(raw), &example); err != nil {
				_ = file.Close()
				return nil, fmt.Errorf("parse instruction line %d in %q: %w", line, path, err)
			}
			if err := example.Validate(); err != nil {
				_ = file.Close()
				return nil, fmt.Errorf("instruction line %d in %q: %w", line, path, err)
			}
			examples = append(examples, example)
		}
		if err := scanner.Err(); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("scan instruction data %q: %w", path, err)
		}
		if err := file.Close(); err != nil {
			return nil, fmt.Errorf("close instruction data %q: %w", path, err)
		}
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("instruction data is empty")
	}
	return examples, nil
}

func BuildInstructionSequences(examples []InstructionExample, tok tokenizer.Tokenizer, contextSize int) ([]arithmetic.SequenceExample, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	if contextSize <= 0 {
		return nil, fmt.Errorf("context size must be positive")
	}
	sequences := make([]arithmetic.SequenceExample, 0)
	for _, example := range examples {
		prompt, completion := example.PromptCompletion()
		promptTokens, err := tok.Encode(prompt)
		if err != nil {
			return nil, fmt.Errorf("encode instruction prompt: %w", err)
		}
		targetTokens, err := tok.Encode(completion + "\n")
		if err != nil {
			return nil, fmt.Errorf("encode instruction completion: %w", err)
		}
		prefix := append([]int(nil), promptTokens...)
		for _, target := range targetTokens {
			context := make([]int, contextSize)
			start := max(0, len(prefix)-contextSize)
			window := prefix[start:]
			copy(context[contextSize-len(window):], window)
			sequences = append(sequences, arithmetic.SequenceExample{
				Context: context,
				Target:  target,
			})
			prefix = append(prefix, target)
		}
	}
	return sequences, nil
}

func (e InstructionExample) Validate() error {
	prompt, completion := e.PromptCompletion()
	if strings.TrimSpace(prompt) == "" {
		return fmt.Errorf("prompt or instruction is required")
	}
	if strings.TrimSpace(completion) == "" {
		return fmt.Errorf("completion or output is required")
	}
	return nil
}

func (e InstructionExample) PromptCompletion() (string, string) {
	if strings.TrimSpace(e.Prompt) != "" || strings.TrimSpace(e.Completion) != "" {
		return strings.TrimSpace(e.Prompt), strings.TrimSpace(e.Completion)
	}
	var builder strings.Builder
	if strings.TrimSpace(e.System) != "" {
		builder.WriteString("System: ")
		builder.WriteString(strings.TrimSpace(e.System))
		builder.WriteString("\n\n")
	}
	builder.WriteString("User: ")
	builder.WriteString(strings.TrimSpace(e.Instruction))
	if strings.TrimSpace(e.Input) != "" {
		builder.WriteString("\n")
		builder.WriteString(strings.TrimSpace(e.Input))
	}
	builder.WriteString("\n\nAssistant:")
	return builder.String(), strings.TrimSpace(e.Output)
}

func expandTextPaths(paths []string) ([]string, error) {
	files := make([]string, 0)
	for _, path := range paths {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		info, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("stat text path %q: %w", path, err)
		}
		if !info.IsDir() {
			files = append(files, path)
			continue
		}
		if err := filepath.WalkDir(path, func(candidate string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			name := strings.ToLower(entry.Name())
			if strings.HasSuffix(name, ".txt") || strings.HasSuffix(name, ".md") {
				files = append(files, candidate)
			}
			return nil
		}); err != nil {
			return nil, fmt.Errorf("walk text dir %q: %w", path, err)
		}
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no text files found")
	}
	return files, nil
}
