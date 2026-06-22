package mathrouter

import (
	"testing"

	"github.com/augahmed/aurelius/internal/runtime"
)

func TestNormalizeArithmetic(t *testing.T) {
	task, ok := Normalize("Can you solve 7 times 8?")
	if !ok {
		t.Fatal("Normalize returned false")
	}
	if task.Route != RouteArithmetic || task.Prompt != "7 * 8 = " {
		t.Fatalf("task = %+v, want arithmetic direct prompt", task)
	}
	if !task.Solved || task.Answer != "56" {
		t.Fatalf("task = %+v, want solved answer 56", task)
	}
}

func TestNormalizeDerivative(t *testing.T) {
	task, ok := Normalize("What is the derivative of 4x^3 + 3x^2 + 4x + 7?")
	if !ok {
		t.Fatal("Normalize returned false")
	}
	want := "Derrivative: 4x^3 + 3x^2 + 4x + 7 "
	if task.Route != RouteDerivative || task.Prompt != want {
		t.Fatalf("task = %+v, want %q", task, want)
	}
	if !task.Solved || task.Answer != "12x^2 + 6x + 4" {
		t.Fatalf("task = %+v, want solved derivative", task)
	}
}

func TestNormalizeConversationUsesLatestUserMessage(t *testing.T) {
	task, ok := Normalize("User: hello\n\nAssistant: hi\n\nUser: What is 12 × 6?\n\nAssistant:")
	if !ok {
		t.Fatal("Normalize returned false")
	}
	if task.Prompt != "12 * 6 = " {
		t.Fatalf("prompt = %q, want direct multiplication prompt", task.Prompt)
	}
}

func TestNormalizeArithmeticEdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		prompt string
		answer string
	}{
		{
			name:   "zero addition",
			input:  "What is 0 + 0?",
			prompt: "0 + 0 = ",
			answer: "0",
		},
		{
			name:   "large carry",
			input:  "What is 99 + 99?",
			prompt: "99 + 99 = ",
			answer: "198",
		},
		{
			name:   "negative operand",
			input:  "Can you solve -4 plus 9?",
			prompt: "-4 + 9 = ",
			answer: "5",
		},
		{
			name:   "negative result",
			input:  "What is 10 - 99?",
			prompt: "10 - 99 = ",
			answer: "-89",
		},
		{
			name:   "letter multiplication sign",
			input:  "Solve: 12 x 12 =",
			prompt: "12 * 12 = ",
			answer: "144",
		},
		{
			name:   "unicode multiplication sign",
			input:  "What is 12 × 6?",
			prompt: "12 * 6 = ",
			answer: "72",
		},
		{
			name:   "word multiplication",
			input:  "What is 6 multiplied by 7?",
			prompt: "6 * 7 = ",
			answer: "42",
		},
		{
			name:   "multiply by zero",
			input:  "What is 1000 * 0?",
			prompt: "1000 * 0 = ",
			answer: "0",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			task, ok := Normalize(test.input)
			if !ok {
				t.Fatal("Normalize returned false")
			}
			if task.Route != RouteArithmetic || task.Prompt != test.prompt {
				t.Fatalf("task = %+v, want arithmetic prompt %q", task, test.prompt)
			}
			if !task.Solved || task.Answer != test.answer {
				t.Fatalf("task = %+v, want solved answer %q", task, test.answer)
			}
		})
	}
}

func TestNormalizeDerivativeEdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		prompt string
		answer string
	}{
		{
			name:   "bare square",
			input:  "What is the derivative of x^2?",
			prompt: "Derrivative: x^2 ",
			answer: "2x",
		},
		{
			name:   "linear expression",
			input:  "What is the derivative of 3x + 2?",
			prompt: "Derrivative: 3x + 2 ",
			answer: "3",
		},
		{
			name:   "constant",
			input:  "Derivative: 5",
			prompt: "Derrivative: 5 ",
			answer: "0",
		},
		{
			name:   "negative leading coefficient",
			input:  "What is the derivative of -x^3 + 4x - 9?",
			prompt: "Derrivative: -x^3 + 4x - 9 ",
			answer: "-3x^2 + 4",
		},
		{
			name:   "implicit negative coefficient",
			input:  "What is the derivative of x^3 - x?",
			prompt: "Derrivative: x^3 - x ",
			answer: "3x^2 - 1",
		},
		{
			name:   "like terms",
			input:  "What is the derivative of 2x^2 + 3x^2 + x?",
			prompt: "Derrivative: 2x^2 + 3x^2 + x ",
			answer: "10x + 1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			task, ok := Normalize(test.input)
			if !ok {
				t.Fatal("Normalize returned false")
			}
			if task.Route != RouteDerivative || task.Prompt != test.prompt {
				t.Fatalf("task = %+v, want derivative prompt %q", task, test.prompt)
			}
			if !task.Solved || task.Answer != test.answer {
				t.Fatalf("task = %+v, want solved answer %q", task, test.answer)
			}
		})
	}
}

func TestRouterUsesDeterministicDerivativeByDefaultWhenSupported(t *testing.T) {
	arithmetic := &recordingGenerator{output: "unused"}
	derivative := &recordingGenerator{output: "Derrivative: x^2 2x\nextra"}
	router := Router{
		Arithmetic: arithmetic,
		Derivative: derivative,
	}

	output, err := router.GenerateWithOptions("User: What is the derivative of x^2?\n\nAssistant:", runtime.GenerateOptions{
		MaxTokens:   64,
		TopK:        40,
		Temperature: 0.7,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if arithmetic.prompt != "" {
		t.Fatalf("arithmetic prompt = %q, want unused", arithmetic.prompt)
	}
	if derivative.prompt != "" {
		t.Fatalf("derivative prompt = %q, want deterministic solver to bypass model", derivative.prompt)
	}
	if output != "User: What is the derivative of x^2?\n\nAssistant:2x" {
		t.Fatalf("output = %q, want original prompt plus answer", output)
	}
}

func TestRouterUsesModelFirstWhenAnswerMatches(t *testing.T) {
	arithmetic := &recordingGenerator{output: "7 * 8 = 56\nextra"}
	router := Router{
		Arithmetic:  arithmetic,
		PreferModel: true,
	}

	output, err := router.GenerateWithOptions("User: What is 7 * 8?\n\nAssistant:", runtime.GenerateOptions{
		MaxTokens:   64,
		TopK:        40,
		Temperature: 0.7,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if arithmetic.prompt != "7 * 8 = " {
		t.Fatalf("arithmetic prompt = %q, want normalized model prompt", arithmetic.prompt)
	}
	if arithmetic.options.TopK != 1 || arithmetic.options.Temperature != 0 {
		t.Fatalf("options = %+v, want forced greedy decoding", arithmetic.options)
	}
	if output != "User: What is 7 * 8?\n\nAssistant:56" {
		t.Fatalf("output = %q, want model answer", output)
	}
}

func TestRouterFallsBackToDeterministicWhenModelAnswerDoesNotMatch(t *testing.T) {
	arithmetic := &recordingGenerator{output: "7 * 8 = 55\nextra"}
	router := Router{
		Arithmetic:  arithmetic,
		PreferModel: true,
	}

	output, err := router.GenerateWithOptions("User: What is 7 * 8?\n\nAssistant:", runtime.GenerateOptions{})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if arithmetic.prompt != "7 * 8 = " {
		t.Fatalf("arithmetic prompt = %q, want normalized model prompt", arithmetic.prompt)
	}
	if output != "User: What is 7 * 8?\n\nAssistant:56" {
		t.Fatalf("output = %q, want deterministic fallback answer", output)
	}
}

func TestRouterFallsBackToDerivativeGeneratorWhenUnsupported(t *testing.T) {
	derivative := &recordingGenerator{output: "Derrivative: sin(x) cos(x)\nextra"}
	router := Router{
		Derivative: derivative,
	}

	output, err := router.GenerateWithOptions("User: What is the derivative of sin(x)?\n\nAssistant:", runtime.GenerateOptions{
		MaxTokens:   64,
		TopK:        40,
		Temperature: 0.7,
	})
	if err != nil {
		t.Fatalf("GenerateWithOptions error: %v", err)
	}
	if derivative.prompt != "Derrivative: sin(x) " {
		t.Fatalf("derivative prompt = %q, want fallback prompt", derivative.prompt)
	}
	if derivative.options.TopK != 1 || derivative.options.Temperature != 0 {
		t.Fatalf("options = %+v, want forced greedy decoding", derivative.options)
	}
	if output != "User: What is the derivative of sin(x)?\n\nAssistant:cos(x)" {
		t.Fatalf("output = %q, want original prompt plus fallback answer", output)
	}
}

type recordingGenerator struct {
	prompt  string
	options runtime.GenerateOptions
	output  string
}

func (g *recordingGenerator) GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error) {
	g.prompt = prompt
	g.options = options
	return g.output, nil
}
