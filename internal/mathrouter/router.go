package mathrouter

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/augahmed/aurelius/internal/runtime"
)

type Generator interface {
	GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error)
}

type Route string

const (
	RouteArithmetic Route = "arithmetic"
	RouteDerivative Route = "derivative"
)

type Task struct {
	Route  Route
	Prompt string
	Answer string
	Solved bool
}

type Router struct {
	Arithmetic  Generator
	Derivative  Generator
	PreferModel bool
}

var (
	arithmeticPattern = regexp.MustCompile(`(?i)(-?\d+)\s*(\+|-|\*|x|×|plus|minus|times|multiplied\s+by)\s*(-?\d+)`)
	derivativePattern = regexp.MustCompile(`(?i)(?:derivative|derrivative)\s*(?:of)?\s*:?[\s]*(.+)$`)
)

func Normalize(input string) (Task, bool) {
	text := latestUserText(input)
	if text == "" {
		text = strings.TrimSpace(input)
	}
	text = strings.TrimSpace(strings.TrimSuffix(text, "?"))
	if text == "" {
		return Task{}, false
	}

	if expr, ok := extractDerivativeExpression(text); ok {
		answer, solved := solvePolynomialDerivative(expr)
		return Task{
			Route:  RouteDerivative,
			Prompt: "Derrivative: " + expr + " ",
			Answer: answer,
			Solved: solved,
		}, true
	}

	if match := arithmeticPattern.FindStringSubmatch(text); len(match) == 4 {
		op := normalizeOperator(match[2])
		answer, solved := solveArithmetic(match[1], op, match[3])
		return Task{
			Route:  RouteArithmetic,
			Prompt: match[1] + " " + op + " " + match[3] + " = ",
			Answer: answer,
			Solved: solved,
		}, true
	}

	return Task{}, false
}

func (r Router) GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error) {
	task, ok := Normalize(prompt)
	if !ok {
		return "", fmt.Errorf("could not recognize a supported math question")
	}
	generator := r.Arithmetic
	if task.Route == RouteDerivative {
		generator = r.Derivative
	}
	if task.Solved && (!r.PreferModel || generator == nil) {
		return prompt + task.Answer, nil
	}
	if generator == nil {
		return "", fmt.Errorf("%s generator is not configured", task.Route)
	}

	completion, err := r.generateTaskCompletion(generator, task, options)
	if err != nil {
		if task.Solved {
			return prompt + task.Answer, nil
		}
		return "", err
	}
	if task.Solved && !answersMatch(completion, task.Answer) {
		return prompt + task.Answer, nil
	}
	return prompt + completion, nil
}

func (r Router) generateTaskCompletion(generator Generator, task Task, options runtime.GenerateOptions) (string, error) {
	innerOptions := options
	innerOptions.TopK = 1
	innerOptions.Temperature = 0
	if innerOptions.MaxTokens <= 0 {
		innerOptions.MaxTokens = 24
	}
	innerOptions.StopTokens = append([]int(nil), innerOptions.StopTokens...)
	innerOptions.StopTokens = append(innerOptions.StopTokens, int('\n'))
	output, err := generator.GenerateWithOptions(task.Prompt, innerOptions)
	if err != nil {
		return "", err
	}
	return firstAnswerLine(extractCompletion(task.Prompt, output)), nil
}

func latestUserText(input string) string {
	text := strings.TrimSpace(input)
	idx := strings.LastIndex(text, "User:")
	if idx < 0 {
		return ""
	}
	text = text[idx+len("User:"):]
	if assistant := strings.Index(text, "Assistant:"); assistant >= 0 {
		text = text[:assistant]
	}
	return strings.TrimSpace(text)
}

func extractDerivativeExpression(text string) (string, bool) {
	match := derivativePattern.FindStringSubmatch(text)
	if len(match) != 2 {
		return "", false
	}
	expr := strings.TrimSpace(match[1])
	expr = strings.TrimSuffix(expr, "?")
	expr = strings.TrimSpace(expr)
	return expr, expr != ""
}

func normalizeOperator(op string) string {
	normalized := strings.ToLower(strings.Join(strings.Fields(op), " "))
	switch normalized {
	case "x", "×", "times", "multiplied by":
		return "*"
	case "plus":
		return "+"
	case "minus":
		return "-"
	default:
		return op
	}
}

func solveArithmetic(left string, op string, right string) (string, bool) {
	a, err := strconv.Atoi(left)
	if err != nil {
		return "", false
	}
	b, err := strconv.Atoi(right)
	if err != nil {
		return "", false
	}
	switch op {
	case "+":
		return strconv.Itoa(a + b), true
	case "-":
		return strconv.Itoa(a - b), true
	case "*":
		return strconv.Itoa(a * b), true
	default:
		return "", false
	}
}

func solvePolynomialDerivative(expr string) (string, bool) {
	terms, ok := parsePolynomial(expr)
	if !ok {
		return "", false
	}
	derived := make(map[int]int)
	for power, coeff := range terms {
		if power == 0 || coeff == 0 {
			continue
		}
		derived[power-1] += coeff * power
	}
	return formatPolynomial(derived), true
}

func parsePolynomial(expr string) (map[int]int, bool) {
	expr = strings.ReplaceAll(expr, " ", "")
	if expr == "" {
		return nil, false
	}
	expr = strings.ReplaceAll(expr, "-", "+-")
	expr = strings.TrimPrefix(expr, "+")
	parts := strings.Split(expr, "+")
	terms := make(map[int]int)
	for _, part := range parts {
		if part == "" {
			continue
		}
		coeff, power, ok := parsePolynomialTerm(part)
		if !ok {
			return nil, false
		}
		terms[power] += coeff
	}
	return terms, len(terms) > 0
}

func parsePolynomialTerm(term string) (int, int, bool) {
	if !strings.Contains(term, "x") {
		coeff, err := strconv.Atoi(term)
		return coeff, 0, err == nil
	}
	coeffText, powerText, hasPower := strings.Cut(term, "x^")
	if !hasPower {
		coeffText = strings.TrimSuffix(term, "x")
		powerText = "1"
	}
	if strings.Contains(coeffText, "x") || strings.Contains(powerText, "x") {
		return 0, 0, false
	}
	coeff, ok := parseCoefficient(coeffText)
	if !ok {
		return 0, 0, false
	}
	power, err := strconv.Atoi(powerText)
	if err != nil || power < 0 {
		return 0, 0, false
	}
	return coeff, power, true
}

func parseCoefficient(value string) (int, bool) {
	switch value {
	case "", "+":
		return 1, true
	case "-":
		return -1, true
	default:
		coeff, err := strconv.Atoi(value)
		return coeff, err == nil
	}
}

func formatPolynomial(terms map[int]int) string {
	powers := make([]int, 0, len(terms))
	for power, coeff := range terms {
		if coeff != 0 {
			powers = append(powers, power)
		}
	}
	if len(powers) == 0 {
		return "0"
	}
	sort.Sort(sort.Reverse(sort.IntSlice(powers)))
	parts := make([]string, 0, len(powers))
	for _, power := range powers {
		coeff := terms[power]
		parts = append(parts, formatPolynomialTerm(coeff, power, len(parts) == 0))
	}
	return strings.Join(parts, " ")
}

func formatPolynomialTerm(coeff int, power int, first bool) string {
	sign := "+"
	if coeff < 0 {
		sign = "-"
		coeff = -coeff
	}
	body := strconv.Itoa(coeff)
	if power == 1 {
		if coeff == 1 {
			body = "x"
		} else {
			body = strconv.Itoa(coeff) + "x"
		}
	} else if power > 1 {
		if coeff == 1 {
			body = "x^" + strconv.Itoa(power)
		} else {
			body = strconv.Itoa(coeff) + "x^" + strconv.Itoa(power)
		}
	}
	if first {
		if sign == "-" {
			return "-" + body
		}
		return body
	}
	return sign + " " + body
}

func extractCompletion(prompt, output string) string {
	if strings.HasPrefix(output, prompt) {
		return output[len(prompt):]
	}
	return output
}

func firstAnswerLine(value string) string {
	value = strings.TrimSpace(value)
	if idx := strings.Index(value, "\n"); idx >= 0 {
		value = value[:idx]
	}
	return strings.TrimSpace(value)
}

func answersMatch(generated, expected string) bool {
	return normalizeAnswer(generated) == normalizeAnswer(expected)
}

func normalizeAnswer(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}
