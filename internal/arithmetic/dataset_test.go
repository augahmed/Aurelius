package arithmetic

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

func TestGenerateDatasetAndLoadExamples(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		MinOperand: 0,
		MaxOperand: 9,
		Operations: []string{"add", "sub"},
		Seed:       7,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}

	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples(train) error: %v", err)
	}
	val, err := LoadExamples(filepath.Join(dir, "val.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples(val) error: %v", err)
	}
	if len(train) != 4 {
		t.Fatalf("len(train) = %d, want 4", len(train))
	}
	if len(val) != 2 {
		t.Fatalf("len(val) = %d, want 2", len(val))
	}
	for _, example := range train {
		if strings.TrimSpace(example.Prompt) == "" || strings.TrimSpace(example.Completion) == "" {
			t.Fatalf("invalid example: %+v", example)
		}
	}
}

func TestLoadExamplesBackfillsAnswerMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.jsonl")
	raw := `{"prompt":"29 - 21 = ","completion":"8","operation":"sub","level":2,"min_operand":10,"max_operand":99,"template":"equation"}` + "\n"
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}
	examples, err := LoadExamples(path)
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	if examples[0].AnswerDigits != 1 {
		t.Fatalf("answer digits = %d, want 1", examples[0].AnswerDigits)
	}
	if !examples[0].SmallDifference {
		t.Fatalf("expected small difference metadata: %+v", examples[0])
	}
}

func TestBuildTrainingSequences(t *testing.T) {
	examples := []Example{{
		Prompt:     "2 + 3 = ",
		Completion: "5",
		Operation:  "add",
	}}
	sequences, err := BuildTrainingSequences(examples, tokenizer.NewByteTokenizer(), 4)
	if err != nil {
		t.Fatalf("BuildTrainingSequences error: %v", err)
	}
	if len(sequences) == 0 {
		t.Fatal("expected at least one training sequence")
	}
	if len(sequences) != 2 {
		t.Fatalf("len(sequences) = %d, want completion plus newline targets", len(sequences))
	}
	if sequences[0].Target != int('5') {
		t.Fatalf("first target = %d, want completion token", sequences[0].Target)
	}
	last := sequences[len(sequences)-1]
	if last.Target != int('\n') {
		t.Fatalf("last target = %d, want newline token", last.Target)
	}
	if len(last.Context) != 4 {
		t.Fatalf("context length = %d, want 4", len(last.Context))
	}
}

func TestGenerateDatasetIncludesCurriculumMetadata(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 36,
		ValCount:   18,
		Operations: []string{"add", "sub"},
		Levels:     []int{1, 2, 3},
		Seed:       11,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}

	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	seen := map[int]bool{}
	seenPairs := map[string]bool{}
	for _, example := range train {
		seen[example.Level] = true
		seenPairs[fmt.Sprintf("%d/%s", example.Level, example.Operation)] = true
		if example.Template == "" {
			t.Fatalf("example missing template: %+v", example)
		}
		if example.MinOperand > example.MaxOperand {
			t.Fatalf("invalid operand metadata: %+v", example)
		}
		if example.AnswerDigits <= 0 {
			t.Fatalf("invalid answer digit metadata: %+v", example)
		}
		if example.Level == 3 && example.Operation == "add" && !example.RequiresCarry {
			t.Fatalf("level 3 addition should require carry: %+v", example)
		}
		if example.Level == 3 && example.Operation == "sub" && !example.RequiresBorrow {
			t.Fatalf("level 3 subtraction should require borrow: %+v", example)
		}
		if example.Level == 2 && (example.RequiresCarry || example.RequiresBorrow) {
			t.Fatalf("level 2 should avoid carry and borrow: %+v", example)
		}
	}
	for _, level := range []int{1, 2, 3} {
		if !seen[level] {
			t.Fatalf("missing level %d in generated dataset", level)
		}
	}
	for _, pair := range []string{"1/add", "1/sub", "2/add", "2/sub", "3/add", "3/sub"} {
		if !seenPairs[pair] {
			t.Fatalf("missing level/operation pair %s in generated dataset", pair)
		}
	}
}

func TestGenerateDatasetFiltersSmallDifferenceSubtraction(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount:          20,
		ValCount:            10,
		Operations:          []string{"sub"},
		Levels:              []int{2},
		AnswerDigits:        []int{1},
		SmallDifferenceOnly: true,
		Seed:                19,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	for _, example := range train {
		if example.Operation != "sub" || example.Level != 2 {
			t.Fatalf("unexpected task: %+v", example)
		}
		if example.AnswerDigits != 1 {
			t.Fatalf("answer digits = %d, want 1: %+v", example.AnswerDigits, example)
		}
		if !example.SmallDifference {
			t.Fatalf("expected small difference metadata: %+v", example)
		}
	}
}

func TestGenerateDatasetFiltersQuestionTemplate(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 9,
		ValCount:   3,
		Operations: []string{"add", "sub"},
		Levels:     []int{2},
		Templates:  []string{"question"},
		Seed:       23,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	for _, example := range train {
		if example.Template != "question" {
			t.Fatalf("template = %q, want question: %+v", example.Template, example)
		}
		if !strings.HasPrefix(example.Prompt, "What is ") {
			t.Fatalf("prompt = %q, want question prompt", example.Prompt)
		}
	}
}

func TestGenerateDatasetFiltersMultipleTemplates(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 12,
		ValCount:   4,
		Operations: []string{"add"},
		Levels:     []int{1},
		Templates:  []string{"equation", "solve"},
		Seed:       29,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	seen := map[string]bool{}
	for _, example := range train {
		if example.Template == "question" {
			t.Fatalf("unexpected question template: %+v", example)
		}
		seen[example.Template] = true
	}
	for _, template := range []string{"equation", "solve"} {
		if !seen[template] {
			t.Fatalf("missing template %q", template)
		}
	}
}

func TestGenerateDatasetFiltersTemplateWithAnswerShape(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount:          12,
		ValCount:            4,
		Operations:          []string{"sub"},
		Levels:              []int{2},
		Templates:           []string{"question"},
		AnswerDigits:        []int{1},
		SmallDifferenceOnly: true,
		Seed:                31,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	for _, example := range train {
		if example.Template != "question" || example.AnswerDigits != 1 || !example.SmallDifference {
			t.Fatalf("unexpected filtered example: %+v", example)
		}
	}
}

func TestGenerateDatasetWorkedReasoningStyle(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount:     12,
		ValCount:       4,
		Operations:     []string{"add", "sub"},
		Levels:         []int{3},
		Templates:      []string{"equation"},
		ReasoningStyle: "worked",
		Seed:           37,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	for _, example := range train {
		if example.ReasoningStyle != "worked" {
			t.Fatalf("reasoning style = %q, want worked: %+v", example.ReasoningStyle, example)
		}
		if example.Answer == "" {
			t.Fatalf("missing final answer: %+v", example)
		}
		if !strings.Contains(example.Completion, "answer: "+example.Answer) {
			t.Fatalf("completion = %q, want answer marker %q", example.Completion, example.Answer)
		}
		if FinalAnswer(example) != example.Answer {
			t.Fatalf("FinalAnswer = %q, want %q", FinalAnswer(example), example.Answer)
		}
		if !strings.Contains(example.Completion, "ones:") {
			t.Fatalf("completion = %q, want worked digit step", example.Completion)
		}
	}
}

func TestGenerateDatasetCompactReasoningStyle(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount:     12,
		ValCount:       4,
		Operations:     []string{"add", "sub"},
		Levels:         []int{3},
		Templates:      []string{"equation"},
		ReasoningStyle: "compact",
		Seed:           41,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	for _, example := range train {
		if example.ReasoningStyle != "compact" {
			t.Fatalf("reasoning style = %q, want compact: %+v", example.ReasoningStyle, example)
		}
		if example.Answer == "" {
			t.Fatalf("missing final answer: %+v", example)
		}
		if !strings.Contains(example.Completion, "ans:"+example.Answer) {
			t.Fatalf("completion = %q, want compact answer marker %q", example.Completion, example.Answer)
		}
		if FinalAnswer(example) != example.Answer {
			t.Fatalf("FinalAnswer = %q, want %q", FinalAnswer(example), example.Answer)
		}
		if !strings.Contains(example.Completion, "o:") {
			t.Fatalf("completion = %q, want compact digit step", example.Completion)
		}
	}
}

func TestGenerateDatasetDerivativeLevel(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount: 12,
		ValCount:   4,
		Operations: []string{"derivative"},
		Levels:     []int{7},
		Templates:  []string{"equation", "question"},
		Seed:       43,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	seen := map[string]bool{}
	for _, example := range train {
		if example.Operation != "derivative" || example.Level != 7 {
			t.Fatalf("unexpected derivative example: %+v", example)
		}
		if example.Template != "equation" && example.Template != "question" {
			t.Fatalf("template = %q, want equation or question", example.Template)
		}
		if example.Template == "equation" && !strings.HasPrefix(example.Prompt, "Derrivative: ") {
			t.Fatalf("prompt = %q, want Derrivative frame", example.Prompt)
		}
		if example.Template == "question" && !strings.HasPrefix(example.Prompt, "What is the derrivative of ") {
			t.Fatalf("prompt = %q, want derrivative question frame", example.Prompt)
		}
		if !strings.Contains(example.Prompt, "x") {
			t.Fatalf("prompt = %q, want polynomial expression", example.Prompt)
		}
		if example.Answer == "" || example.Completion != example.Answer {
			t.Fatalf("completion/answer mismatch: %+v", example)
		}
		seen[example.Template] = true
	}
	for _, template := range []string{"equation", "question"} {
		if !seen[template] {
			t.Fatalf("missing derivative template %q", template)
		}
	}
}

func TestGenerateDatasetDerivativeFiltersAnswerDigits(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount:   20,
		ValCount:     8,
		Operations:   []string{"derivative"},
		Levels:       []int{7},
		Templates:    []string{"equation", "question"},
		AnswerDigits: []int{1, 2},
		Seed:         47,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	for _, example := range train {
		if example.Operation != "derivative" || example.Level != 7 {
			t.Fatalf("unexpected derivative example: %+v", example)
		}
		if example.AnswerDigits != 1 && example.AnswerDigits != 2 {
			t.Fatalf("answer digits = %d, want 1 or 2: %+v", example.AnswerDigits, example)
		}
	}
}

func TestGenerateDatasetDerivativeCoefficientReasoningStyle(t *testing.T) {
	dir := t.TempDir()
	cfg := GenerateConfig{
		TrainCount:     20,
		ValCount:       8,
		Operations:     []string{"derivative"},
		Levels:         []int{7},
		Templates:      []string{"equation", "question"},
		ReasoningStyle: "coefficients",
		Seed:           53,
	}
	if err := GenerateDataset(dir, cfg); err != nil {
		t.Fatalf("GenerateDataset error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(dir, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples error: %v", err)
	}
	seenComma := false
	for _, example := range train {
		if example.Operation != "derivative" || example.Level != 7 {
			t.Fatalf("unexpected derivative example: %+v", example)
		}
		if example.ReasoningStyle != "coefficients" {
			t.Fatalf("reasoning style = %q, want coefficients: %+v", example.ReasoningStyle, example)
		}
		if example.Completion != example.Answer {
			t.Fatalf("completion/answer mismatch: %+v", example)
		}
		if strings.ContainsAny(example.Answer, "x^+ ") {
			t.Fatalf("answer = %q, want coefficient vector", example.Answer)
		}
		for _, part := range strings.Split(example.Answer, ",") {
			if part == "" {
				t.Fatalf("answer = %q, has empty coefficient", example.Answer)
			}
			for _, char := range part {
				if char < '0' || char > '9' {
					t.Fatalf("answer = %q, want digits and commas only", example.Answer)
				}
			}
		}
		if strings.Contains(example.Answer, ",") {
			seenComma = true
		}
	}
	if !seenComma {
		t.Fatal("expected at least one multi-coefficient derivative answer")
	}
}

func TestGenerateDatasetDerivativeRejectsSolveOnly(t *testing.T) {
	err := GenerateDataset(t.TempDir(), GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		Operations: []string{"derivative"},
		Levels:     []int{7},
		Templates:  []string{"solve"},
		Seed:       1,
	})
	if err == nil {
		t.Fatal("expected derivative solve-only generation error")
	}
}

func TestExtractFinalAnswer(t *testing.T) {
	tests := []struct {
		text string
		want string
	}{
		{text: "68", want: "68"},
		{text: "ones: 7+1=8; tens: 5+1=6; answer: 68", want: "68"},
		{text: "o:7+1=8; t:5+1=6; ans:68", want: "68"},
		{text: "steps; Answer: 105 extra", want: "105"},
	}
	for _, test := range tests {
		if got := ExtractFinalAnswer(test.text); got != test.want {
			t.Fatalf("ExtractFinalAnswer(%q) = %q, want %q", test.text, got, test.want)
		}
	}
}

func TestGenerateDatasetRejectsInvalidTemplate(t *testing.T) {
	err := GenerateDataset(t.TempDir(), GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		Operations: []string{"add"},
		Levels:     []int{1},
		Templates:  []string{"story"},
		Seed:       1,
	})
	if err == nil {
		t.Fatal("expected invalid template error")
	}
}

func TestGenerateDatasetRejectsInvalidReasoningStyle(t *testing.T) {
	err := GenerateDataset(t.TempDir(), GenerateConfig{
		TrainCount:     4,
		ValCount:       2,
		Operations:     []string{"add"},
		Levels:         []int{1},
		ReasoningStyle: "proof",
		Seed:           1,
	})
	if err == nil {
		t.Fatal("expected invalid reasoning style error")
	}
}

func TestMixDatasetsAppliesWeights(t *testing.T) {
	root := t.TempDir()
	sourceA := filepath.Join(root, "a")
	sourceB := filepath.Join(root, "b")
	output := filepath.Join(root, "mixed")
	if err := os.MkdirAll(sourceA, 0o755); err != nil {
		t.Fatalf("MkdirAll sourceA error: %v", err)
	}
	if err := os.MkdirAll(sourceB, 0o755); err != nil {
		t.Fatalf("MkdirAll sourceB error: %v", err)
	}
	if err := writeJSONL(filepath.Join(sourceA, "train.jsonl"), []Example{{Prompt: "1 + 1 = ", Completion: "2", Operation: "add", Level: 1, Template: "equation", AnswerDigits: 1}}); err != nil {
		t.Fatalf("write sourceA train error: %v", err)
	}
	if err := writeJSONL(filepath.Join(sourceA, "val.jsonl"), []Example{{Prompt: "2 + 1 = ", Completion: "3", Operation: "add", Level: 1, Template: "equation", AnswerDigits: 1}}); err != nil {
		t.Fatalf("write sourceA val error: %v", err)
	}
	if err := writeJSONL(filepath.Join(sourceB, "train.jsonl"), []Example{{Prompt: "9 - 8 = ", Completion: "1", Operation: "sub", Level: 2, Template: "equation", AnswerDigits: 1, SmallDifference: true}}); err != nil {
		t.Fatalf("write sourceB train error: %v", err)
	}
	if err := writeJSONL(filepath.Join(sourceB, "val.jsonl"), []Example{{Prompt: "8 - 7 = ", Completion: "1", Operation: "sub", Level: 2, Template: "equation", AnswerDigits: 1, SmallDifference: true}}); err != nil {
		t.Fatalf("write sourceB val error: %v", err)
	}

	if err := MixDatasets(output, MixConfig{
		Sources: []MixSource{
			{Path: sourceA, Weight: 1},
			{Path: sourceB, Weight: 2},
		},
		Seed: 3,
	}); err != nil {
		t.Fatalf("MixDatasets error: %v", err)
	}
	train, err := LoadExamples(filepath.Join(output, "train.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples train error: %v", err)
	}
	val, err := LoadExamples(filepath.Join(output, "val.jsonl"))
	if err != nil {
		t.Fatalf("LoadExamples val error: %v", err)
	}
	if len(train) != 3 {
		t.Fatalf("len(train) = %d, want 3", len(train))
	}
	if len(val) != 3 {
		t.Fatalf("len(val) = %d, want 3", len(val))
	}
	subCount := 0
	for _, example := range train {
		if example.Operation == "sub" {
			subCount++
		}
	}
	if subCount != 2 {
		t.Fatalf("sub train count = %d, want 2", subCount)
	}
}

func TestGenerateDatasetRejectsIncompatibleLevelAndOperation(t *testing.T) {
	err := GenerateDataset(t.TempDir(), GenerateConfig{
		TrainCount: 4,
		ValCount:   2,
		Operations: []string{"mul"},
		Levels:     []int{1},
		Seed:       1,
	})
	if err == nil {
		t.Fatal("expected incompatible level and operation error")
	}
}
