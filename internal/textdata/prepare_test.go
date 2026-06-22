package textdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInspectTextDatasetReportsDuplicatesAndLargestFiles(t *testing.T) {
	dir := t.TempDir()
	writeTextFile(t, filepath.Join(dir, "a.txt"), "Repeated paragraph.\n\nUnique A.\n")
	writeTextFile(t, filepath.Join(dir, "b.txt"), "Repeated paragraph.\n\nUnique B has more content.\n")

	report, err := InspectTextDataset([]string{dir}, 20)
	if err != nil {
		t.Fatalf("InspectTextDataset error: %v", err)
	}
	if report.FileCount != 2 {
		t.Fatalf("FileCount = %d, want 2", report.FileCount)
	}
	if report.TotalParagraphs != 4 {
		t.Fatalf("TotalParagraphs = %d, want 4", report.TotalParagraphs)
	}
	if report.DuplicateParagraphs != 1 {
		t.Fatalf("DuplicateParagraphs = %d, want 1", report.DuplicateParagraphs)
	}
	if len(report.LargestFiles) != 2 || report.LargestFiles[0].Bytes < report.LargestFiles[1].Bytes {
		t.Fatalf("LargestFiles = %+v, want descending byte order", report.LargestFiles)
	}
}

func TestDedupeTextDatasetWritesUniqueParagraphs(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input")
	output := filepath.Join(dir, "output")
	if err := os.MkdirAll(input, 0o755); err != nil {
		t.Fatalf("MkdirAll input error: %v", err)
	}
	writeTextFile(t, filepath.Join(input, "a.txt"), "Repeated paragraph.\n\nUnique A.\n")
	writeTextFile(t, filepath.Join(input, "b.txt"), "Repeated paragraph.\n\nz\n\nUnique B.\n")

	report, err := DedupeTextDataset([]string{input}, DedupeConfig{
		OutputDir:         output,
		MinParagraphRunes: 2,
	})
	if err != nil {
		t.Fatalf("DedupeTextDataset error: %v", err)
	}
	if report.InputFiles != 2 || report.OutputFiles != 2 || report.DuplicateParagraphs != 1 || report.TooShortParagraphs != 1 {
		t.Fatalf("report = %+v, want dedupe/short counts", report)
	}
	text, err := LoadText([]string{output})
	if err != nil {
		t.Fatalf("LoadText output error: %v", err)
	}
	if strings.Count(text, "Repeated paragraph.") != 1 {
		t.Fatalf("deduped text = %q, want repeated paragraph once", text)
	}
	if strings.Contains(text, "\nz\n") {
		t.Fatalf("deduped text = %q, did not expect short paragraph", text)
	}
}

func TestSplitTextDatasetCreatesTrainAndValDirs(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input")
	output := filepath.Join(dir, "split")
	if err := os.MkdirAll(input, 0o755); err != nil {
		t.Fatalf("MkdirAll input error: %v", err)
	}
	for i := 0; i < 10; i++ {
		writeTextFile(t, filepath.Join(input, string(rune('a'+i))+".txt"), "content")
	}

	report, err := SplitTextDataset([]string{input}, SplitConfig{
		OutputDir: output,
		ValRatio:  0.2,
		Seed:      7,
	})
	if err != nil {
		t.Fatalf("SplitTextDataset error: %v", err)
	}
	if report.InputFiles != 10 || report.TrainFiles != 8 || report.ValFiles != 2 {
		t.Fatalf("report = %+v, want 8 train / 2 val", report)
	}
	train, err := filepath.Glob(filepath.Join(output, "train", "*.txt"))
	if err != nil {
		t.Fatalf("Glob train error: %v", err)
	}
	val, err := filepath.Glob(filepath.Join(output, "val", "*.txt"))
	if err != nil {
		t.Fatalf("Glob val error: %v", err)
	}
	if len(train) != 8 || len(val) != 2 {
		t.Fatalf("train=%d val=%d, want 8/2", len(train), len(val))
	}
}

func writeTextFile(t *testing.T, path string, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("WriteFile %q error: %v", path, err)
	}
}
