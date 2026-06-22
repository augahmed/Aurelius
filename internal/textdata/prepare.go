package textdata

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type TextFileSummary struct {
	Path                string `json:"path"`
	Bytes               int    `json:"bytes"`
	Paragraphs          int    `json:"paragraphs"`
	DuplicateParagraphs int    `json:"duplicate_paragraphs"`
}

type TextDatasetReport struct {
	FileCount           int               `json:"file_count"`
	TotalBytes          int               `json:"total_bytes"`
	TotalParagraphs     int               `json:"total_paragraphs"`
	DuplicateParagraphs int               `json:"duplicate_paragraphs"`
	EmptyFiles          int               `json:"empty_files"`
	ShortFiles          []TextFileSummary `json:"short_files"`
	LargestFiles        []TextFileSummary `json:"largest_files"`
	Files               []TextFileSummary `json:"files,omitempty"`
}

type DedupeConfig struct {
	OutputDir         string
	MinParagraphRunes int
}

type DedupeReport struct {
	InputFiles          int `json:"input_files"`
	OutputFiles         int `json:"output_files"`
	InputParagraphs     int `json:"input_paragraphs"`
	OutputParagraphs    int `json:"output_paragraphs"`
	DuplicateParagraphs int `json:"duplicate_paragraphs"`
	TooShortParagraphs  int `json:"too_short_paragraphs"`
	EmptyOutputFiles    int `json:"empty_output_files"`
}

type SplitConfig struct {
	OutputDir string
	ValRatio  float64
	Seed      int64
}

type SplitReport struct {
	InputFiles int `json:"input_files"`
	TrainFiles int `json:"train_files"`
	ValFiles   int `json:"val_files"`
}

func InspectTextDataset(paths []string, shortBytes int) (TextDatasetReport, error) {
	files, err := expandTextPaths(paths)
	if err != nil {
		return TextDatasetReport{}, err
	}
	if shortBytes <= 0 {
		shortBytes = 256
	}
	seenParagraphs := make(map[string]struct{})
	report := TextDatasetReport{
		FileCount:    len(files),
		Files:        make([]TextFileSummary, 0, len(files)),
		LargestFiles: make([]TextFileSummary, 0, min(5, len(files))),
	}
	for _, path := range files {
		text, err := readNormalizedText(path)
		if err != nil {
			return TextDatasetReport{}, err
		}
		paragraphs := splitParagraphs(text)
		summary := TextFileSummary{
			Path:       path,
			Bytes:      len([]byte(text)),
			Paragraphs: len(paragraphs),
		}
		if text == "" {
			report.EmptyFiles++
		}
		if summary.Bytes < shortBytes {
			report.ShortFiles = append(report.ShortFiles, summary)
		}
		for _, paragraph := range paragraphs {
			key := canonicalParagraphKey(paragraph)
			if _, ok := seenParagraphs[key]; ok {
				summary.DuplicateParagraphs++
				report.DuplicateParagraphs++
				continue
			}
			seenParagraphs[key] = struct{}{}
		}
		report.TotalBytes += summary.Bytes
		report.TotalParagraphs += summary.Paragraphs
		report.Files = append(report.Files, summary)
		report.LargestFiles = append(report.LargestFiles, summary)
	}
	slices.SortFunc(report.LargestFiles, func(a, b TextFileSummary) int {
		return b.Bytes - a.Bytes
	})
	if len(report.LargestFiles) > 5 {
		report.LargestFiles = report.LargestFiles[:5]
	}
	return report, nil
}

func DedupeTextDataset(paths []string, cfg DedupeConfig) (DedupeReport, error) {
	if strings.TrimSpace(cfg.OutputDir) == "" {
		return DedupeReport{}, fmt.Errorf("output dir is required")
	}
	if cfg.MinParagraphRunes <= 0 {
		cfg.MinParagraphRunes = 1
	}
	files, err := expandTextPaths(paths)
	if err != nil {
		return DedupeReport{}, err
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return DedupeReport{}, fmt.Errorf("create output dir: %w", err)
	}
	seenParagraphs := make(map[string]struct{})
	report := DedupeReport{InputFiles: len(files)}
	for _, path := range files {
		text, err := readNormalizedText(path)
		if err != nil {
			return DedupeReport{}, err
		}
		paragraphs := splitParagraphs(text)
		report.InputParagraphs += len(paragraphs)
		kept := make([]string, 0, len(paragraphs))
		for _, paragraph := range paragraphs {
			if len([]rune(paragraph)) < cfg.MinParagraphRunes {
				report.TooShortParagraphs++
				continue
			}
			key := canonicalParagraphKey(paragraph)
			if _, ok := seenParagraphs[key]; ok {
				report.DuplicateParagraphs++
				continue
			}
			seenParagraphs[key] = struct{}{}
			kept = append(kept, paragraph)
		}
		if len(kept) == 0 {
			report.EmptyOutputFiles++
			continue
		}
		outPath := filepath.Join(cfg.OutputDir, safeTextFilename(path))
		if err := os.WriteFile(outPath, []byte(strings.Join(kept, "\n\n")+"\n"), 0o644); err != nil {
			return DedupeReport{}, fmt.Errorf("write deduped text %q: %w", outPath, err)
		}
		report.OutputFiles++
		report.OutputParagraphs += len(kept)
	}
	return report, nil
}

func SplitTextDataset(paths []string, cfg SplitConfig) (SplitReport, error) {
	if strings.TrimSpace(cfg.OutputDir) == "" {
		return SplitReport{}, fmt.Errorf("output dir is required")
	}
	if cfg.ValRatio < 0 || cfg.ValRatio >= 1 {
		return SplitReport{}, fmt.Errorf("val ratio must be >= 0 and < 1")
	}
	files, err := expandTextPaths(paths)
	if err != nil {
		return SplitReport{}, err
	}
	indices := make([]int, len(files))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	valCount := int(float64(len(files)) * cfg.ValRatio)
	if cfg.ValRatio > 0 && valCount == 0 && len(files) > 1 {
		valCount = 1
	}
	trainDir := filepath.Join(cfg.OutputDir, "train")
	valDir := filepath.Join(cfg.OutputDir, "val")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		return SplitReport{}, fmt.Errorf("create train dir: %w", err)
	}
	if err := os.MkdirAll(valDir, 0o755); err != nil {
		return SplitReport{}, fmt.Errorf("create val dir: %w", err)
	}

	report := SplitReport{InputFiles: len(files)}
	for order, index := range indices {
		source := files[index]
		targetDir := trainDir
		if order < valCount {
			targetDir = valDir
			report.ValFiles++
		} else {
			report.TrainFiles++
		}
		data, err := os.ReadFile(source)
		if err != nil {
			return SplitReport{}, fmt.Errorf("read text %q: %w", source, err)
		}
		outPath := filepath.Join(targetDir, safeTextFilename(source))
		if err := os.WriteFile(outPath, data, 0o644); err != nil {
			return SplitReport{}, fmt.Errorf("write split text %q: %w", outPath, err)
		}
	}
	return report, nil
}

func readNormalizedText(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read text %q: %w", path, err)
	}
	return NormalizePlainText(string(data)), nil
}

func splitParagraphs(text string) []string {
	blocks := strings.Split(NormalizePlainText(text), "\n\n")
	paragraphs := make([]string, 0, len(blocks))
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block != "" {
			paragraphs = append(paragraphs, block)
		}
	}
	return paragraphs
}

func canonicalParagraphKey(paragraph string) string {
	return strings.ToLower(strings.Join(strings.Fields(paragraph), " "))
}

func safeTextFilename(path string) string {
	base := filepath.Base(path)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	if stem == "" {
		stem = "text"
	}
	if ext == "" {
		ext = ".txt"
	}
	sum := sha1.Sum([]byte(path))
	return stem + "-" + hex.EncodeToString(sum[:])[:10] + ext
}
