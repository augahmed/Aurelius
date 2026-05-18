package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestBPETokenizerRoundTrip(t *testing.T) {
	tok, err := NewBPETokenizer(testVocab(), testMerges())
	if err != nil {
		t.Fatalf("NewBPETokenizer error: %v", err)
	}

	encoded, err := tok.Encode("hello!")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if !reflect.DeepEqual(encoded, []int{7, 8}) {
		t.Fatalf("encoded tokens = %v, want %v", encoded, []int{7, 8})
	}

	decoded, err := tok.Decode(encoded)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "hello!" {
		t.Fatalf("decoded text = %q, want %q", decoded, "hello!")
	}
}

func TestBPETokenizerSupportsUTF8Bytes(t *testing.T) {
	tok, err := NewBPETokenizer(testVocab(), testMerges())
	if err != nil {
		t.Fatalf("NewBPETokenizer error: %v", err)
	}

	encoded, err := tok.Encode("é")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if !reflect.DeepEqual(encoded, []int{9, 10}) {
		t.Fatalf("encoded tokens = %v, want %v", encoded, []int{9, 10})
	}

	decoded, err := tok.Decode(encoded)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "é" {
		t.Fatalf("decoded text = %q, want %q", decoded, "é")
	}
}

func TestLoadBPETokenizer(t *testing.T) {
	dir := t.TempDir()
	vocabPath := filepath.Join(dir, "vocab.json")
	mergesPath := filepath.Join(dir, "merges.txt")

	if err := os.WriteFile(vocabPath, []byte(`{"h":0,"e":1,"l":2,"o":3,"he":4,"hel":5,"hell":6,"hello":7,"!":8,"Ã":9,"©":10}`), 0o644); err != nil {
		t.Fatalf("WriteFile vocab error: %v", err)
	}
	if err := os.WriteFile(mergesPath, []byte("#version: 0.2\nh e\nhe l\nhel l\nhell o\n"), 0o644); err != nil {
		t.Fatalf("WriteFile merges error: %v", err)
	}

	tok, err := LoadBPETokenizer(vocabPath, mergesPath)
	if err != nil {
		t.Fatalf("LoadBPETokenizer error: %v", err)
	}

	got, err := tok.Encode("hello!")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if !reflect.DeepEqual(got, []int{7, 8}) {
		t.Fatalf("encoded tokens = %v, want %v", got, []int{7, 8})
	}
}

func testVocab() map[string]int {
	return map[string]int{
		"h":     0,
		"e":     1,
		"l":     2,
		"o":     3,
		"he":    4,
		"hel":   5,
		"hell":  6,
		"hello": 7,
		"!":     8,
		"Ã":     9,
		"©":     10,
	}
}

func testMerges() []MergeRule {
	return []MergeRule{
		{Left: "h", Right: "e"},
		{Left: "he", Right: "l"},
		{Left: "hel", Right: "l"},
		{Left: "hell", Right: "o"},
	}
}
