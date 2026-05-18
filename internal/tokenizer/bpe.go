package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

var gpt2TokenPattern = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

type MergeRule struct {
	Left  string
	Right string
}

type bpePair struct {
	left  string
	right string
}

// BPETokenizer implements a GPT-2 style byte-pair encoding tokenizer.
type BPETokenizer struct {
	vocab      map[string]int
	inverse    map[int]string
	mergeRanks map[bpePair]int
	cache      map[string][]string
	byteEncode [256]string
	byteDecode map[rune]byte
}

func NewBPETokenizer(vocab map[string]int, merges []MergeRule) (*BPETokenizer, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("vocabulary is required")
	}

	inverse := make(map[int]string, len(vocab))
	for token, id := range vocab {
		if id < 0 {
			return nil, fmt.Errorf("token %q has negative id %d", token, id)
		}
		if existing, ok := inverse[id]; ok {
			return nil, fmt.Errorf("duplicate token id %d for %q and %q", id, existing, token)
		}
		inverse[id] = token
	}

	mergeRanks := make(map[bpePair]int, len(merges))
	for i, merge := range merges {
		if merge.Left == "" || merge.Right == "" {
			return nil, fmt.Errorf("merge rule %d must contain two non-empty tokens", i)
		}
		pair := bpePair{left: merge.Left, right: merge.Right}
		if _, ok := mergeRanks[pair]; ok {
			continue
		}
		mergeRanks[pair] = i
	}

	byteEncode, byteDecode := buildByteAlphabet()
	return &BPETokenizer{
		vocab:      cloneVocab(vocab),
		inverse:    inverse,
		mergeRanks: mergeRanks,
		cache:      make(map[string][]string),
		byteEncode: byteEncode,
		byteDecode: byteDecode,
	}, nil
}

func LoadBPETokenizer(vocabPath, mergesPath string) (*BPETokenizer, error) {
	vocabBytes, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("read vocab %q: %w", vocabPath, err)
	}

	var vocab map[string]int
	if err := json.Unmarshal(vocabBytes, &vocab); err != nil {
		return nil, fmt.Errorf("parse vocab %q: %w", vocabPath, err)
	}

	mergesFile, err := os.Open(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("open merges %q: %w", mergesPath, err)
	}
	defer mergesFile.Close()

	var merges []MergeRule
	scanner := bufio.NewScanner(mergesFile)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) != 2 {
			return nil, fmt.Errorf("parse merges %q line %d: expected 2 fields, got %d", mergesPath, lineNumber, len(fields))
		}
		merges = append(merges, MergeRule{
			Left:  fields[0],
			Right: fields[1],
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan merges %q: %w", mergesPath, err)
	}

	return NewBPETokenizer(vocab, merges)
}

func (t *BPETokenizer) Encode(text string) ([]int, error) {
	pieces := gpt2TokenPattern.FindAllString(text, -1)
	tokens := make([]int, 0, len(pieces))

	for _, piece := range pieces {
		encodedPiece := t.encodePiece(piece)
		for _, symbol := range t.mergeSymbols(encodedPiece) {
			id, ok := t.vocab[symbol]
			if !ok {
				return nil, fmt.Errorf("token %q not found in vocabulary", symbol)
			}
			tokens = append(tokens, id)
		}
	}

	return tokens, nil
}

func (t *BPETokenizer) Decode(tokens []int) (string, error) {
	var encoded strings.Builder
	for i, token := range tokens {
		value, ok := t.inverse[token]
		if !ok {
			return "", fmt.Errorf("token %d at position %d out of range", token, i)
		}
		encoded.WriteString(value)
	}

	bytes := make([]byte, 0, encoded.Len())
	for _, r := range encoded.String() {
		b, ok := t.byteDecode[r]
		if !ok {
			return "", fmt.Errorf("decoded rune %q has no byte mapping", r)
		}
		bytes = append(bytes, b)
	}

	return string(bytes), nil
}

func (t *BPETokenizer) VocabSize() int {
	return len(t.vocab)
}

func (t *BPETokenizer) encodePiece(piece string) string {
	var builder strings.Builder
	for _, b := range []byte(piece) {
		builder.WriteString(t.byteEncode[b])
	}
	return builder.String()
}

func (t *BPETokenizer) mergeSymbols(piece string) []string {
	if cached, ok := t.cache[piece]; ok {
		return cloneStrings(cached)
	}

	symbols := splitSymbols(piece)
	if len(symbols) < 2 {
		t.cache[piece] = cloneStrings(symbols)
		return cloneStrings(symbols)
	}

	for {
		bestPair, found := t.bestPair(symbols)
		if !found {
			break
		}
		symbols = mergePair(symbols, bestPair)
		if len(symbols) < 2 {
			break
		}
	}

	t.cache[piece] = cloneStrings(symbols)
	return cloneStrings(symbols)
}

func (t *BPETokenizer) bestPair(symbols []string) (bpePair, bool) {
	bestRank := len(t.mergeRanks) + 1
	var best bpePair
	found := false
	for i := 0; i < len(symbols)-1; i++ {
		pair := bpePair{left: symbols[i], right: symbols[i+1]}
		rank, ok := t.mergeRanks[pair]
		if !ok {
			continue
		}
		if !found || rank < bestRank {
			bestRank = rank
			best = pair
			found = true
		}
	}
	return best, found
}

func mergePair(symbols []string, pair bpePair) []string {
	merged := make([]string, 0, len(symbols))
	for i := 0; i < len(symbols); {
		if i < len(symbols)-1 && symbols[i] == pair.left && symbols[i+1] == pair.right {
			merged = append(merged, pair.left+pair.right)
			i += 2
			continue
		}
		merged = append(merged, symbols[i])
		i++
	}
	return merged
}

func splitSymbols(value string) []string {
	symbols := make([]string, 0, len(value))
	for _, r := range value {
		symbols = append(symbols, string(r))
	}
	return symbols
}

func buildByteAlphabet() ([256]string, map[rune]byte) {
	var byteEncode [256]string
	byteDecode := make(map[rune]byte, 256)

	base := make([]int, 0, 256)
	for b := int('!'); b <= int('~'); b++ {
		base = append(base, b)
	}
	for b := int('¡'); b <= int('¬'); b++ {
		base = append(base, b)
	}
	for b := int('®'); b <= int('ÿ'); b++ {
		base = append(base, b)
	}

	seen := make(map[int]struct{}, len(base))
	for _, b := range base {
		seen[b] = struct{}{}
	}

	chars := append([]int(nil), base...)
	extra := 0
	for b := 0; b < 256; b++ {
		if _, ok := seen[b]; ok {
			continue
		}
		base = append(base, b)
		chars = append(chars, 256+extra)
		extra++
	}

	for i, b := range base {
		r := rune(chars[i])
		byteEncode[b] = string(r)
		byteDecode[r] = byte(b)
	}

	return byteEncode, byteDecode
}

func cloneVocab(vocab map[string]int) map[string]int {
	cloned := make(map[string]int, len(vocab))
	for key, value := range vocab {
		cloned[key] = value
	}
	return cloned
}

func cloneStrings(values []string) []string {
	return append([]string(nil), values...)
}
