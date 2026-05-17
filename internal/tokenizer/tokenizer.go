package tokenizer

import "fmt"

type Tokenizer interface {
	Encode(text string) ([]int, error)
	Decode(tokens []int) (string, error)
	VocabSize() int
}

type ByteTokenizer struct{}

func NewByteTokenizer() *ByteTokenizer {
	return &ByteTokenizer{}
}

func (t *ByteTokenizer) Encode(text string) ([]int, error) {
	bytes := []byte(text)
	tokens := make([]int, len(bytes))
	for i, b := range bytes {
		tokens[i] = int(b)
	}
	return tokens, nil
}

func (t *ByteTokenizer) Decode(tokens []int) (string, error) {
	bytes := make([]byte, len(tokens))
	for i, token := range tokens {
		if token < 0 || token >= t.VocabSize() {
			return "", fmt.Errorf("token %d at position %d out of range", token, i)
		}
		bytes[i] = byte(token)
	}
	return string(bytes), nil
}

func (t *ByteTokenizer) VocabSize() int {
	return 256
}
