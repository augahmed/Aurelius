package tokenizer

import "testing"

func TestByteTokenizerRoundTrip(t *testing.T) {
	tok := NewByteTokenizer()
	input := "hello world"
	encoded, err := tok.Encode(input)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	decoded, err := tok.Decode(encoded)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != input {
		t.Fatalf("decoded text = %q, want %q", decoded, input)
	}
}

func TestByteTokenizerDecodeInvalidToken(t *testing.T) {
	tok := NewByteTokenizer()
	if _, err := tok.Decode([]int{256}); err == nil {
		t.Fatal("expected invalid token error")
	}
}
