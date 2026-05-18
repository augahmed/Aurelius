package textutil

import "testing"

func TestSanitizeVisibleTextRemovesControlRunes(t *testing.T) {
	input := "hello\x02\tworld\n\x00"

	got := SanitizeVisibleText(input)

	if got != "hello\tworld" {
		t.Fatalf("SanitizeVisibleText() = %q, want %q", got, "hello\tworld")
	}
}

func TestSanitizeVisibleOrFallbackReturnsPrototypeMessage(t *testing.T) {
	got := SanitizeVisibleOrFallback("\x02\x03\x00")

	if got != PrototypeFallback {
		t.Fatalf("SanitizeVisibleOrFallback() = %q, want %q", got, PrototypeFallback)
	}
}
