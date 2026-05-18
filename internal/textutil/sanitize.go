package textutil

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

const PrototypeFallback = "Aurelius prototype note: the current toy model emitted non-text tokens, so there is no readable assistant reply yet."

// SanitizeVisibleText keeps only display-safe runes for local presentation.
func SanitizeVisibleText(value string) string {
	value = strings.ToValidUTF8(value, "")

	var builder strings.Builder
	builder.Grow(len(value))

	for _, r := range value {
		switch {
		case r == '\n' || r == '\t' || r == ' ':
			builder.WriteRune(r)
		case r == utf8.RuneError:
			continue
		case unicode.IsGraphic(r):
			builder.WriteRune(r)
		}
	}

	return strings.TrimSpace(builder.String())
}

func SanitizeVisibleOrFallback(value string) string {
	sanitized := SanitizeVisibleText(value)
	if sanitized == "" {
		return PrototypeFallback
	}
	return sanitized
}
