package textdata

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const defaultIngestUserAgent = "AureliusTextIngest/0.1"

type WebIngestConfig struct {
	OutputDir string
	MaxPages  int
	MaxBytes  int64
	Timeout   time.Duration
	UserAgent string
}

type IngestedDocument struct {
	Source       string `json:"source"`
	Path         string `json:"path"`
	Title        string `json:"title,omitempty"`
	ContentType  string `json:"content_type,omitempty"`
	TextBytes    int    `json:"text_bytes"`
	FetchedBytes int    `json:"fetched_bytes"`
}

func LoadURLList(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open url file %q: %w", path, err)
	}
	defer file.Close()

	urls := make([]string, 0)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		urls = append(urls, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan url file %q: %w", path, err)
	}
	return urls, nil
}

func IngestWebText(ctx context.Context, urls []string, cfg WebIngestConfig) ([]IngestedDocument, error) {
	if strings.TrimSpace(cfg.OutputDir) == "" {
		return nil, fmt.Errorf("output dir is required")
	}
	if len(urls) == 0 {
		return nil, fmt.Errorf("at least one URL is required")
	}
	if cfg.MaxPages <= 0 || cfg.MaxPages > len(urls) {
		cfg.MaxPages = len(urls)
	}
	if cfg.MaxBytes <= 0 {
		cfg.MaxBytes = 2 * 1024 * 1024
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 15 * time.Second
	}
	if cfg.UserAgent == "" {
		cfg.UserAgent = defaultIngestUserAgent
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create output dir: %w", err)
	}

	client := &http.Client{Timeout: cfg.Timeout}
	results := make([]IngestedDocument, 0, cfg.MaxPages)
	for _, rawURL := range urls[:cfg.MaxPages] {
		doc, err := ingestOneURL(ctx, client, rawURL, cfg)
		if err != nil {
			return nil, err
		}
		results = append(results, doc)
	}
	if err := writeIngestMetadata(cfg.OutputDir, results); err != nil {
		return nil, err
	}
	return results, nil
}

func ingestOneURL(ctx context.Context, client *http.Client, rawURL string, cfg WebIngestConfig) (IngestedDocument, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return IngestedDocument{}, fmt.Errorf("parse URL %q: %w", rawURL, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return IngestedDocument{}, fmt.Errorf("URL %q must use http or https", rawURL)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	if err != nil {
		return IngestedDocument{}, fmt.Errorf("create request %q: %w", rawURL, err)
	}
	req.Header.Set("User-Agent", cfg.UserAgent)

	res, err := client.Do(req)
	if err != nil {
		return IngestedDocument{}, fmt.Errorf("fetch %q: %w", rawURL, err)
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return IngestedDocument{}, fmt.Errorf("fetch %q: status %d", rawURL, res.StatusCode)
	}

	body, err := readLimited(res.Body, cfg.MaxBytes)
	if err != nil {
		return IngestedDocument{}, fmt.Errorf("read %q: %w", rawURL, err)
	}
	contentType := res.Header.Get("Content-Type")
	text, title, err := ExtractReadableText(body, contentType)
	if err != nil {
		return IngestedDocument{}, fmt.Errorf("extract %q: %w", rawURL, err)
	}
	if text == "" {
		return IngestedDocument{}, fmt.Errorf("extract %q: empty cleaned text", rawURL)
	}

	name := filenameForURL(parsed)
	path := filepath.Join(cfg.OutputDir, name+".txt")
	if err := os.WriteFile(path, []byte(text+"\n"), 0o644); err != nil {
		return IngestedDocument{}, fmt.Errorf("write text %q: %w", path, err)
	}
	return IngestedDocument{
		Source:       parsed.String(),
		Path:         path,
		Title:        title,
		ContentType:  contentType,
		TextBytes:    len([]byte(text)),
		FetchedBytes: len(body),
	}, nil
}

func readLimited(r io.Reader, maxBytes int64) ([]byte, error) {
	limited := io.LimitReader(r, maxBytes+1)
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > maxBytes {
		return nil, fmt.Errorf("response exceeds max bytes %d", maxBytes)
	}
	return body, nil
}

func ExtractReadableText(body []byte, contentType string) (string, string, error) {
	mediaType, _, _ := mime.ParseMediaType(contentType)
	raw := string(body)
	switch {
	case mediaType == "" || mediaType == "text/html" || mediaType == "application/xhtml+xml":
		return htmlToText(raw), extractHTMLTitle(raw), nil
	case strings.HasPrefix(mediaType, "text/"):
		return NormalizePlainText(raw), "", nil
	default:
		return "", "", fmt.Errorf("unsupported content type %q", contentType)
	}
}

var (
	htmlDropRE       = regexp.MustCompile(`(?is)<(script|style|noscript|svg|template|head|nav|footer|form|iframe)\b[^>]*>.*?</(script|style|noscript|svg|template|head|nav|footer|form|iframe)>`)
	htmlBlockRE      = regexp.MustCompile(`(?i)</?(p|div|section|article|main|header|h[1-6]|li|ul|ol|tr|table|br|blockquote)\b[^>]*>`)
	htmlTagRE        = regexp.MustCompile(`(?s)<[^>]+>`)
	htmlTitleRE      = regexp.MustCompile(`(?is)<title\b[^>]*>(.*?)</title>`)
	nonFilenameChars = regexp.MustCompile(`[^a-zA-Z0-9._-]+`)
)

func htmlToText(raw string) string {
	cleaned := htmlDropRE.ReplaceAllString(raw, "\n")
	cleaned = htmlBlockRE.ReplaceAllString(cleaned, "\n")
	cleaned = htmlTagRE.ReplaceAllString(cleaned, " ")
	cleaned = html.UnescapeString(cleaned)
	return NormalizePlainText(cleaned)
}

func extractHTMLTitle(raw string) string {
	matches := htmlTitleRE.FindStringSubmatch(raw)
	if len(matches) < 2 {
		return ""
	}
	return NormalizePlainText(htmlTagRE.ReplaceAllString(html.UnescapeString(matches[1]), " "))
}

func NormalizePlainText(raw string) string {
	raw = strings.ReplaceAll(raw, "\r\n", "\n")
	raw = strings.ReplaceAll(raw, "\r", "\n")
	lines := strings.Split(raw, "\n")
	out := make([]string, 0, len(lines))
	blank := false
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line == "" {
			if !blank && len(out) > 0 {
				out = append(out, "")
			}
			blank = true
			continue
		}
		out = append(out, line)
		blank = false
	}
	for len(out) > 0 && out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

func filenameForURL(parsed *url.URL) string {
	base := parsed.Host + parsed.EscapedPath()
	if base == "" {
		base = "document"
	}
	base = strings.Trim(base, "/")
	base = strings.ReplaceAll(base, "/", "-")
	base = nonFilenameChars.ReplaceAllString(base, "-")
	base = strings.Trim(base, "-._")
	if base == "" {
		base = "document"
	}
	if len(base) > 80 {
		base = base[:80]
	}
	sum := sha1.Sum([]byte(parsed.String()))
	return strings.ToLower(base) + "-" + hex.EncodeToString(sum[:])[:10]
}

func writeIngestMetadata(outputDir string, results []IngestedDocument) error {
	path := filepath.Join(outputDir, "sources.jsonl")
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create metadata %q: %w", path, err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	for _, result := range results {
		if err := encoder.Encode(result); err != nil {
			return fmt.Errorf("write metadata %q: %w", path, err)
		}
	}
	return nil
}
