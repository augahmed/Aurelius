package textdata

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestExtractReadableTextFromHTML(t *testing.T) {
	text, title, err := ExtractReadableText([]byte(`<!doctype html>
<html>
<head><title> Algebra Basics </title><style>.hidden{display:none}</style></head>
<body>
<nav>skip nav</nav>
<h1>Derivative Rule</h1>
<p>The derivative of x^2 is 2x.</p>
<script>skip()</script>
</body>
</html>`), "text/html; charset=utf-8")
	if err != nil {
		t.Fatalf("ExtractReadableText error: %v", err)
	}
	if title != "Algebra Basics" {
		t.Fatalf("title = %q, want Algebra Basics", title)
	}
	if strings.Contains(text, "skip") || !strings.Contains(text, "Derivative Rule") || !strings.Contains(text, "2x") {
		t.Fatalf("text = %q, want cleaned readable text", text)
	}
}

func TestIngestWebTextWritesTextAndMetadata(t *testing.T) {
	srv := newLocalHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("User-Agent"); got != "test-agent" {
			t.Fatalf("User-Agent = %q, want test-agent", got)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(`<html><head><title>Math</title></head><body><p>x^2 differentiates to 2x.</p></body></html>`))
	}))
	defer srv.Close()

	outputDir := t.TempDir()
	results, err := IngestWebText(context.Background(), []string{srv.URL + "/lesson"}, WebIngestConfig{
		OutputDir: outputDir,
		MaxBytes:  1024,
		Timeout:   2 * time.Second,
		UserAgent: "test-agent",
	})
	if err != nil {
		t.Fatalf("IngestWebText error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	raw, err := os.ReadFile(results[0].Path)
	if err != nil {
		t.Fatalf("ReadFile ingested text error: %v", err)
	}
	if !strings.Contains(string(raw), "x^2 differentiates to 2x.") {
		t.Fatalf("ingested text = %q", string(raw))
	}

	metaRaw, err := os.ReadFile(filepath.Join(outputDir, "sources.jsonl"))
	if err != nil {
		t.Fatalf("ReadFile metadata error: %v", err)
	}
	var meta IngestedDocument
	if err := json.Unmarshal(bytesTrimSpace(metaRaw), &meta); err != nil {
		t.Fatalf("Unmarshal metadata error: %v", err)
	}
	if meta.Title != "Math" || meta.Source == "" || meta.Path == "" {
		t.Fatalf("metadata = %+v, want source/path/title", meta)
	}
}

func TestIngestWebTextRejectsLargeResponse(t *testing.T) {
	srv := newLocalHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("0123456789"))
	}))
	defer srv.Close()

	_, err := IngestWebText(context.Background(), []string{srv.URL}, WebIngestConfig{
		OutputDir: t.TempDir(),
		MaxBytes:  4,
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds max bytes") {
		t.Fatalf("error = %v, want max bytes error", err)
	}
}

func TestLoadURLList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "urls.txt")
	if err := os.WriteFile(path, []byte("# comment\n\nhttps://example.com/a\nhttps://example.com/b\n"), 0o644); err != nil {
		t.Fatalf("WriteFile error: %v", err)
	}
	urls, err := LoadURLList(path)
	if err != nil {
		t.Fatalf("LoadURLList error: %v", err)
	}
	if len(urls) != 2 || urls[0] != "https://example.com/a" || urls[1] != "https://example.com/b" {
		t.Fatalf("urls = %v, want parsed URL lines", urls)
	}
}

func bytesTrimSpace(value []byte) []byte {
	return []byte(strings.TrimSpace(string(value)))
}

func newLocalHTTPServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("local listener unavailable: %v", err)
	}
	srv := httptest.NewUnstartedServer(handler)
	srv.Listener = listener
	srv.Start()
	return srv
}
