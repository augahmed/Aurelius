package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/textutil"
)

func TestServerHealth(t *testing.T) {
	srv := New(nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("body = %q, want health payload", rec.Body.String())
	}
}

func TestServerIndex(t *testing.T) {
	srv := New(nil)

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if !strings.Contains(rec.Body.String(), "Aurelius Console") {
		t.Fatalf("body = %q, want Aurelius UI", rec.Body.String())
	}
}

func TestServerGenerate(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "response"}
	srv := New(engine)

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"hello","max_tokens":3,"use_cache":true}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	if engine.prompt != "hello" {
		t.Fatalf("prompt = %q, want %q", engine.prompt, "hello")
	}
	if engine.options.MaxTokens != 3 || !engine.options.UseCache {
		t.Fatalf("options = %+v, want max_tokens=3 use_cache=true", engine.options)
	}

	var res GenerateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if res.Output != "response" {
		t.Fatalf("output = %q, want %q", res.Output, "response")
	}
}

func TestServerGenerateWithMessagesBuildsConversationPrompt(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply"}
	srv := New(engine)

	body := `{"max_tokens":2,"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"how are you?"}]}`
	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	wantPrompt := "User: hello\n\nAssistant: hi\n\nUser: how are you?\n\nAssistant:"
	if engine.prompt != wantPrompt {
		t.Fatalf("prompt = %q, want %q", engine.prompt, wantPrompt)
	}
}

func TestServerGenerateSanitizesNonDisplayableCompletion(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply\x02\x00"}
	srv := New(engine)

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"hello","max_tokens":3}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}

	var res GenerateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if res.Output != "reply" {
		t.Fatalf("output = %q, want %q", res.Output, "reply")
	}
}

func TestServerGenerateUsesPrototypeFallbackWhenCompletionIsNotReadable(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "\x02\x03\x00"}
	srv := New(engine)

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"hello","max_tokens":3}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}

	var res GenerateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if res.Output != textutil.PrototypeFallback {
		t.Fatalf("output = %q, want %q", res.Output, textutil.PrototypeFallback)
	}
}

func TestServerGenerateRejectsInvalidRequestBody(t *testing.T) {
	srv := New(&fakeGenerator{})

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestServerGenerateRejectsEmptyPrompt(t *testing.T) {
	srv := New(&fakeGenerator{})

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"   ","max_tokens":2}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestServerGenerateRejectsMethod(t *testing.T) {
	srv := New(&fakeGenerator{})

	req := httptest.NewRequest(http.MethodGet, "/generate", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

type fakeGenerator struct {
	prompt       string
	options      runtime.GenerateOptions
	outputSuffix string
	err          error
}

func (f *fakeGenerator) GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error) {
	f.prompt = prompt
	f.options = options
	if f.err != nil {
		return "", f.err
	}
	return prompt + f.outputSuffix, nil
}
