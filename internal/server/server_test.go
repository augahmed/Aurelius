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
	if !strings.Contains(rec.Body.String(), "Aurelius Chat") || !strings.Contains(rec.Body.String(), `id="chat-form"`) {
		t.Fatalf("body = %q, want Aurelius chat UI", rec.Body.String())
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

func TestServerGenerateWithMessagesUsesAssistantPreamble(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply"}
	srv := New(engine, WithGeneratePolicy(GeneratePolicy{
		AssistantPreamble: "You are a helpful assistant. Answer directly and completely.",
	}))

	body := `{"max_tokens":4,"messages":[{"role":"user","content":"what is 2 + 2?"}]}`
	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	wantPrompt := "You are a helpful assistant. Answer directly and completely.\n\nUser: what is 2 + 2?\n\nAssistant:"
	if engine.prompt != wantPrompt {
		t.Fatalf("prompt = %q, want %q", engine.prompt, wantPrompt)
	}
}

func TestServerGenerateAppliesPolicyToOptions(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply"}
	srv := New(engine, WithGeneratePolicy(GeneratePolicy{
		DefaultMaxTokens:   8,
		MaxTokensCap:       12,
		DefaultTemperature: 0.8,
		MinTemperature:     0.2,
		MaxTemperature:     1.2,
		DefaultTopK:        20,
		MaxTopK:            40,
		DefaultStopStrings: []string{"\nUser:"},
		MaxStopStrings:     3,
		MaxStopRunes:       8,
	}))

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"hello","max_tokens":32,"temperature":9,"top_k":100,"use_cache":true,"stop":["END","END","oversized-stop"]}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	if engine.options.MaxTokens != 12 {
		t.Fatalf("options.MaxTokens = %d, want %d", engine.options.MaxTokens, 12)
	}
	if engine.options.Temperature != 1.2 {
		t.Fatalf("options.Temperature = %f, want %f", engine.options.Temperature, 1.2)
	}
	if engine.options.TopK != 40 {
		t.Fatalf("options.TopK = %d, want %d", engine.options.TopK, 40)
	}
	if !engine.options.UseCache {
		t.Fatal("expected use_cache to remain enabled")
	}
	if want := []string{"END", "oversize", "\nUser:"}; !equalStrings(engine.options.StopStrings, want) {
		t.Fatalf("options.StopStrings = %q, want %q", engine.options.StopStrings, want)
	}
}

func TestServerGenerateUsesDefaultTemperatureWhenUnset(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply"}
	srv := New(engine, WithGeneratePolicy(GeneratePolicy{
		DefaultTemperature: 0.8,
		DefaultTopK:        20,
	}))

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"hello","max_tokens":4}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	if engine.options.Temperature != 0.8 {
		t.Fatalf("options.Temperature = %f, want %f", engine.options.Temperature, 0.8)
	}
	if engine.options.TopK != 20 {
		t.Fatalf("options.TopK = %d, want %d", engine.options.TopK, 20)
	}
}

func TestServerGenerateTrimsConversationByPolicy(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply"}
	srv := New(engine, WithGeneratePolicy(GeneratePolicy{
		MaxMessages:     2,
		MaxMessageRunes: 3,
	}))

	body := `{"max_tokens":2,"messages":[{"role":"user","content":"alpha"},{"role":"assistant","content":"bravo"},{"role":"user","content":"charlie"}]}`
	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	wantPrompt := "Assistant: avo\n\nUser: lie\n\nAssistant:"
	if engine.prompt != wantPrompt {
		t.Fatalf("prompt = %q, want %q", engine.prompt, wantPrompt)
	}
}

func TestServerGenerateTrimsDirectPromptByPolicy(t *testing.T) {
	engine := &fakeGenerator{outputSuffix: "reply"}
	srv := New(engine, WithGeneratePolicy(GeneratePolicy{
		MaxPromptRunes: 5,
	}))

	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"0123456789","max_tokens":1}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d, body = %q", rec.Code, http.StatusOK, rec.Body.String())
	}
	if engine.prompt != "56789" {
		t.Fatalf("prompt = %q, want %q", engine.prompt, "56789")
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

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
