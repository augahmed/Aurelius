package server

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"strings"

	"github.com/augahmed/aurelius/internal/runtime"
	"github.com/augahmed/aurelius/internal/textutil"
)

//go:embed ui/*
var uiFiles embed.FS

type Generator interface {
	GenerateWithOptions(prompt string, options runtime.GenerateOptions) (string, error)
}

type Server struct {
	engine   Generator
	mux      *http.ServeMux
	assetsFS fs.FS
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type GenerateRequest struct {
	Prompt    string        `json:"prompt"`
	MaxTokens int           `json:"max_tokens"`
	UseCache  bool          `json:"use_cache"`
	Messages  []ChatMessage `json:"messages"`
}

type GenerateResponse struct {
	Output string `json:"output"`
}

func New(engine Generator) *Server {
	assets, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err)
	}
	s := &Server{
		engine:   engine,
		mux:      http.NewServeMux(),
		assetsFS: assets,
	}
	s.routes()
	return s
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) routes() {
	s.mux.HandleFunc("/", s.handleIndex)
	s.mux.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.FS(s.assetsFS))))
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/generate", s.handleGenerate)
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	http.ServeFileFS(w, r, s.assetsFS, "index.html")
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.engine == nil {
		http.Error(w, "generation engine not configured", http.StatusNotImplemented)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	prompt, err := buildPrompt(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	output, err := s.engine.GenerateWithOptions(prompt, runtime.GenerateOptions{
		MaxTokens: req.MaxTokens,
		UseCache:  req.UseCache,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	writeJSON(w, http.StatusOK, GenerateResponse{
		Output: textutil.SanitizeVisibleOrFallback(extractCompletion(prompt, output)),
	})
}

func buildPrompt(req GenerateRequest) (string, error) {
	if len(req.Messages) == 0 {
		prompt := strings.TrimSpace(req.Prompt)
		if prompt == "" {
			return "", fmt.Errorf("prompt cannot be empty")
		}
		return prompt, nil
	}

	var builder strings.Builder
	for _, message := range req.Messages {
		content := strings.TrimSpace(message.Content)
		if content == "" {
			continue
		}
		builder.WriteString(normalizeRole(message.Role))
		builder.WriteString(": ")
		builder.WriteString(content)
		builder.WriteString("\n\n")
	}
	if builder.Len() == 0 {
		return "", fmt.Errorf("prompt cannot be empty")
	}
	builder.WriteString("Assistant:")
	return builder.String(), nil
}

func normalizeRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "assistant":
		return "Assistant"
	default:
		return "User"
	}
}

func extractCompletion(prompt, output string) string {
	if strings.HasPrefix(output, prompt) {
		return strings.TrimLeft(output[len(prompt):], "\n ")
	}
	return output
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
