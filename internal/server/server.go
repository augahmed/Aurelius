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
	policy   GeneratePolicy
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type GenerateRequest struct {
	Prompt      string        `json:"prompt"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	TopK        int           `json:"top_k"`
	UseCache    bool          `json:"use_cache"`
	Stop        []string      `json:"stop"`
	Messages    []ChatMessage `json:"messages"`
}

type GenerateResponse struct {
	Output string `json:"output"`
}

type GeneratePolicy struct {
	DefaultMaxTokens   int
	MaxTokensCap       int
	DefaultTemperature float64
	MinTemperature     float64
	MaxTemperature     float64
	DefaultTopK        int
	MaxTopK            int
	MaxMessages        int
	MaxMessageRunes    int
	MaxPromptRunes     int
	AssistantPreamble  string
	DisableCache       bool
	DefaultStopStrings []string
	MaxStopStrings     int
	MaxStopRunes       int
}

type Option func(*Server)

func WithGeneratePolicy(policy GeneratePolicy) Option {
	return func(s *Server) {
		s.policy = policy
	}
}

func New(engine Generator, options ...Option) *Server {
	assets, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err)
	}
	s := &Server{
		engine:   engine,
		mux:      http.NewServeMux(),
		assetsFS: assets,
	}
	for _, option := range options {
		if option != nil {
			option(s)
		}
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

	req = applyGeneratePolicy(req, s.policy)
	prompt, err := buildPrompt(req, s.policy)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	options := s.resolveGenerateOptions(req)
	output, err := s.engine.GenerateWithOptions(prompt, runtime.GenerateOptions{
		MaxTokens:   options.MaxTokens,
		TopK:        options.TopK,
		UseCache:    options.UseCache,
		StopStrings: options.StopStrings,
		Temperature: options.Temperature,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	writeJSON(w, http.StatusOK, GenerateResponse{
		Output: textutil.SanitizeVisibleOrFallback(extractCompletion(prompt, output)),
	})
}

func buildPrompt(req GenerateRequest, policy GeneratePolicy) (string, error) {
	preamble := strings.TrimSpace(policy.AssistantPreamble)
	if len(req.Messages) == 0 {
		prompt := strings.TrimSpace(req.Prompt)
		if prompt == "" {
			return "", fmt.Errorf("prompt cannot be empty")
		}
		if preamble == "" {
			return prompt, nil
		}
		return preamble + "\n\nUser: " + prompt + "\n\nAssistant:", nil
	}

	var builder strings.Builder
	if preamble != "" {
		builder.WriteString(preamble)
		builder.WriteString("\n\n")
	}
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
	prompt := builder.String() + "Assistant:"
	return prompt, nil
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

func (s *Server) resolveGenerateOptions(req GenerateRequest) runtime.GenerateOptions {
	maxTokens := req.MaxTokens
	if s.policy.DefaultMaxTokens > 0 && maxTokens <= 0 {
		maxTokens = s.policy.DefaultMaxTokens
	}
	if s.policy.MaxTokensCap > 0 && maxTokens > s.policy.MaxTokensCap {
		maxTokens = s.policy.MaxTokensCap
	}

	useCache := req.UseCache
	if s.policy.DisableCache {
		useCache = false
	}

	return runtime.GenerateOptions{
		MaxTokens:   maxTokens,
		TopK:        resolveTopK(req.TopK, s.policy),
		UseCache:    useCache,
		StopStrings: resolveStopStrings(req.Stop, s.policy),
		Temperature: resolveTemperature(req.Temperature, s.policy),
	}
}

func applyGeneratePolicy(req GenerateRequest, policy GeneratePolicy) GenerateRequest {
	if len(req.Messages) == 0 {
		req.Prompt = trimTrailingRunes(req.Prompt, policy.MaxPromptRunes)
		return req
	}

	if policy.MaxMessages > 0 && len(req.Messages) > policy.MaxMessages {
		req.Messages = append([]ChatMessage(nil), req.Messages[len(req.Messages)-policy.MaxMessages:]...)
	}
	if policy.MaxMessageRunes <= 0 {
		return req
	}

	trimmed := make([]ChatMessage, len(req.Messages))
	for i, message := range req.Messages {
		trimmed[i] = ChatMessage{
			Role:    message.Role,
			Content: trimTrailingRunes(message.Content, policy.MaxMessageRunes),
		}
	}
	req.Messages = trimmed
	return req
}

func trimTrailingRunes(value string, maxRunes int) string {
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[len(runes)-maxRunes:])
}

func resolveTemperature(requested float64, policy GeneratePolicy) float64 {
	temperature := requested
	if temperature == 0 && policy.DefaultTemperature > 0 {
		temperature = policy.DefaultTemperature
	}
	if temperature <= 0 {
		return 0
	}
	if policy.MinTemperature > 0 && temperature < policy.MinTemperature {
		temperature = policy.MinTemperature
	}
	if policy.MaxTemperature > 0 && temperature > policy.MaxTemperature {
		temperature = policy.MaxTemperature
	}
	return temperature
}

func resolveTopK(requested int, policy GeneratePolicy) int {
	topK := requested
	if topK <= 0 && policy.DefaultTopK > 0 {
		topK = policy.DefaultTopK
	}
	if topK <= 0 {
		return 0
	}
	if policy.MaxTopK > 0 && topK > policy.MaxTopK {
		topK = policy.MaxTopK
	}
	return topK
}

func resolveStopStrings(requested []string, policy GeneratePolicy) []string {
	values := append([]string(nil), requested...)
	values = append(values, policy.DefaultStopStrings...)
	limit := len(values)
	if policy.MaxStopStrings > 0 && limit > policy.MaxStopStrings {
		limit = policy.MaxStopStrings
	}
	out := make([]string, 0, limit)
	for _, value := range values {
		value = strings.Trim(value, "\r")
		if value == "" {
			continue
		}
		if policy.MaxStopRunes > 0 {
			value = trimLeadingRunes(value, policy.MaxStopRunes)
		}
		seen := false
		for _, existing := range out {
			if existing == value {
				seen = true
				break
			}
		}
		if !seen {
			out = append(out, value)
		}
		if len(out) >= limit {
			break
		}
	}
	return out
}

func trimLeadingRunes(value string, maxRunes int) string {
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}
