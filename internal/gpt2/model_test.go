package gpt2

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/augahmed/aurelius/internal/tokenizer"
)

func TestLoadSafeTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	state := tinyStateDict()
	if err := writeSafeTensors(path, state); err != nil {
		t.Fatalf("writeSafeTensors error: %v", err)
	}

	loaded, err := LoadSafeTensors(path)
	if err != nil {
		t.Fatalf("LoadSafeTensors error: %v", err)
	}

	if len(loaded) != len(state) {
		t.Fatalf("loaded tensor count = %d, want %d", len(loaded), len(state))
	}
	got := loaded["transformer.wte.weight"].Data[0]
	want := state["transformer.wte.weight"].Data[0]
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("first embedding value = %v, want %v", got, want)
	}
}

func TestModelForwardMatchesReference(t *testing.T) {
	cfg := tinyConfig()
	state := tinyStateDict()

	model, err := NewModel(cfg, state)
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}

	input := []int{7, 8}
	got, err := model.Forward(input, nil)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}

	want := referenceForward(cfg, state, input)
	if len(got) != len(want) {
		t.Fatalf("logit length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Fatalf("logit[%d] = %.8f, want %.8f", i, got[i], want[i])
		}
	}
}

func TestLoadModel(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.json")
	weightsPath := filepath.Join(dir, "model.safetensors")

	if err := os.WriteFile(configPath, []byte(`{"model_type":"gpt2","vocab_size":11,"n_ctx":8,"n_embd":2,"n_layer":1,"n_head":1,"n_inner":3}`), 0o644); err != nil {
		t.Fatalf("WriteFile config error: %v", err)
	}
	if err := writeSafeTensors(weightsPath, tinyStateDict()); err != nil {
		t.Fatalf("writeSafeTensors error: %v", err)
	}

	model, err := LoadModel(configPath, weightsPath)
	if err != nil {
		t.Fatalf("LoadModel error: %v", err)
	}

	logits, err := model.Forward([]int{7, 8}, nil)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if len(logits) != 11 {
		t.Fatalf("len(logits) = %d, want %d", len(logits), 11)
	}
}

func tinyConfig() Config {
	return Config{
		ModelType:        "gpt2",
		VocabSize:        11,
		ContextLength:    8,
		EmbeddingDim:     2,
		NumLayers:        1,
		NumHeads:         1,
		FeedForwardDim:   3,
		LayerNormEpsilon: 1e-5,
	}
}

func tinyStateDict() map[string]Tensor {
	return map[string]Tensor{
		"transformer.wte.weight": tensor2D(11, 2, []float64{
			0.10, 0.00,
			0.00, 0.10,
			0.20, -0.10,
			-0.10, 0.20,
			0.30, 0.40,
			-0.30, 0.50,
			0.40, -0.20,
			0.60, 0.10,
			-0.20, 0.70,
			0.50, -0.40,
			-0.50, -0.30,
		}),
		"transformer.wpe.weight": tensor2D(8, 2, []float64{
			0.01, -0.02,
			0.03, 0.04,
			0.05, -0.01,
			-0.04, 0.02,
			0.02, 0.03,
			-0.02, -0.03,
			0.04, 0.01,
			-0.01, 0.05,
		}),
		"transformer.h.0.ln_1.weight": tensor1D([]float64{1.10, 0.90}),
		"transformer.h.0.ln_1.bias":   tensor1D([]float64{0.01, -0.02}),
		"transformer.h.0.attn.c_attn.weight": tensor2D(2, 6, []float64{
			0.20, -0.10, 0.05, 0.30, -0.20, 0.10,
			0.10, 0.15, -0.05, -0.25, 0.20, 0.05,
		}),
		"transformer.h.0.attn.c_attn.bias": tensor1D([]float64{0.01, -0.02, 0.03, 0.02, -0.01, 0.04}),
		"transformer.h.0.attn.c_proj.weight": tensor2D(2, 2, []float64{
			0.20, -0.30,
			0.40, 0.10,
		}),
		"transformer.h.0.attn.c_proj.bias": tensor1D([]float64{0.01, -0.02}),
		"transformer.h.0.ln_2.weight":      tensor1D([]float64{0.95, 1.05}),
		"transformer.h.0.ln_2.bias":        tensor1D([]float64{-0.03, 0.02}),
		"transformer.h.0.mlp.c_fc.weight": tensor2D(2, 3, []float64{
			0.30, -0.20, 0.10,
			-0.10, 0.25, 0.20,
		}),
		"transformer.h.0.mlp.c_fc.bias": tensor1D([]float64{0.01, -0.02, 0.03}),
		"transformer.h.0.mlp.c_proj.weight": tensor2D(3, 2, []float64{
			0.20, -0.10,
			0.05, 0.30,
			-0.25, 0.15,
		}),
		"transformer.h.0.mlp.c_proj.bias": tensor1D([]float64{0.02, -0.01}),
		"transformer.ln_f.weight":         tensor1D([]float64{1.00, 0.85}),
		"transformer.ln_f.bias":           tensor1D([]float64{0.00, 0.03}),
	}
}

func tensor1D(data []float64) Tensor {
	return Tensor{Shape: []int{len(data)}, Data: append([]float64(nil), data...)}
}

func tensor2D(rows, cols int, data []float64) Tensor {
	return Tensor{Shape: []int{rows, cols}, Data: append([]float64(nil), data...)}
}

func testVocab() map[string]int {
	return map[string]int{
		"h":     0,
		"e":     1,
		"l":     2,
		"o":     3,
		"he":    4,
		"hel":   5,
		"hell":  6,
		"hello": 7,
		"!":     8,
		"Ã":     9,
		"©":     10,
	}
}

func testMerges() []tokenizer.MergeRule {
	return []tokenizer.MergeRule{
		{Left: "h", Right: "e"},
		{Left: "he", Right: "l"},
		{Left: "hel", Right: "l"},
		{Left: "hell", Right: "o"},
	}
}

func writeSafeTensors(path string, tensors map[string]Tensor) error {
	type headerEntry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}

	header := make(map[string]headerEntry, len(tensors))
	payload := make([]byte, 0)
	offset := 0
	for name, tensor := range tensors {
		start := offset
		for _, value := range tensor.Data {
			var bytes [4]byte
			binary.LittleEndian.PutUint32(bytes[:], math.Float32bits(float32(value)))
			payload = append(payload, bytes[:]...)
			offset += 4
		}
		header[name] = headerEntry{
			DType:       "F32",
			Shape:       append([]int(nil), tensor.Shape...),
			DataOffsets: []int{start, offset},
		}
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}

	file := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(file[:8], uint64(len(headerBytes)))
	copy(file[8:], headerBytes)
	copy(file[8+len(headerBytes):], payload)
	return os.WriteFile(path, file, 0o644)
}

func referenceForward(cfg Config, state map[string]Tensor, input []int) []float64 {
	model, err := NewModel(cfg, state)
	if err != nil {
		panic(err)
	}

	hidden, err := model.embed(input)
	if err != nil {
		panic(err)
	}
	block := model.blocks[0]
	norm1 := refApplyLayerNorm(hidden, len(input), cfg.EmbeddingDim, block.AttentionNorm, cfg.ResolvedLayerNormEpsilon())
	attn := refAttention(norm1, len(input), cfg.EmbeddingDim, block.Attention)
	refAdd(hidden, attn)
	norm2 := refApplyLayerNorm(hidden, len(input), cfg.EmbeddingDim, block.MLPNorm, cfg.ResolvedLayerNormEpsilon())
	mlp := refMLP(norm2, len(input), cfg.EmbeddingDim, block.MLP)
	refAdd(hidden, mlp)

	last := hidden[(len(input)-1)*cfg.EmbeddingDim : len(input)*cfg.EmbeddingDim]
	last = refApplyLayerNormRow(last, model.finalNorm, cfg.ResolvedLayerNormEpsilon())

	logits := make([]float64, cfg.VocabSize)
	for token := 0; token < cfg.VocabSize; token++ {
		sum := 0.0
		for dim := 0; dim < cfg.EmbeddingDim; dim++ {
			sum += last[dim] * model.lmHead.Data[token*cfg.EmbeddingDim+dim]
		}
		logits[token] = sum
	}
	return logits
}

func refApplyLayerNorm(hidden []float64, rows, cols int, norm LayerNorm, epsilon float64) []float64 {
	out := make([]float64, len(hidden))
	for row := 0; row < rows; row++ {
		copy(out[row*cols:(row+1)*cols], refApplyLayerNormRow(hidden[row*cols:(row+1)*cols], norm, epsilon))
	}
	return out
}

func refApplyLayerNormRow(row []float64, norm LayerNorm, epsilon float64) []float64 {
	mean := 0.0
	for _, value := range row {
		mean += value
	}
	mean /= float64(len(row))

	variance := 0.0
	for _, value := range row {
		diff := value - mean
		variance += diff * diff
	}
	variance /= float64(len(row))

	denom := math.Sqrt(variance + epsilon)
	out := make([]float64, len(row))
	for i, value := range row {
		out[i] = ((value - mean) / denom * norm.Weight[i]) + norm.Bias[i]
	}
	return out
}

func refAttention(hidden []float64, seqLen, embDim int, attention Attention) []float64 {
	qkv := refAffine(hidden, seqLen, embDim, attention.CombinedWeight, attention.CombinedBias)
	queries := make([]float64, seqLen*embDim)
	keys := make([]float64, seqLen*embDim)
	values := make([]float64, seqLen*embDim)
	for row := 0; row < seqLen; row++ {
		offset := row * embDim * 3
		copy(queries[row*embDim:(row+1)*embDim], qkv[offset:offset+embDim])
		copy(keys[row*embDim:(row+1)*embDim], qkv[offset+embDim:offset+2*embDim])
		copy(values[row*embDim:(row+1)*embDim], qkv[offset+2*embDim:offset+3*embDim])
	}

	headDim := embDim / attention.NumHeads
	context := make([]float64, seqLen*embDim)
	scale := math.Sqrt(float64(headDim))
	for row := 0; row < seqLen; row++ {
		for head := 0; head < attention.NumHeads; head++ {
			headOffset := head * headDim
			scores := make([]float64, row+1)
			maxScore := math.Inf(-1)
			for keyRow := 0; keyRow <= row; keyRow++ {
				dot := 0.0
				for dim := 0; dim < headDim; dim++ {
					dot += queries[row*embDim+headOffset+dim] * keys[keyRow*embDim+headOffset+dim]
				}
				scores[keyRow] = dot / scale
				if scores[keyRow] > maxScore {
					maxScore = scores[keyRow]
				}
			}

			total := 0.0
			for i, score := range scores {
				scores[i] = math.Exp(score - maxScore)
				total += scores[i]
			}

			for keyRow, score := range scores {
				weight := score / total
				for dim := 0; dim < headDim; dim++ {
					context[row*embDim+headOffset+dim] += weight * values[keyRow*embDim+headOffset+dim]
				}
			}
		}
	}

	return refAffine(context, seqLen, embDim, attention.ProjectWeight, attention.ProjectBias)
}

func refMLP(hidden []float64, seqLen, embDim int, mlp MLP) []float64 {
	up := refAffine(hidden, seqLen, embDim, mlp.UpWeight, mlp.UpBias)
	for i, value := range up {
		up[i] = gelu(value)
	}
	return refAffine(up, seqLen, len(mlp.UpBias), mlp.DownWeight, mlp.DownBias)
}

func refAffine(hidden []float64, rows, inDim int, weight Tensor, bias []float64) []float64 {
	outDim := len(bias)
	out := make([]float64, rows*outDim)
	for row := 0; row < rows; row++ {
		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := bias[outIndex]
			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += hidden[row*inDim+inIndex] * weight.Data[inIndex*outDim+outIndex]
			}
			out[row*outDim+outIndex] = sum
		}
	}
	return out
}

func refAdd(dst, src []float64) {
	for i, value := range src {
		dst[i] += value
	}
}
