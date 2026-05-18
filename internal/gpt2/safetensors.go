package gpt2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type Tensor struct {
	Shape []int
	Data  []float64
}

type safeTensorHeader struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

func LoadSafeTensors(path string) (map[string]Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read safetensors %q: %w", path, err)
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("invalid safetensors %q: missing header length", path)
	}

	headerSize := int(binary.LittleEndian.Uint64(data[:8]))
	if headerSize < 0 || 8+headerSize > len(data) {
		return nil, fmt.Errorf("invalid safetensors %q: header exceeds file size", path)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+headerSize], &raw); err != nil {
		return nil, fmt.Errorf("parse safetensors header %q: %w", path, err)
	}

	payload := data[8+headerSize:]
	out := make(map[string]Tensor, len(raw))
	for name, rawValue := range raw {
		if name == "__metadata__" {
			continue
		}

		var header safeTensorHeader
		if err := json.Unmarshal(rawValue, &header); err != nil {
			return nil, fmt.Errorf("parse tensor header %q in %q: %w", name, path, err)
		}
		tensor, err := tensorFromPayload(name, header, payload)
		if err != nil {
			return nil, err
		}
		out[name] = tensor
	}

	return out, nil
}

func tensorFromPayload(name string, header safeTensorHeader, payload []byte) (Tensor, error) {
	if len(header.Shape) == 0 {
		return Tensor{}, fmt.Errorf("tensor %q has empty shape", name)
	}
	if len(header.DataOffsets) != 2 {
		return Tensor{}, fmt.Errorf("tensor %q must define exactly two data offsets", name)
	}

	elementCount := 1
	for _, dim := range header.Shape {
		if dim <= 0 {
			return Tensor{}, fmt.Errorf("tensor %q has invalid dimension %d", name, dim)
		}
		elementCount *= dim
	}

	start, end := header.DataOffsets[0], header.DataOffsets[1]
	if start < 0 || end < start || end > len(payload) {
		return Tensor{}, fmt.Errorf("tensor %q has invalid data offsets [%d %d]", name, start, end)
	}

	var elementSize int
	switch header.DType {
	case "F32":
		elementSize = 4
	case "F64":
		elementSize = 8
	default:
		return Tensor{}, fmt.Errorf("tensor %q has unsupported dtype %q", name, header.DType)
	}

	byteCount := end - start
	if byteCount != elementCount*elementSize {
		return Tensor{}, fmt.Errorf("tensor %q byte size %d does not match shape %v for dtype %s", name, byteCount, header.Shape, header.DType)
	}

	values := make([]float64, elementCount)
	slice := payload[start:end]
	switch header.DType {
	case "F32":
		for i := 0; i < elementCount; i++ {
			bits := binary.LittleEndian.Uint32(slice[i*4 : (i+1)*4])
			values[i] = float64(math.Float32frombits(bits))
		}
	case "F64":
		for i := 0; i < elementCount; i++ {
			bits := binary.LittleEndian.Uint64(slice[i*8 : (i+1)*8])
			values[i] = math.Float64frombits(bits)
		}
	}

	return Tensor{
		Shape: append([]int(nil), header.Shape...),
		Data:  values,
	}, nil
}
