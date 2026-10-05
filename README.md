# Aurelius

Aurelius is a Go toolkit for training, evaluating, and serving small math-focused language models. It includes a CPU inference runtime, synthetic math datasets, specialist arithmetic and derivative checkpoints, and a local browser chat interface.

The project is a research prototype built for inspectable, reproducible experiments. The math backend combines model inference with optional deterministic answer correction. Uncorrected checkpoint output remains experimental.

## Requirements

- Go 1.25 or later
- Git
- A modern web browser
- `curl` for the command-line download instructions, or a browser to download release assets manually

The website is served by Go with embedded HTML, CSS, and JavaScript. No Node.js installation, frontend build, or GPU is required.

## Run the Website

### 1. Get the source

```sh
git clone https://github.com/augahmed/Aurelius.git
cd Aurelius
```

Use the current repository source for the generation controls described below. All subsequent commands run from the repository root.

### 2. Download the released checkpoints

Open the [math-router-v1 release](https://github.com/augahmed/Aurelius/releases/tag/math-router-v1) and download **both JSON files** under **Assets** into an `artifacts` directory at the repository root:

| Asset | Purpose |
| --- | --- |
| [math-router-arithmetic-v4b.json](https://github.com/augahmed/Aurelius/releases/download/math-router-v1/math-router-arithmetic-v4b.json) | Arithmetic specialist checkpoint |
| [math-router-derivative-full-v2.json](https://github.com/augahmed/Aurelius/releases/download/math-router-v1/math-router-derivative-full-v2.json) | Polynomial derivative specialist checkpoint |

Alternatively, download them from the terminal:

```sh
mkdir -p artifacts

curl -fL --retry 3 \
  -o artifacts/math-router-arithmetic-v4b.json \
  https://github.com/augahmed/Aurelius/releases/download/math-router-v1/math-router-arithmetic-v4b.json

curl -fL --retry 3 \
  -o artifacts/math-router-derivative-full-v2.json \
  https://github.com/augahmed/Aurelius/releases/download/math-router-v1/math-router-derivative-full-v2.json
```

The resulting files should be:

```text
Aurelius/
  artifacts/
    math-router-arithmetic-v4b.json
    math-router-derivative-full-v2.json
```

Checkpoint files are distributed separately from the source. Cloning the repository or downloading GitHub's source archive does not include them. The release notes provide SHA-256 checksums for verification. Browse [all releases](https://github.com/augahmed/Aurelius/releases) for other published versions.

### 3. Start the server

```sh
go run ./cmd/aurelius serve \
  -backend math-router \
  -checkpoint ./artifacts/math-router-arithmetic-v4b.json \
  -derivative-checkpoint ./artifacts/math-router-derivative-full-v2.json
```

### 4. Open the website

Open [http://localhost:8080](http://localhost:8080) in your browser. Keep the terminal running while using the website. Press **Ctrl+C** in the terminal to stop the server.

Example prompts:

- `What is 7 times 8?`
- `What is 12 + 7?`
- `What is the derivative of 4x^2 + 9x + 8?`

To use a different port, add `-addr localhost:8081` to the server command and open `http://localhost:8081`.

## Generation Controls

Expand **Generation controls** below the prompt field to adjust the next request:

| Control | Behavior |
| --- | --- |
| Max tokens | Limits generated tokens. Math checkpoints use byte tokenization; longer expressions need more tokens. |
| Temperature | Controls sampling randomness when sampling is enabled. Use `0` for greedy math decoding. |
| Top K | Limits candidate tokens. The math-router web backend caps this at `1` for greedy decoding. |
| KV cache | Requests cached decoding when the model supports it. |
| Deterministic math router | Enables or disables deterministic answer correction for the math-router backend. Enabled by default. |

With **Deterministic math router checked**, Aurelius asks the selected model first and replaces incorrect answers with a deterministic result for expressions the solver supports. It also uses that result if model generation fails on a supported expression.

With **the checkbox unchecked**, Aurelius returns the model's answer without deterministic correction or solver fallback. Prompt normalization and specialist checkpoint selection still occur. For example, `What is 7 times 8?` becomes `7 * 8 = ` before reaching the arithmetic model. Both `derivative` and `derrivative` are recognized; derivative prompts use the training prefix `Derrivative: `.

The checkbox applies to the `math-router` backend. Chat history and generation settings are saved in the browser's `localStorage`. For longer derivative answers, increase Max tokens to `32` or `64`; the UI initially uses `8`.

## Model Scope and Limitations

The router recognizes integer addition, subtraction, multiplication, and derivative questions. Its deterministic derivative solver handles supported polynomials. This backend is intended for supported math tasks rather than unrestricted conversation.

Corrected website answers and raw model accuracy are different measurements. The released derivative checkpoint can produce incorrect or malformed answers without correction, including for standalone powers such as `x^2` and `x^3`.

The current derivative data generator creates degree 1–3 polynomials with positive coefficients for every term, including a constant. Missing terms, zero or negative coefficients, and degree 4 or higher require broader training coverage. Normalizing the wording does not guarantee a correct model answer.

See [model evaluation](docs/model-evaluation.md) for raw checkpoint evaluation, error analysis, and targeted replay training.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `read checkpoint: ... no such file or directory` | Download both release assets and confirm their filenames and locations. Relative paths are resolved from the terminal's current directory. |
| The website uses the toy backend | Start the server with `-backend math-router` and both checkpoint flags shown above. |
| The math router checkbox is missing after updating | Restart the Go server and refresh the browser. The UI is embedded in the server binary. |
| The model answer is incomplete | Increase Max tokens to `32` or `64`. |
| Uncorrected answers are incorrect or malformed | Keep correction enabled for supported tasks, or evaluate and improve the checkpoint's training coverage. |
| `could not recognize a supported math question` | Use an explicit arithmetic expression or a derivative question like the examples above. |
| Port 8080 is already in use | Add `-addr localhost:8081` and open the corresponding URL. |

## Other Backends

`serve` accepts `-backend auto|toy|gpt2|mathlm|math-router`.

- **math-router:** Loads both specialist checkpoints and normalizes supported math questions.
- **mathlm:** Loads a single Aurelius JSON checkpoint.
- **gpt2:** Loads local GPT-2 configuration, tokenizer, and safetensors assets.
- **toy:** Exercises the inference and web interface without trained checkpoints.
- **auto:** Selects math-router when both checkpoints are supplied, mathlm when one is supplied, GPT-2 when complete assets exist under `artifacts/gpt2/`, and otherwise toy.

To try the interface without downloading checkpoints:

```sh
go run ./cmd/aurelius serve -backend toy
```

To serve a single trained checkpoint:

```sh
go run ./cmd/aurelius serve \
  -backend mathlm \
  -checkpoint ./artifacts/your-checkpoint.json
```

## Training and Evaluation

Aurelius supports small autoregressive MLP and transformer models, synthetic math curricula, JSON checkpoint save/resume, and exact-match evaluation grouped by operation, curriculum level, and prompt template.

Run a bounded training experiment:

```sh
mkdir -p artifacts

go run ./cmd/aurelius gen-math-data \
  -output-dir ./data/arithmetic-smoke \
  -operations add,sub,mul \
  -levels 1,2,4 \
  -train-count 2000 \
  -val-count 300

go run ./cmd/aurelius train-math \
  -model transformer \
  -data-dir ./data/arithmetic-smoke \
  -checkpoint ./artifacts/math-transformer-smoke.json \
  -context-size 32 \
  -embedding-dim 64 \
  -hidden-dim 256 \
  -num-heads 4 \
  -num-layers 2 \
  -max-steps 1000 \
  -log-every 100 \
  -grad-clip 1

go run ./cmd/aurelius eval-math \
  -checkpoint ./artifacts/math-transformer-smoke.json \
  -data ./data/arithmetic-smoke/val.jsonl \
  -max-tokens 24
```

Additional guides:

- [Training and inference](docs/llm-training.md)
- [Model evaluation and regression checks](docs/model-evaluation.md)
- [Architecture](docs/architecture.md)

## Publishing Checkpoints

Training checkpoints include Adam optimizer state for resuming training. Export inference assets before attaching them to a GitHub Release:

```sh
go run ./cmd/aurelius export-checkpoint \
  -checkpoint ./artifacts/your-training-checkpoint.json \
  -output ./release-checkpoints/your-inference-checkpoint.json
```

Export removes optimizer state by default. Run `go test ./...` and review exported files for private data before publishing. Upload only the selected inference files; `artifacts/` may contain intermediate checkpoints, datasets, and local experiment outputs. Generated data and checkpoint directories are excluded from Git.

## Development

```sh
gofmt -w ./cmd ./internal
go test ./...
```

Core packages:

| Package | Responsibility |
| --- | --- |
| `internal/arithmetic` | Synthetic math curricula and training examples |
| `internal/mathlm` | Trainable MLP and transformer models, checkpoints, and evaluation |
| `internal/mathrouter` | Prompt normalization, specialist selection, and optional answer correction |
| `internal/runtime` | Autoregressive generation, sampling, stopping, and cache support |
| `internal/server` | Embedded website and JSON generation API |
| `internal/tokenizer` | Byte tokenization and GPT-2 BPE |
| `internal/gpt2` | GPT-2 asset loading and inference |
| `internal/tensor`, `internal/transformer`, `internal/model`, `internal/sampler` | Tensor operations, prototype transformer, model contracts, and token selection |
| `internal/textdata` | Text ingestion, preparation, and instruction datasets |

## License

Aurelius is released under the [MIT License](LICENSE).
