<h1 align="center">KVHarbor</h1>

<p align="center">
  <a href="#quick-start">Quick start</a> · <a href="#run-real-models">Run real models</a> · <a href="#architecture">Architecture</a> · <a href="#benchmarks">Benchmarks</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

KVHarbor runs text generation, speech recognition, and speech synthesis in one
C++20 server. Choose **llama.cpp** or **TensorRT Edge-LLM** for GPU text
generation. **Qwen3-ASR** and **Kokoro** run on the CPU, alongside text or on
their own.

## Quick start

This Ubuntu 24.04 example runs on Linux or WSL2 with a CPU test backend. You can try the streaming API before downloading model weights.

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 build-essential cmake libboost-dev nlohmann-json3-dev
git clone https://github.com/Vitolearns/KVHarbor.git
cd KVHarbor
cmake -S serving -B build-serving-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-serving-cpu --parallel 2
./build-serving-cpu/kvharbor-server --test-backend --port 18082
```

With the server running, send a request from another terminal.

```bash
curl -fsSN http://127.0.0.1:18082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":4}'
```

The stream sends four content chunks and ends with `data: [DONE]`. Stop the server with Ctrl+C.

## Run real models

<a name="cuda-text-generation"></a>
<a name="tensorrt-edge-llm-text-generation"></a>
<a name="enable-speech"></a>

| Workload | Engine | Setup |
| --- | --- | --- |
| Text generation from GGUF | llama.cpp on GPU | [Run Qwen3.8-27B](docs/serving.md#cuda-text-generation) |
| Text generation from a TensorRT engine | TensorRT Edge-LLM on GPU | [Run Qwen3.5-0.8B](docs/serving.md#tensorrt-edge-llm-text-generation) |
| Transcription and speech synthesis | sherpa-onnx on CPU | [Run Qwen3-ASR and Kokoro](docs/serving.md#enable-speech) |

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/architecture-dark.svg">
  <img src="assets/readme/architecture-light.svg" width="960" alt="HTTP ingress, separate task queues, runtime adapters, engines, and response formats">
</picture>

Text, recognition, and synthesis have separate queues and worker slots. A
disconnect cancels the request and wakes any worker blocked on output. For a
running request, the slot becomes available again after the native call has
finished. [Request lifecycle and engine notes](docs/serving.md#implementation-notes)

## Benchmarks

These runs used Qwen3.8-27B Q4_K_M on an RTX 5090 Laptop GPU.

- The [serving load test](docs/experiments/serving-acceptance.md) recorded 1,152
  arrivals across three rates and checked overload and recovery separately.
- [Reserving one logits row](docs/experiments/perf-003-gpu.md) cut the CUDA
  workspace from 507 to 124 MiB and added about 5 ms to median HTTP completion.

See also [prefill/decode interleaving](docs/experiments/prefill-decode.md) and
[greedy sampling](docs/experiments/perf-002-sampling.md).

## API

| Task | Endpoint | Payload and response |
| --- | --- | --- |
| Text | `POST /v1/chat/completions`, `POST /v1/completions` | JSON request with `stream: true` and `temperature: 0`. Successful responses use SSE. |
| Speech recognition | `POST /v1/audio/transcriptions` | Mono PCM16 WAV request with JSON transcript |
| Speech synthesis | `POST /v1/audio/speech` | JSON request with raw PCM16 response and sample-rate header |
| Service state | `GET /health`, `GET /state`, `GET /metrics`, `GET /v1/models` | Runtime health, task occupancy, Prometheus metrics and loaded models |

## Development

Use the quick-start build to check the service. For source entry points and
benchmark script checks, see [development notes](docs/serving.md#development).

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
```

## License

Contributions covered by this repository's license use
[PolyForm Noncommercial 1.0.0](LICENSE). See [NOTICE](NOTICE.md) for inherited
code, third-party dependencies and model terms.
