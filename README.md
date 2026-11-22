<h1 align="center">
  <img src="assets/readme/banner.svg" width="860" alt="KVHarbor">
</h1>

<p align="center"><strong>A C++20 server for GPU text generation and CPU speech.</strong></p>

<p align="center">
  <a href="#quick-start">Quick start</a> · <a href="#run-real-models">Run real models</a> · <a href="#architecture">Architecture</a> · <a href="#benchmarks">Benchmarks</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

## Quick start

On Ubuntu 24.04, either natively or in WSL2, use the CPU test backend to try the
streaming API without downloading model weights.

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 build-essential cmake libboost-dev nlohmann-json3-dev
git clone https://github.com/Vitolearns/KVHarbor.git
cd KVHarbor
cmake -S serving -B build-serving-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-serving-cpu --parallel 2
./build-serving-cpu/kvharbor-server --test-backend --port 18080
```

Send a request from a second terminal.

```bash
curl -fsSN http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":4}'
```

Look for four content chunks followed by `data: [DONE]`. Press Ctrl+C in the
server terminal to stop it.

## Run real models

<a name="cuda-text-generation"></a>
<a name="tensorrt-edge-llm-text-generation"></a>
<a name="enable-speech"></a>

| Task | Engine | Run it |
| --- | --- | --- |
| Text from GGUF | llama.cpp on GPU | [Qwen3.8-27B](docs/serving.md#cuda-text-generation) |
| Text from a TensorRT engine | TensorRT Edge-LLM on GPU | [Qwen3.5-0.8B](docs/serving.md#tensorrt-edge-llm-text-generation) |
| Transcribe audio or synthesize speech | sherpa-onnx on CPU | [Qwen3-ASR and Kokoro](docs/serving.md#enable-speech) |

[Interactive demo with switchable LLM and speech](docs/serving.md#interactive-demo)

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/architecture-dark.svg">
    <img src="assets/readme/architecture-light.svg" width="720" alt="KVHarbor in one process. The HTTP API passes typed requests to a shared scheduler with separate queues and adapters for text generation, speech recognition, and speech synthesis.">
  </picture>
</p>

[Runtime and request lifecycle](docs/serving.md#implementation-notes)

## Benchmarks

RTX 5090 Laptop GPU, Qwen3.8-27B Q4_K_M.

| Experiment | Result and scope |
| --- | --- |
| [Serving capacity](docs/experiments/serving-acceptance.md) | 1,152 arrivals. At 0.4 req/s, all three repeats reached 95% SLO compliance. |
| [Single-row logits](docs/experiments/perf-003-gpu.md#results) | CUDA workspace 507 → 124 MiB, with 5 ms higher median HTTP completion. |
| [Greedy sampling](docs/experiments/perf-002-sampling.md) | Median sampling 461 → 108 µs/token, with 0.55% lower median HTTP completion. |

More experiments cover [prefill/decode scheduling](docs/experiments/prefill-decode.md) and [Q4_K shape dispatch](docs/experiments/perf-003-gpu.md#shape-aware-q4_k-dispatch-on-sm120).

## Development

<a name="api"></a>

[API and payloads](docs/serving.md#api) · [Source and tests](docs/serving.md#development)

## License

Contributions covered by this repository's license use
[PolyForm Noncommercial 1.0.0](LICENSE). See [NOTICE](NOTICE.md) for inherited
code, third-party dependencies and model terms.
