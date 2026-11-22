# Serving guide

[README](../README.md) · [简体中文](serving.zh-CN.md)

## Build prerequisites

Use Ubuntu 24.04 or WSL2, Bash, Git, curl, Python 3.12+, Docker with NVIDIA GPU
access, `libboost-dev` and `nlohmann-json3-dev`. The reference GPU is an
RTX 5090 Laptop, and the pinned build targets SM120.

Run commands from the repository root. Build the shared CUDA dependencies once.
Run servers in the foreground, send requests from another terminal, and stop
the server with Ctrl+C before switching backends.

```bash
image=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["build_image"])')
docker pull "$image"
python3 scripts/llama_runtime.py build --jobs 4
```

## CUDA text generation

Save the locked Qwen3.8-27B Q4_K_M GGUF in `models/`, verify it, then build
and start the server. [Weights](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/resolve/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf) · [Runtime lock](../configs/runtime/llama.json)

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha256=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["model"]["sha256"])')
printf '%s  %s\n' "$model_sha256" "$GGUF_MODEL" | sha256sum --check -
bash scripts/native_serving.sh build
bash scripts/native_serving.sh serve --text-backend llama.cpp \
  --workers 1 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

Check health and send a request. A complete SSE response ends with `data: [DONE]`.

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsSN http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Q4_K_M","stream":true,"temperature":0,"max_tokens":48,
       "messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}]}'
```

## TensorRT Edge-LLM text generation

Follow the [TensorRT recipe](tensorrt.md) to prepare the SDK, export environment
and engine, then serve Qwen3.5-0.8B. The reference engine accepts 1,024 input
tokens and a 2,048-token KV cache.

## Enable speech

The preparation script verifies sherpa-onnx, Qwen3-ASR and Kokoro.
Use `--text-backend none` for speech-only serving. For mixed serving, set
`KVHARBOR_AUDIO=1` during build and launch, and append both audio model flags
to the chosen text command. [Runtime lock](../configs/runtime/audio.json)

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_SERVING_BUILD_DIR
export KVHARBOR_AUDIO=1
python3 scripts/prepare_audio_runtime.py
bash scripts/native_serving.sh build
bash scripts/native_serving.sh serve --text-backend none \
  --asr-model-dir /workspace/models/audio/qwen3-asr-0.6b-int8 \
  --tts-model-dir /workspace/models/audio/kokoro-v1.0
```

Transcribe the bundled WAV or synthesize PCM16. The output sample rate is in
`X-Audio-Sample-Rate`.

```bash
curl -fsS http://127.0.0.1:8082/v1/audio/transcriptions \
  -H 'Content-Type: audio/wav' --data-binary @assets/readme/ferry.wav
curl -fsS http://127.0.0.1:8082/v1/audio/speech \
  -H 'Content-Type: application/json' -d '{"input":"See you at the pier."}' \
  -D speech.headers -o speech.pcm
```

To combine ASR and TTS with llama.cpp, stop the speech-only server above.
In the same terminal, set the GGUF path and launch the text backend using the
audio build.

```bash
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
bash scripts/native_serving.sh serve --text-backend llama.cpp --context 2048 \
  --workers 1 --reuse-context --max-output-rows 1 --timeout-ms 180000 \
  --asr-model-dir /workspace/models/audio/qwen3-asr-0.6b-int8 \
  --tts-model-dir /workspace/models/audio/kokoro-v1.0
```

## Interactive demo

Prepare the GGUF, [Edge engine and SDK](tensorrt.md), and the speech models above.
Build one server with both text backends and speech enabled.

```bash
export KVHARBOR_AUDIO=1
export KVHARBOR_EDGE_SDK="$PWD/.cache/kvharbor/edge-sdk-v0.12.0-sm120"
export KVHARBOR_SERVING_BUILD_DIR="$PWD/build-serving-262k-edge"
bash scripts/native_serving.sh build
bash scripts/demo/serve.sh --llama
```

In a second terminal run `python3 -B scripts/demo/client.py`. Use `--trt` on
the launcher to start with TensorRT. Set `KVHARBOR_DEMO_PORT` in both terminals
to use a free port other than 18082. Stop the launcher with Ctrl+C.

| In the client | What happens |
| --- | --- |
| A question or `/engine` | Stream a reply or show the current text backend. |
| `/engine trt`, `/engine llama` | Reload the model on the same GPU and port. On a change, text history resets and the client reads the new output limit. |
| `/tts See you at the pier.`, `/play` | Synthesize a WAV, then play the most recent WAV. |
| `/asr`, `/asr assets/readme/ferry.wav` | Transcribe the last synthesized WAV or a supplied mono PCM16 WAV. |
| `/mix`, `/status` | Submit text, transcription and synthesis concurrently or inspect the worker queues. |

The reference profiles use 262,144 context / 16,384 output tokens for llama.cpp and
2,048 / 1,024 for TensorRT. Logs and WAVs are saved under `.local/demo/runs/`.

## API

| Endpoint | Request → response |
| --- | --- |
| `POST /v1/chat/completions`, `POST /v1/completions` | JSON with `stream: true` and `temperature: 0` → SSE |
| `POST /v1/audio/transcriptions` | Mono PCM16 WAV → JSON transcript |
| `POST /v1/audio/speech` | JSON text → PCM16 with a sample-rate header |
| `GET /health`, `/state`, `/metrics`, `/v1/models` | Runtime health, task occupancy, Prometheus metrics and loaded models |

Integer fields such as `max_tokens` and voice IDs require JSON integers.
`--max-output-tokens` caps generated tokens, and `--timeout-ms` sets the
request deadline. Both configured limits appear in `/health`.

## Implementation notes

<a name="contexts-and-scheduling"></a>
<a name="sampling-and-output-capacity"></a>
<a name="cancellation-and-ownership"></a>

| Mechanism | Behavior |
| --- | --- |
| Worker ownership | Each llama.cpp worker owns one single-sequence context until native execution finishes. |
| `--reuse-context` | Retains allocations and clears attention KV and recurrent state between requests. |
| `--policy fifo\|deadline` | Orders waiting HTTP requests. |
| `--engine-schedule fifo\|decode-first` | Interleaves serial prefill/decode calls across independent contexts, with cancellable permit waits. |
| Sampling | Greedy decoding. Direct argmax borrows logits and selects AVX2 or scalar code. One-row output consumes the last logits row. |
| Long contexts | llama.cpp accepts up to 262,144 context tokens. `--kv-cache-q8` selects Q8_0 KV and flash attention. Edge-LLM capacity follows its built engine. |

On disconnect or expiry, cancellation stops the response pipe and wakes blocked
producers. A running worker releases its slot after engine synchronization.
Queued cancellation removes the queue entry. Request and worker accounting use
`accepted = terminal + active + queued` and `started = slot_releases + active`.

<a name="evidence-identity"></a>
<a name="performance-interpretation"></a>

Each experiment records its source, binary, workload and measurement scope.
Stage timings and end-to-end results are reported together in the
[sampling](experiments/perf-002-sampling.md) and
[dispatch](experiments/prefill-decode.md) reports.

## Development

Build the [CPU target](../README.md#quick-start), then run the checks below.
Source entry points are [HTTP transport](../serving/src/server.cc),
[admission](../serving/src/admission.cc) and the
[runtime interface](../serving/include/kvharbor/serving/runtime.h).

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
python3 tests/test_stream_bench.py
python3 tests/test_capacity_bench.py
```

[README diagram source](../scripts/render_readme_architecture.py) · [Attribution and dependency terms](../NOTICE.md)
