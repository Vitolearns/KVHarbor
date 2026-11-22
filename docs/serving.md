# Serving guide

[README](../README.md) · [简体中文](serving.zh-CN.md)

## Build prerequisites

The reference environment is Ubuntu 24.04 on Linux or WSL2 with an
RTX 5090 Laptop GPU. The pinned CUDA build targets SM120. Use Bash, Git, curl,
Python 3.12+, Docker with NVIDIA GPU access, `libboost-dev` and
`nlohmann-json3-dev`. Run the commands from the repository root.

The native launcher uses the same image and llama.cpp build for all three
setups below. Pull the image and build llama.cpp once.

```bash
image=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["build_image"])')
docker pull "$image"
python3 scripts/llama_runtime.py build --jobs 4
```

## CUDA text generation

Create `models/` and save the [locked GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/resolve/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf) there.
The [text runtime lock](../configs/runtime/llama.json) specifies the model
checksum, llama.cpp revision, and CUDA image.

```bash
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha256=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["model"]["sha256"])')
printf '%s  %s\n' "$model_sha256" "$GGUF_MODEL" | sha256sum --check -
```

Build the server after the checksum passes.

```bash
bash scripts/native_serving.sh build
```

Start the server in the foreground.

```bash
bash scripts/native_serving.sh serve --text-backend llama.cpp \
  --workers 1 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

In another terminal, check health and send a chat request. The last SSE record
is `data: [DONE]`.

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsSN http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Q4_K_M","stream":true,"temperature":0,"max_tokens":48,
       "messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}]}'
```

## TensorRT Edge-LLM text generation

Complete the [build prerequisites](#build-prerequisites) first.
Install the TensorRT Edge-LLM SDK specified in the [runtime lock](../configs/runtime/edge-llm.json).
The reference text engine uses Qwen3.5-0.8B, one sequence, up to 1,024 input
tokens and a 2,048-token KV cache. Install the SDK's ONNX export dependencies
for `trtedge export`. Build on the GPU that will serve the engine. If the build
image needs extra TensorRT builder resources, set `KVHARBOR_TRT_BUILDER_LIB_DIR`
to the directory containing the matching libraries. The launcher mounts the
repository at `/workspace`.

Stop the running text server before switching backends.

```bash
python3 scripts/prepare_pc_model.py Qwen3.5-0.8B
export KVHARBOR_EDGE_SDK=/absolute/path/to/edge-llm-sdk
trtedge export models/qwen3.5-0.8b .cache/qwen3.5-edge-onnx --skip-visual --skip-audio
bash scripts/build_pc_edge_engine.sh .cache/qwen3.5-edge-onnx/llm
bash scripts/native_serving.sh build

bash scripts/native_serving.sh serve --text-backend tensorrt-edge-llm \
  --model-id Qwen3.5-0.8B --context 2048 --workers 1 \
  --engine-dir /workspace/models/qwen3.5-0.8b-edge \
  --checkpoint-dir /workspace/models/qwen3.5-0.8b
```

Check `GET /health` for `runtimes.llm.backend` set to `tensorrt-edge-llm`.
Reuse the chat request above with `"model":"Qwen3.5-0.8B"`. CUDA Graph is
enabled by default and `--edge-no-cuda-graph` disables it for a comparison.

## Enable speech

Complete the [build prerequisites](#build-prerequisites) and stop any
server using port 8082. The preparation script verifies sherpa-onnx,
Qwen3-ASR and Kokoro against the [speech runtime lock](../configs/runtime/audio.json).
Use `--text-backend none` to start speech on its own without a GGUF model.
For mixed serving, build and serve with `KVHARBOR_AUDIO=1` and append the
ASR and TTS model directory flags to the chosen text server command.

```bash
python3 scripts/prepare_audio_runtime.py
KVHARBOR_AUDIO=1 bash scripts/native_serving.sh build
KVHARBOR_AUDIO=1 bash scripts/native_serving.sh serve --text-backend none \
  --asr-model-dir /workspace/models/audio/qwen3-asr-0.6b-int8 \
  --tts-model-dir /workspace/models/audio/kokoro-v1.0
```

In another terminal, transcribe the bundled Kokoro WAV and synthesize a raw
PCM response. The `X-Audio-Sample-Rate` response header gives the output rate.

```bash
curl -fsS http://127.0.0.1:8082/v1/audio/transcriptions \
  -H 'Content-Type: audio/wav' --data-binary @assets/readme/ferry.wav
curl -fsS http://127.0.0.1:8082/v1/audio/speech \
  -H 'Content-Type: application/json' -d '{"input":"See you at the pier."}' \
  -D speech.headers -o speech.pcm
```

## Implementation notes

[NOTICE.md](../NOTICE.md) records attribution. The [llama.cpp](../configs/runtime/llama.json),
[Edge-LLM](../configs/runtime/edge-llm.json) and [speech](../configs/runtime/audio.json)
runtime locks identify upstream components and model artifacts.

### Contexts and scheduling

In the llama.cpp adapter, each worker owns one request until native execution returns. Each context
contains one sequence (`n_seq_max = 1`). Context reuse retains allocations and
clears both attention KV and recurrent state between requests. It does not
reuse the previous request's prefix.

With `--engine-schedule off`, the worker calls the engine directly. With FIFO
or decode-first, `--prefill-chunk` exposes scheduling points between real
prefill chunks and decode steps in independent contexts. This is interleaving
of serial engine calls. It does not combine several sequences into one
`llama_batch`, implement cross-request continuous batching, or replace the
upstream KV allocator. Sampling and socket output stay with the request worker.

`--policy fifo|deadline` controls waiting HTTP requests. It is independent of
`--engine-schedule fifo|decode-first`, which controls engine calls after a
request has acquired a worker. Waiting for the engine is cancellable; an
already executing call retains its permit until synchronization finishes.

### Sampling and output capacity

Direct greedy argmax borrows the logits array and has runtime AVX2 dispatch
with a scalar fallback. The adapter supports temperature zero. Legacy and
upstream sampler paths remain available for comparison.

The one-output-row setting uses llama.cpp's existing output-capacity parameter.
The current adapter consumes only the last logits row from each single-sequence
call. Multi-sequence batching would require reassessing this assumption.

Integer request fields, including `max_tokens` and speech voice IDs, require
JSON integers within their supported range. Booleans, strings and floating-
point values are rejected. Numeric fields such as temperature and speech
speed accept JSON numbers and reject booleans.

### Evidence identity

The [sampling](../docs/experiments/perf-002-sampling.md),
[GPU](../docs/experiments/perf-003-gpu.md),
[single-worker capacity](../docs/experiments/serving-capacity.md), and
[dispatch](../docs/experiments/prefill-decode.md) reports identify their own runtime
settings. Historical upstream llama-server slot measurements belong to that
binary. They are not measurements of KVHarbor's single-worker service.

New measurements retain the source and binary used during the run. Historical
datasets keep their original fingerprints when the current source changes.
Small samples, partial streams and negative end-to-end results remain visible.

### Cancellation and ownership

The reactor owns admission and terminal transitions. A running job retains its
slot while the worker is inside the native call. Disconnect or expiry sets the
shared cancellation flag, stops the response pipe and wakes blocked producers.
The worker returns after engine synchronization; the reactor then clears the
slot, increments its release counter and records one terminal outcome. Queued
cancellation removes the job without incrementing either started workers or
worker releases.

`accepted = terminal + active + queued` checks request accounting.
`started = slot_releases + active` checks worker ownership. Requiring every
accepted request to have a worker release would incorrectly count queued
timeouts as leaks. Requiring releases merely to stay below terminal count
would miss a leaked release. The capacity driver checks both equalities.

The dispatcher has a separate engine-call permit. A cancelled waiter removes
its ticket before acquiring that permit. An executing operation returns it
through a scope guard. This permit is narrower than the worker lease:
sampling and network backpressure can keep a worker occupied between engine
calls without keeping the dispatch permit.

### Performance interpretation

Sampling speedups apply to the sampling stage. The HTTP request also includes
queueing, prefill, matrix-vector operations, synchronization and transport.
The sampling experiment reduced stage p50 from 461.42 to 107.61 microseconds per
token while HTTP completion p50 changed from 1,632.85 to 1,623.92 ms. Those
measurements support a local optimization with a small end-to-end effect.
They do not support a 76.7% request-level speedup.

The dispatch experiment makes a similar tradeoff visible: decode-first lowered
TPOT p95 in its four paired runs but did not improve SLO goodput. The opt-in
policy and its negative TTFT results remain part of the published experiment.

## Development

The service entry points are [HTTP transport](../serving/src/server.cc),
[task admission](../serving/src/admission.cc), and the
[runtime interface](../serving/include/kvharbor/serving/runtime.h). The four README SVGs share one
[diagram source](../scripts/render_readme_architecture.py).

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
python3 tests/test_stream_bench.py
python3 tests/test_capacity_bench.py
```
