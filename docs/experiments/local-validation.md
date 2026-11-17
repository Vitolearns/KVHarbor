# Single-GPU serving validation

This dataset covers the initial upstream capacity sweep, native service
checks, context reuse, queue order, and audio in isolation and alongside
text. It was collected on one NVIDIA GeForce RTX 5090 Laptop GPU with the
[locked 27B GGUF](../../configs/runtime/llama.json) and
[locked audio models](../../configs/runtime/audio.json).
[Dataset](../../results/experiments/local-validation.json)

## Text capacity and request lifecycle

The upstream llama-server baseline used 2 slots
with a total context of 4,096 tokens. A fixed
backlog of 18 [mixed-length requests](../../benchmarks/kvharbor_text_mixed.jsonl)
was replayed at client concurrency 1, 2 and 4. A separate Poisson
replay offered 24 requests at 2 requests/s, using seed 42. The native
KVHarbor single-worker run used its own executable and is listed under
`native_service`. Approximately 24.2 and 34.5 output tokens/s were
observed at client concurrency 1 and 2 in the upstream baseline. The
p95 time to first token increased as concurrency rose from 2 to 4.
These are capacity probes for that configuration. They leave the
current service's SLO-constrained capacity undetermined.

For context reuse, the native runtime ran identical fixtures using a new
context per request or clearing and reusing the existing allocation.
Nine native token-sequence pairs and 18 HTTP output pairs matched.
Median native context preparation fell from about 25.3 to 0.47 ms and
HTTP first output p50 fell from 89.2 to 57.0 ms. HTTP tail latency did
not consistently improve.

FIFO and deadline ordering were compared with the same
[six-request burst](../../benchmarks/kvharbor_deadline_burst.jsonl),
offered at 10 requests/s with concurrency 6 and a 4,000 ms TTFT SLO.
Three requests completed under FIFO and six under deadline ordering.
The small burst is a policy check. The fault cases exercise queue,
cancellation and resource release behavior independently.

## Audio and mixed workload

The audio benchmark ran six rounds each of LLM alone, ASR alone, TTS
alone, and the mixed phase on the same service. The JSON reports
per-request latency, real-time factor where applicable, peak observed
activity and the ledger delta. ASR can return a deadline response before
native decoding finishes. The measured recovery interval is recorded
separately from the HTTP deadline. The TTS disconnect probe measures
when the slot becomes available again.

The ASR check used six upstream audio samples with supplied reference
transcripts. It computes character error rate from edit distance
divided by the normalized reference character count. The dataset
contains per-sample checksums, character counts, errors and timing.
All six completed, with micro CER of 10.59% on this sample set. The
TTS benchmark compares timing, duration and amplitude. The tested
model has random operators, so repeated PCM need not match byte for
byte. No listening panel was used.

## Reproduce

Start the desired runtime as described in the
[quick start](../../README.md#quick-start). Use one service at a time
with the intended slot or worker count. The JSON contains any
additional client settings for each series. For example, this is the
open-loop run against llama-server. Set `BASE_URL` and `MODEL_ID` to
match that instance.

```bash
python3 scripts/stream_bench.py --base-url "$BASE_URL" --model "$MODEL_ID" \
  --workload benchmarks/kvharbor_text_mixed.jsonl --requests 24 \
  --concurrency 8 --rate 2 --arrival poisson --seed 42 --timeout 120 \
  --output results/raw/replay-open-loop.json
```

Use `--reuse-context` on the native server for the reused-context
comparison. For the burst, switch between `--policy fifo` and
`--policy deadline` on the server and use the saved client settings.
For the audio checks, start KVHarbor with the audio
backends from the [speech setup](../../README.md#enable-speech).
Set `ASR_MODEL_DIR` to the extracted locked model package and
`BASE_URL` to the running service.

```bash
python3 scripts/asr_bench.py --base-url "$BASE_URL" --model-dir "$ASR_MODEL_DIR" \
  --output results/raw/replay-asr.json
python3 scripts/audio_bench.py --base-url "$BASE_URL" --rounds 6 \
  --workload benchmarks/kvharbor_text_mixed.jsonl \
  --output results/raw/replay-audio/audio-report.json
```

The data can be read without a GPU. `ttft_ms` starts at client dispatch,
`e2e_ms` ends at response completion, and `client_queue_ms` is time
spent waiting to dispatch. Request positions are local to a run.
