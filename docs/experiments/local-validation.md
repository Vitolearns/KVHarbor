# Initial text and speech validation

<a name="single-gpu-serving-validation"></a>

Initial probes on an RTX 5090 Laptop GPU covered the upstream llama-server,
KVHarbor's native context reuse, request ordering and CPU speech.
[Dataset](../../results/experiments/local-validation.json) ·
[Text runtime](../../configs/runtime/llama.json) · [Audio runtime](../../configs/runtime/audio.json)

## Text capacity and request lifecycle

| Check | Observed result |
| --- | --- |
| Upstream llama-server | Two slots, 4,096 total context tokens. About 24.2 / 34.5 output tokens/s at client concurrency 1 / 2. TTFT p95 rose from concurrency 2 to 4. |
| Native context preparation | Median 25.3 → 0.47 ms with allocation reuse. |
| HTTP first output | Median 89.2 → 57.0 ms with reuse, with variable tail latency. |
| Reuse correctness | 9 native token-sequence pairs and 18 HTTP output pairs matched. |
| Six-request ordering burst | FIFO completed 3, deadline ordering completed 6. |

## Audio and mixed workload

Six rounds per isolated task and mixed phase recorded per-request latency,
real-time factor, peak activity and ledger deltas.

| Check | Observed result |
| --- | --- |
| ASR quality | 6/6 upstream recordings completed, micro CER 10.59%. The dataset stores references, character counts, errors and checksums. |
| TTS | Timing, duration and amplitude checks on stochastic PCM. Human listening evaluation is pending. |
| Cancellation | ASR recovery is timed through native decode completion. TTS disconnect checks when its worker slot becomes available. |

<details>
<summary>Method and reproduction</summary>

## Method

The upstream baseline replayed 18 mixed-length requests at concurrency 1, 2
and 4, then 24 Poisson arrivals at 2 req/s with seed 42. Its measurements belong
to the upstream binary. KVHarbor's single-worker records are under
`native_service`. The repeated native capacity sweep is documented
[separately](serving-acceptance.md).

The reuse comparison used identical fixtures with fresh or cleared contexts.
The six-request FIFO/deadline burst ran at 10 req/s, concurrency 6 and a
4,000 ms TTFT SLO. Fault cases independently checked resource release.
[Mixed-length workload](../../benchmarks/kvharbor_text_mixed.jsonl) ·
[Burst workload](../../benchmarks/kvharbor_deadline_burst.jsonl)

ASR CER is edit distance divided by normalized reference character count.
TTS checks record waveform identity because the model uses random operators.

## Reproduce

Complete the [CUDA build](../serving.md#cuda-text-generation). Stop the current
text server before starting the two-slot upstream baseline.

```bash
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
python3 scripts/llama_runtime.py start --model "$GGUF_MODEL" \
  --name kvharbor-replay-upstream --port 8081 --context 4096 --slots 2
```

Run the open-loop client in another terminal, then stop the baseline.

```bash
mkdir -p results/raw
run=$(mktemp -d "$PWD/results/raw/replay-initial.XXXXXX")
python3 scripts/stream_bench.py --base-url http://127.0.0.1:8081 \
  --model Qwen3.8-27B-Q4_K_M --workload benchmarks/kvharbor_text_mixed.jsonl \
  --requests 24 --concurrency 8 --rate 2 --arrival poisson --seed 42 --timeout 120 \
  --output "$run/open-loop.json"
python3 scripts/llama_runtime.py stop --name kvharbor-replay-upstream
docker rm kvharbor-replay-upstream
```

For native context reuse, run the same fixtures with fresh and reused contexts.
The ordering burst uses `--policy fifo` or `--policy deadline` and the six-request
fixture. Their original settings are recorded in the dataset.

For audio, start the [LLM/ASR/TTS profile](../serving.md#enable-speech), then run
the client below. `asr_bench.py --output` is a JSON file.
`audio_bench.py --output` is a directory containing `audio-report.json` and WAVs.

```bash
mkdir -p results/raw
audio_run=$(mktemp -d "$PWD/results/raw/replay-audio.XXXXXX")
python3 scripts/asr_bench.py --base-url http://127.0.0.1:8082 \
  --model-dir models/audio/qwen3-asr-0.6b-int8 --output "$audio_run/asr.json"
python3 scripts/audio_bench.py --base-url http://127.0.0.1:8082 --rounds 6 \
  --workload benchmarks/kvharbor_text_mixed.jsonl --output "$audio_run/audio"
```

`ttft_ms` starts at client dispatch, `e2e_ms` ends at response completion, and
`client_queue_ms` measures dispatch wait. Request positions are local to a run.

</details>
