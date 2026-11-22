# Interleaving prefill and decode

Decode-first lowered TPOT p95 by 9.0%–19.0% in four paired runs. TTFT p95 improved
in one pair, and SLO goodput stayed flat or decreased. The policy is opt-in.

Each worker owns a single-sequence context. A shared queue permits up to four
decode calls ahead of a waiting prefill, then serves the oldest prefill.
The permit covers native execution and synchronization.

## Controlled comparison

Three workers used the same CUDA binary, locked 27B weights, 2,048-token contexts
and 64-token prefill chunks. Each engine order ran two 24-request repeats at
1.6 and 3.2 offered req/s.

<details>
<summary>Per-run latency and outcomes</summary>

| Offered req/s | Repeat | Engine order | TTFT p95 / p99 (ms) | TPOT p95 (ms) | Goodput req/s | Completed / 429 / incomplete SSE | Server deadline |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.6 | 1 | fifo | 5678 / 6127 | 95.1 | 0.052 | 24 / 0 / 0 | 0 |
| 1.6 | 1 | decode-first | 6105 / 6451 | 86.5 | 0.051 | 24 / 0 / 0 | 0 |
| 1.6 | 2 | fifo | 5743 / 5743 | 106.9 | 0.062 | 16 / 3 / 5 | 5 |
| 1.6 | 2 | decode-first | 5626 / 5626 | 86.6 | 0.062 | 16 / 3 / 5 | 5 |
| 3.2 | 1 | fifo | 6755 / 6755 | 95.9 | 0.000 | 14 / 7 / 3 | 3 |
| 3.2 | 1 | decode-first | 6806 / 6806 | 86.7 | 0.000 | 14 / 7 / 3 | 3 |
| 3.2 | 2 | fifo | 5915 / 5915 | 95.8 | 0.087 | 12 / 9 / 3 | 3 |
| 3.2 | 2 | decode-first | 6893 / 6893 | 86.3 | 0.084 | 13 / 9 / 2 | 2 |

Percentiles cover completed requests. Server deadlines also count HTTP-200
streams interrupted before [DONE].

</details>

## Correctness and dispatch evidence

All 63 requests completed in both arms of a pair matched in output digest and
token usage. Traces recorded 528 decodes ahead of a waiting prefill and 159
prefills selected after the decode limit. A disconnect released one worker
and a subsequent request completed.
[Dataset](../../results/experiments/prefill-decode.json)

FIFO order, decode priority, the prefill bound and cancellation while waiting
are covered by the [C++ test](../../serving/tests/engine_schedule_test.cc).

<details>
<summary>Setup, replay and measured source</summary>

Both orders used three text workers, FIFO request admission, 2,048 context
tokens, 64-token prefill chunks, direct greedy sampling and one reserved
logits row. They used the same CUDA binary, 27B weights and
[mixed-length fixtures](../../benchmarks/kvharbor_text_mixed.jsonl). Each mode
ran two repeats at 1.6 and 3.2 offered requests/s, using the same Poisson
seeds and 24 requests per run. The setting for engine dispatch order was
the only intended difference. The SLO and percentile definitions are in
the [single-worker report](serving-capacity.md).

## Replay

Use the [CUDA build](../serving.md#cuda-text-generation) and run each mode on the
same GPU with the same environment. Set ENGINE_ORDER to fifo or decode-first.

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
ENGINE_ORDER=fifo
bash scripts/native_serving.sh serve --port 8083 --workers 3 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo \
  --prefill-chunk 64 --engine-schedule "$ENGINE_ORDER"
```

In another Bash terminal, run the client below. Then restart the server and
repeat with `ENGINE_ORDER=decode-first` in both terminals. Each output directory
contains `report.json`. Current replays use schema-2 arrival-based SLO accounting.

```bash
ENGINE_ORDER=fifo
mkdir -p results/raw
run=$(mktemp -d "$PWD/results/raw/replay-prefill.XXXXXX")
python3 scripts/capacity_bench.py --url http://127.0.0.1:8083 \
  --model Qwen3.8-27B-Q4_K_M --rates 1.6 3.2 --requests 24 --repeats 2 --burst-count 18 \
  --server-binary build-serving-cuda/kvharbor-server \
  --model-file models/Qwen3.8-27B-Q4_K_M.gguf --output-dir "$run/$ENGINE_ORDER"
```

## Measured source snapshot

The [source archive](../../results/experiments/prefill-decode-source.tar.gz)
contains 19 runtime, driver, fixture and project-notice files. Every archived measured source
was checked byte-for-byte against this dataset's recorded SHA-256. Retained
versions supply the files that have since changed in the working tree. Two
generated Python bytecode entries are listed as excluded; they are not treated
as source. The archive includes a per-file manifest and the locked runtime and
workload inputs. Its SHA-256 is recorded in the dataset's `source_snapshot`.

The current service has additional started-worker accounting and runtime
configuration fields. Those additions do not relabel these historical
measurements as runs of the current binary.

</details>
