# Interleaving prefill and decode

KVHarbor can dispatch the text worker's real llama.cpp prefill chunks and
decode steps through a shared queue. Each worker owns a separate single-sequence
context. The experimental decode-first order lets up to four decodes run while
a prefill is queued, then gives the oldest waiting prefill a turn.
The permit covers the engine call and synchronization. Streaming and
sampling stay on the request's worker. A cancelled waiter leaves the queue
before it acquires the permit.

## Controlled comparison

Both orders used three text workers, FIFO request admission, 2,048 context
tokens, 64-token prefill chunks, direct greedy sampling and one reserved
logits row. They used the same CUDA binary, 27B weights and
[mixed-length fixtures](../../benchmarks/kvharbor_text_mixed.jsonl). Each mode
ran two repeats at 1.6 and 3.2 offered requests/s, using the same Poisson
seeds and 24 requests per run. The setting for engine dispatch order was
the only intended difference. The SLO and percentile definitions are in
the [single-worker report](serving-capacity.md).

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

Decode-first reduced TPOT p95 by 9.0% to 19.0% across the four paired runs. TTFT p95 improved in 1 of four pairs. SLO goodput did not increase in these runs. The decode-first mode is opt-in. The rates in this
experiment create substantially more contention than the single-worker SLO
baseline, and the percentiles describe only completed requests. The server
deadline column includes requests whose HTTP 200 stream ended before
[DONE], as well as any explicit deadline responses. These counts use the
server's terminal records.

## Correctness and dispatch evidence

The [request-level dataset](../../results/experiments/prefill-decode.json)
contains the workload and binary fingerprints, per-run outcomes, and
paired output digests. 63 requests completed in both members of the
same pair with identical output digests and token usage. Across the
measured runs and bursts, the decode-first terminal records include
528 decodes selected ahead of a waiting prefill and 159 prefills
selected after the decode limit. The release test disconnected a real
client after its first streamed content, observed one terminal outcome and
one worker release, and completed a subsequent request. The
[C++ test](../../serving/tests/engine_schedule_test.cc) covers FIFO order,
decode priority, the prefill bound, and cancellation while waiting.

## Replay

Use the [CUDA build](../../README.md#cuda-text-generation) and run each mode on the
same GPU with the same environment. Set ENGINE_ORDER to fifo or decode-first.

```bash
bash scripts/native_serving.sh serve --workers 3 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo --prefill-chunk 64 --engine-schedule "$ENGINE_ORDER"
```

In another Bash terminal, set BASE_URL and MODEL_ID to match the running
service. Use a fresh output directory per mode.

```bash
python3 scripts/capacity_bench.py --url "$BASE_URL" --model "$MODEL_ID" \
  --rates 1.6 3.2 --requests 24 --repeats 2 --burst-count 18 \
  --output-dir "results/raw/replay-prefill-$ENGINE_ORDER"
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
