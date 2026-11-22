# Repeated single-worker capacity and process recovery

At **0.4 req/s**, all three repeats met the target of 95% SLO compliance.
The sweep offered 1,152 arrivals across three rates. SLO limits were 2,000 ms
arrival-to-first-token and 75 ms TPOT.
[Dataset](../../results/experiments/serving-acceptance.json) ·
[Capacity driver](../../scripts/capacity_bench.py) · [Recovery driver](../../scripts/reliability_bench.py)

## Results

RTX 5090 Laptop GPU, locked 27B Q4_K_M model, one text worker and a reused
2,048-token context.

| Offered req/s | SLO successes in three 128-request runs | Completed / 429 |
| ---: | --- | ---: |
| 0.4 | 128, 126, 128 | 384 / 0 |
| 0.8 | 118, 110, 125 | 384 / 0 |
| 1.6 | 26, 11, 36 | 343 / 41 |

All 1,111 completed responses matched the output digest and token usage of
their fixture and ended with [DONE].

<details>
<summary>Per-run latency and outcomes</summary>

| Offered req/s | Repeat | SLO / offered | Goodput req/s | Arrival TTFT p95 / p99 (ms) | TPOT p95 (ms) | Completed / 429 / timeout / incomplete / failed | Server deadline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 1 | 128/128 | 0.398 | 856 / 1324 | 30.0 | 128 / 0 / 0 / 0 / 0 | 0 |
| 0.4 | 2 | 126/128 | 0.406 | 1335 / 1792 | 30.5 | 128 / 0 / 0 / 0 / 0 | 0 |
| 0.4 | 3 | 128/128 | 0.371 | 932 / 1613 | 30.3 | 128 / 0 / 0 / 0 / 0 | 0 |
| 0.8 | 1 | 118/128 | 0.732 | 2446 / 3343 | 29.1 | 128 / 0 / 0 / 0 / 0 | 0 |
| 0.8 | 2 | 110/128 | 0.702 | 3106 / 3877 | 29.1 | 128 / 0 / 0 / 0 / 0 | 0 |
| 0.8 | 3 | 125/128 | 0.719 | 1778 / 2166 | 29.0 | 128 / 0 / 0 / 0 / 0 | 0 |
| 1.6 | 1 | 26/128 | 0.309 | 5895 / 6166 | 29.1 | 113 / 15 / 0 / 0 / 0 | 0 |
| 1.6 | 2 | 11/128 | 0.133 | 6125 / 6459 | 27.6 | 111 / 17 / 0 / 0 / 0 | 0 |
| 1.6 | 3 | 36/128 | 0.401 | 5797 / 5945 | 29.2 | 119 / 9 / 0 / 0 / 0 | 0 |

</details>

Each of three separate overload bursts completed 9 requests, rejected 15
with HTTP 429 and recovered for three follow-up requests. Including warm-up
and recovery, 1,236 arrivals produced 1,150 accepted and 86 rejected requests.
Accepted requests, terminal records, started workers and releases reconciled.

GPU sampling covered 29.3 minutes and 349 samples, with 15,788–15,798 MiB of
memory and 48–78 C. The restart count stayed at one after the injected crash.

## Fault recovery

Disconnect and 50 ms deadline probes each checked an accepted request, terminal
outcome and worker release, then completed a fresh inference request.

| Check | Observed result |
| --- | --- |
| Disconnect after real content | One client-disconnected terminal and one worker release; idle polling took 51.2 ms after the client returned |
| 50 ms execution deadline | One deadline terminal and one worker release; idle polling took 51.7 ms after the client returned |
| Process crash during SSE | Docker restart count 0 → 1; crash to healthy HTTP 7.483 s; fresh inference completed |
| Current FIFO and decode-first dispatch | Six paired greedy outputs and token usages matched; each mode correctly recorded one HTTP-200 stream interrupted by its deadline |

The crash test uses an isolated Docker-supervised container. The server child
is killed after streamed content, Docker replaces the process, and fresh
inference must complete. The interrupted request and in-memory ledger end with
the old process. [Docker restart policy](https://docs.docker.com/engine/containers/start-containers-automatically/)

The three-worker regression observed 32 prioritized decodes and 10 bounded-wait
prefills. Its counters verify dispatch execution. The latency and goodput
comparison has its own [report](prefill-decode.md).

## Limits

Each rate covers 384 arrivals across three repeats. The 1,000-arrivals-per-scenario
target for each repeat and day-long reliability evaluation are pending.
This report covers one model, one text worker and the recorded fault cases.

<details>
<summary>Configuration, measurement definitions and reproduction</summary>

## Scope and identity

This run uses KVHarbor's own C++ HTTP/SSE service, the locked 27B Q4_K_M
weights, and one RTX 5090 Laptop GPU. It follows the original
[24-request capacity sweep](serving-capacity.md) with a separate binary and
schema-2 measurement protocol. Historical results retain their original
source and binary fingerprints.

The text service has one worker, a reused 2,048-token context, 512-token
prefill batches, four CPU threads, direct greedy sampling, one output row,
FIFO admission, eight queue entries and an 8 s server deadline. Engine dispatch
is off for this single-worker baseline. The workload cycles through the same
short/long/short fixtures. There is no cross-request prefix cache.

Three rates (0.4, 0.8, 1.6 req/s) each have three 128-request Poisson runs,
using seeds 41, 42 and 43. Rate order is reversed in the second repeat.
Warm-up, overload bursts and recovery requests are separate from the 1,152
measured arrivals. GPU temperature, clock, power, utilization and memory are
sampled every five seconds through nvidia-smi; unsupported fields remain
visible in the sampled output. Device clocks are not locked.

## Measurement definitions

A completed response requires HTTP 200, successful SSE parsing and [DONE].
HTTP 200 followed by a missing end marker is reported as incomplete SSE.
Client or explicit HTTP timeouts have a separate count. The server's terminal
deadline count is reported independently; an incomplete stream is not assumed
to be a deadline merely because another request timed out.

The SLO is arrival-to-first-token at most 2,000 ms and TPOT at most 75 ms.
Arrival-to-first-token includes client scheduling/queue delay; send-to-first-
token latency is also retained. Responses with at most one generated token use
the first-token criterion when TPOT is unavailable. Goodput divides compliant
responses by the full observed measurement window, including the drain.
All latency percentiles are nearest rank over completed responses with the
corresponding metric. Their sample counts are recorded. At 128 completions,
p99 is only the second-largest observation, so these are limited tail samples.

The driver checks `accepted = terminal + active + queued` and
`started = slot_releases + active` from a single reactor snapshot. Each run
also reconciles client arrivals with server accepted plus rejected counters.
This distinguishes queued cancellations, which never acquired a worker, from
running jobs, which must return one lease.

## Reproduction

Build the text runtime as in the [CUDA setup](../serving.md#cuda-text-generation).
Start an isolated instance with the verified local weights.

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
export KVHARBOR_NATIVE_NAME=kvharbor-acceptance
bash scripts/native_serving.sh supervise --port 8083 --workers 1 \
  --context 2048 --batch 512 --threads 4 --gpu-layers 99 \
  --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

In another terminal, run fault acceptance first, then the uninterrupted
capacity sweep. The driver waits for the service to become healthy. The full
capacity command takes about half an hour. Its current schema-2 reports are
written below `$run`, with the current source and binary fingerprints.

```bash
export KVHARBOR_NATIVE_NAME=kvharbor-acceptance
mkdir -p results/raw
run=$(mktemp -d "$PWD/results/raw/replay-acceptance.XXXXXX")
python3 scripts/reliability_bench.py --url http://127.0.0.1:8083 \
  --model Qwen3.8-27B-Q4_K_M --container "$KVHARBOR_NATIVE_NAME" --crash \
  --output-dir "$run/reliability"
python3 scripts/capacity_bench.py --url http://127.0.0.1:8083 \
  --model Qwen3.8-27B-Q4_K_M --rates 0.4 0.8 1.6 --requests 128 --repeats 3 \
  --server-binary build-serving-cuda/kvharbor-server \
  --model-file models/Qwen3.8-27B-Q4_K_M.gguf --output-dir "$run/capacity"
```

Keep logs before stopping and removing this test instance:

```bash
docker logs "$KVHARBOR_NATIVE_NAME" > "$run/server.log" 2>&1
docker stop "$KVHARBOR_NATIVE_NAME"
docker rm "$KVHARBOR_NATIVE_NAME"
```

### Dispatch regression

Start the three-worker [dispatch profile](prefill-decode.md#replay).
The live [regression script](../../serving/tests/test_dispatch_http.py) checks
the configured order and compares a decode-first run with a saved FIFO result.

## Source and artifact integrity

The [source archive](../../results/experiments/serving-acceptance-source.tar.gz)
contains the measured runtime and drivers, validation sources, runtime lock,
fixtures, LICENSE and NOTICE. Each archived file has a SHA-256 in snapshot.json.
Measured runtime files match the fingerprints taken before the sweep. The
dataset records the archive and server binary digests. The current regression
records its own test-source digest, separately from the historical dispatch
measurements.

</details>
