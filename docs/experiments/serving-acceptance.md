# Repeated single-worker capacity and process recovery

[Dataset](../../results/experiments/serving-acceptance.json) ·
[capacity driver](../../scripts/capacity_bench.py) ·
[recovery driver](../../scripts/reliability_bench.py)

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

## Results

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

The highest tested offered rate reaching at least 95% SLO compliance in every
repeat was **0.4 req/s** (128/128, 126/128, 128/128). At 0.8 req/s, the
counts were 118/128, 110/128 and 125/128. The 1.6 req/s runs reached
26/128, 11/128 and 36/128; most misses were waiting for the worker. This
sweep does not locate a precise capacity boundary between the tested rates.

The measured arrivals produced 1,111 completed responses and 41 HTTP 429s.
All completed measured responses had the same output digest and token usage
within each of the three fixtures. No normal-run response was counted as
completed without [DONE].

Each of three separate 24-request overload bursts completed 9 requests and
returned 15 HTTP 429s; each was followed by three complete recovery requests.
Including warm-up, bursts and recovery, the driver offered 1,236 requests:
1,150 accepted and 86 rejected. All 1,150 accepted requests had terminal
records, and started workers and worker releases both totalled 1,150.

GPU sampling covered 29.3 minutes in 349 samples. Observed
GPU memory ranged from 15,788 to 15,798 MiB and temperature from 48 to 78 C.
The container's restart count stayed at one after the deliberately injected
crash; there was no additional restart during this capacity sweep. Clock and
power ranges, including idle periods, remain in the dataset.


## Fault recovery

The recovery driver disconnects after real streamed content, imposes a 50 ms
execution deadline, and completes a follow-up request. It checks one accepted
request, terminal outcome, started worker and release for each running fault.

An isolated acceptance container runs with Docker on-failure supervision and
an init process. After receiving actual content, the driver kills the named
server child inside that container. It observes an increased Docker restart
count, waits for HTTP health and requires fresh inference to finish with
[DONE]. This tests process replacement by Docker; the interrupted request is
not replayed, and the old process's in-memory ledger is not durable.
The crash driver requires the acceptance label and init/restart configuration.

| Check | Observed result |
| --- | --- |
| Disconnect after real content | One client-disconnected terminal and one worker release; idle polling took 51.2 ms after the client returned |
| 50 ms execution deadline | One deadline terminal and one worker release; idle polling took 51.7 ms after the client returned |
| Process crash during SSE | Docker restart count 0 → 1; crash to healthy HTTP 7.483 s; fresh inference completed |
| Current FIFO and decode-first dispatch | Six paired greedy outputs and token usages matched; each mode correctly recorded one HTTP-200 stream interrupted by its deadline |

The current multi-context regression uses three workers and 64-token prefill
chunks. Its complete runs and fault checks observed 32 decodes selected ahead
of a waiting prefill and 10 bounded-wait prefills in decode-first mode.
These counters establish dispatch execution; this regression is not a new
throughput comparison. The historical [dispatch experiment](prefill-decode.md)
retains its separate latency and goodput measurements.

The live regression is available as
[test_dispatch_http.py](../../serving/tests/test_dispatch_http.py). With the
three-worker service running, use --engine-order fifo or decode-first and
an output filename. The decode-first run can compare the FIFO result:

```bash
python3 serving/tests/test_dispatch_http.py --url http://127.0.0.1:8083 \
  --engine-order decode-first --output results/raw/dispatch-decode-first.json \
  --compare results/raw/dispatch-fifo.json
```


The restart mechanism follows Docker's [restart policy](https://docs.docker.com/engine/containers/start-containers-automatically/).
A manual Docker stop is not used as evidence of an automatically recovered
crash. The PID-specific fault is confined to the explicitly named test
container created by the supervise command below.

## Reproduction

Build the text runtime as in the [CUDA setup](../../README.md#cuda-text-generation).
Set GGUF_MODEL to the verified local file, then start an isolated instance:

```bash
export KVHARBOR_NATIVE_NAME=kvharbor-acceptance
bash scripts/native_serving.sh supervise --port 8083 --workers 1 \
  --context 2048 --batch 512 --threads 4 --gpu-layers 99 \
  --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

In another terminal, run fault acceptance first, then the uninterrupted
capacity sweep. Use fresh output directories. The capacity command waits
about half an hour for the offered-arrival windows; it is not a smoke test.

```bash
export KVHARBOR_NATIVE_NAME=kvharbor-acceptance
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
python3 scripts/reliability_bench.py --url http://127.0.0.1:8083 \
  --model Qwen3.8-27B-Q4_K_M --container "$KVHARBOR_NATIVE_NAME" --crash \
  --output-dir results/raw/replay-reliability
python3 scripts/capacity_bench.py --url http://127.0.0.1:8083 \
  --model Qwen3.8-27B-Q4_K_M --rates 0.4 0.8 1.6 --requests 128 --repeats 3 \
  --server-binary build-serving-cuda/kvharbor-server --model-file "$GGUF_MODEL" \
  --output-dir results/raw/replay-capacity
```

Keep logs before stopping and removing this test instance:

```bash
docker logs "$KVHARBOR_NATIVE_NAME" > results/raw/replay-server.log 2>&1
docker stop "$KVHARBOR_NATIVE_NAME"
docker rm "$KVHARBOR_NATIVE_NAME"
```

## Limits

Each arrival rate has 384 measured requests across three runs. The earlier
publication target of 1,000 arrivals **per key scenario per repetition** has
not been met. This bounded run does not establish day-long reliability,
crash durability, multiple-model capacity, global token-weighted admission,
or a throughput gain from decode-first. The multi-context
[dispatch comparison](prefill-decode.md) is a separate experiment.


## Source and artifact integrity

The [source archive](../../results/experiments/serving-acceptance-source.tar.gz)
contains the measured runtime and drivers, validation sources, runtime lock,
fixtures, LICENSE and NOTICE. Each archived file has a SHA-256 in snapshot.json.
Measured runtime files match the fingerprints taken before the sweep. The
dataset records the archive and server binary digests. The current regression
records its own test-source digest, separately from the historical dispatch
measurements.
