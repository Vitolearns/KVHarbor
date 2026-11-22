# Single-worker HTTP capacity and overload recovery

This initial sweep used 24 requests per run on an RTX 5090 Laptop GPU with the
locked 27B model. A later sweep uses three 128-request runs per rate.
[Dataset](../../results/experiments/serving-capacity.json) ·
[Runtime lock](../../configs/runtime/llama.json) · [Repeated sweep](serving-acceptance.md)

## Measurements

| Offered req/s | Repeat | SLO-compliant / offered | Goodput req/s | TTFT p95 / p99 (ms) | TPOT p95 (ms) | Completed / 429 / timeout / failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 1 | 24/24 | 0.480 | 728 / 1011 | 28.3 | 24 / 0 / 0 / 0 |
| 0.4 | 2 | 24/24 | 0.588 | 1024 / 1100 | 33.0 | 24 / 0 / 0 / 0 |
| 0.8 | 1 | 24/24 | 0.932 | 1172 / 1331 | 28.9 | 24 / 0 / 0 / 0 |
| 0.8 | 2 | 17/24 | 0.805 | 2759 / 3150 | 29.0 | 24 / 0 / 0 / 0 |
| 1.6 | 1 | 14/24 | 0.781 | 4658 / 5229 | 29.0 | 24 / 0 / 0 / 0 |
| 1.6 | 2 | 7/24 | 0.430 | 5820 / 5861 | 28.9 | 22 / 2 / 0 / 0 |

Both repetitions met 95% SLO compliance at 0.4 offered requests/s. The repeat count and prompt mix define the scope of this result.
Each run has at most 24 completed requests, so p99 coincides with the
maximum. The historical upstream llama-server sweep is documented
[separately](local-validation.md).

## Overload and recovery

Each of two 24-request bursts at 100 requests/s completed 9 requests and
returned 15 HTTP 429 responses. Three subsequent requests per burst completed.
The service returned to zero active requests and its total accepted and
released counts agreed. The recovery field in the JSON measures polling
time after the client has finished observing the burst. This is a bounded
overload test.

<details>
<summary>Setup and replay</summary>

## Setup

One text worker, FIFO admission, eight queue entries, an 8 s server deadline,
2,048 context tokens, a 512-token batch, four CPU threads, reused context,
direct greedy sampling and one output row. The client used three warm-ups and
24 measured requests per run, cycling over the published
[mixed-length workload](../../benchmarks/kvharbor_text_mixed.jsonl). The three
arrival rates were offered as Poisson arrivals with seeds 41 and 42. The
rate order was reversed in the second repeat.

The SLO requires a completed response and client-observed TTFT at most
2,000 ms and TPOT at most 75 ms. Completed responses with one or zero
generated tokens use the TTFT check alone when TPOT is unavailable.
Goodput is SLO-compliant requests divided by the observed run duration.
Percentiles use nearest rank over completed requests with a recorded value.

## Replay

Complete the [CUDA build](../serving.md#cuda-text-generation), then start the
single-worker service in one Bash terminal.

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
bash scripts/native_serving.sh serve --workers 1 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

In another terminal, explicitly select 24 requests and two repeats. The current
driver defaults to 128 and three. It records schema-2 arrival-based SLO results
in `$run/capacity/report.json`. The historical table above keeps its collected
protocol and fingerprints.

```bash
mkdir -p results/raw
run=$(mktemp -d "$PWD/results/raw/replay-serving-capacity.XXXXXX")
python3 scripts/capacity_bench.py --url http://127.0.0.1:8082 \
  --model Qwen3.8-27B-Q4_K_M --rates 0.4 0.8 1.6 --requests 24 --repeats 2 \
  --burst-count 24 --server-binary build-serving-cuda/kvharbor-server \
  --model-file models/Qwen3.8-27B-Q4_K_M.gguf --output-dir "$run/capacity"
```

</details>
