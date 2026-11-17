# Single-worker HTTP capacity and overload recovery

The [request-level dataset](../../results/experiments/serving-capacity.json)
was collected from KVHarbor's CUDA HTTP/SSE service on a single RTX 5090
Laptop GPU. The [model and llama.cpp revision](../../configs/runtime/llama.json)
were locked, the weight checksum was verified, and the dataset records the
serving binary and source file digests used for this run.

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

## Replay

Complete the [CUDA build](../../README.md#cuda-text-generation), set MODEL_ID to the
locked model ID, and run the service in one Bash terminal.

```bash
bash scripts/native_serving.sh serve --workers 1 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

In another terminal, set BASE_URL to the running service and use a new output
directory.

```bash
python3 scripts/capacity_bench.py --url "$BASE_URL" --model "$MODEL_ID" \
  --output-dir results/raw/replay-serving-capacity
```
