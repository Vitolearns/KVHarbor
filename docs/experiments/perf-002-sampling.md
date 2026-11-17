# Greedy sampling on the 27B decode path

This experiment compared llama.cpp sampler paths and KVHarbor's direct
greedy scan on one NVIDIA GeForce RTX 5090 Laptop GPU. It used the locked
[27B GGUF](../../configs/runtime/llama.json), the
[four-request workload](../../benchmarks/kvharbor_sampling.jsonl) and
the [sampling implementation](../../serving/src/llama_runtime.cc).
[Dataset](../../results/experiments/perf-002-sampling.json)

## Method

The native benchmark evaluated the legacy candidate array, the upstream
sampler chain, KVHarbor's direct scan with AVX2, and the upstream GPU
sampler. It used a 2,048-token context and four CPU threads. Each of
four rounds ran all four modes in a different order, with four warm-up
requests and eight measured requests per mode per round. The native
binary and workload fingerprints are recorded in the dataset.

The HTTP comparison used one text worker, the same context size and
four threads. In each of four rounds it ran the workload once with the
legacy sampler and once with the direct sampler. The order alternated.
The JSON preserves the native and HTTP blocks separately.

## Results

| Measurement | Legacy | Direct AVX2 |
| --- | ---: | ---: |
| Native sampling p50 | 461.42 µs/token | 107.61 µs/token |
| HTTP completion p50 | 1,632.85 ms | 1,623.92 ms |

Native sampling time per token decreased 76.7%.
The HTTP completion p50 decreased 0.55%, with
run-to-run variation. Across the 128 native and 32 HTTP requests,
112 matched comparisons confirmed equivalent output. The Nsight
captures also showed that the tested upstream GPU sampler still
copied the full logits to the host with this engine revision.

## Reproduce

Use the [CUDA build](../../README.md#cuda-text-generation) and the locked
`GGUF_MODEL` weights. Run the native comparison with a fresh output
directory.

```bash
python3 scripts/sampling_bench.py --model "$GGUF_MODEL" \
  --workload benchmarks/kvharbor_sampling.jsonl --rounds 4 --repeat 2 \
  --warmup 4 --context 2048 --threads 4 --output-dir results/raw/replay-sampling
```

For the HTTP half, start the text server with `--workers 1 --threads 4
--context 2048` and either `--sampling legacy` or `--sampling direct`.
Replay four requests per mode using
[stream_bench.py](../../scripts/stream_bench.py) with concurrency 1.
Alternate which mode runs first. Reset each server between blocks.
Output digests and the recorded token counts can be checked without
transcript text. Use the native `timings.sampling_ms` and completion
token count for sampling cost, and `e2e_ms` for HTTP completion.
