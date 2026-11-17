# Output reservation and Q4_K warp count

These are two paired experiments on one NVIDIA GeForce RTX 5090 Laptop GPU, using the locked
[27B Q4_K_M GGUF and CUDA runtime](../../configs/runtime/llama.json). The
[results](../../results/experiments/perf-003-gpu.json) include engine and
binary fingerprints, per-block memory measurements, per-request timings,
output digests and the comparison tables. The historical runs predate
the first public Git snapshot.

## Method

The text service used one worker and one sequence, a 2,048-token context,
four CPU threads, a reused context and direct greedy sampling. The
[workload](../../benchmarks/kvharbor_sampling.jsonl) contains four fixed
requests. Each native block used two warm-up requests followed by the
workload once. The three configurations were rotated across six rounds
so each ran in every position twice. Each HTTP comparison ran the two
configurations in opposite orders over two rounds. The workload ran
twice per HTTP block after two warm-ups. The profiler was off for these
latency measurements.

| Mode | Output rows reserved | Warps in the selected fused Q4_K path |
| --- | --- | --- |
| `reference` | llama.cpp default | 4 |
| `one-row` | 1 | 4 |
| `two-warps` | llama.cpp default | 2 |

The one-row mode sets the existing llama.cpp output reservation parameter
while retaining the prompt batch size and requesting a single logits row.
This relies on the current single-sequence greedy path. The separate
[Q4_K patch](../../patches/llama.cpp/perf-003-q4k-warps.patch) selects two
warps for the measured specialization. The profiled shape had
17,408 output rows. The candidate also passed CPU reference
comparisons for that shape and a smaller shape with a tail iteration.

## Results

| Measurement | Reference | Candidate |
| --- | ---: | ---: |
| Native CUDA compute buffer, one-row comparison | 507.02 MiB | 124.02 MiB |
| HTTP completion p50, one-row comparison | 1,635.03 ms | 1,640.03 ms |
| Fused Q4_K kernel time, matched profile runs | 1,163.17 ms | 1,136.38 ms |
| HTTP completion p50, two-warps comparison | 1,635.80 ms | 1,620.51 ms |

The one-row setting saved 383 MiB of the
engine's compute buffer and increased HTTP completion p50 by
5.00 ms. This run did not measure the maximum
context or number of users that could fit with the recovered memory.

Both candidates matched the reference's token sequence on all 24 native
pairs and matched the response and token usage on all 16 HTTP pairs in
their respective comparison. The trace recorded 8,064
calls to the selected kernel in each configuration. The two-warps
end-to-end change varied in direction across the native rounds, so the
patch remains an experiment.

## Reproduce

Complete the [CUDA build](../../README.md#cuda-text-generation), set `GGUF_MODEL`
to the locked weights, and free the GPU before starting. The candidate
engine build needs `patch`. Choose unused output directories.

```bash
python3 scripts/build_mmvq_experiment.py --jobs 2
python3 scripts/perf003_bench.py --model "$GGUF_MODEL" \
  --modes reference one-row two-warps --rounds 6 --repeat 1 --warmup 2 \
  --output-dir results/raw/replay-gpu-native
python3 scripts/perf003_bench.py --http --model "$GGUF_MODEL" \
  --modes reference one-row --rounds 2 --repeat 2 --warmup 2 \
  --output-dir results/raw/replay-gpu-memory
python3 scripts/perf003_bench.py --http --model "$GGUF_MODEL" \
  --modes reference two-warps --rounds 2 --repeat 2 --warmup 2 \
  --output-dir results/raw/replay-gpu-warps
```

Each block in the JSON names its round, position and mode. Match rows by
round and request position. Use `e2e_ms` for HTTP completion latency,
`wall_ms` for the native runtime, and `resources.cuda_compute_mib` for
the allocation. Warm-ups are excluded from the request rows.
