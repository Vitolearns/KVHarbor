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

## Shape-aware Q4_K dispatch on SM120

The optional build adds `KVHARBOR_CUDA_Q4K_WARPS=auto`. For single-vector
Q4_K SwiGLU on SM120 it uses two warps for weight shapes with 8,192 output
rows and reduction width 4,096 or 5,120, or with 12,288 output rows and
width 4,096. All other calls keep the upstream launch policy. The
environment variable still accepts `2` for the original forced experiment.

The synthetic sweep used two Q4_K weight matrices and the actual GGML CUDA
graph for the gated FFN. Each run warmed the graph 20 times, then timed
five blocks of 100 evaluations per shape. Processes used warp settings in the
order 4, 2, 2, 4, 4, 2.

| Output rows × reduction width | Median of block medians, 4 warps → 2 warps | Paired changes |
| --- | ---: | ---: |
| 8,192 × 4,096 | 40.19 → 37.13 µs | −5.8%, −6.6%, −7.6% |
| 8,192 × 5,120 | 45.47 → 42.71 µs | −3.3%, −7.6%, −5.4% |
| 12,288 × 4,096 | 53.15 → 50.09 µs | −7.1%, −3.7%, −7.0% |
| 17,408 × 5,120 | 142.90 → 142.37 µs | +1.3%, −2.1%, −0.7% |

The timings measure synchronized graph evaluations. Each weight matrix
repeats a fixed set of 128 seeded rows. Nsight Systems confirmed two warps
for the three selected shapes and four for three fallback cases. All six
CUDA results passed the CPU comparison with normalized mean squared error
below `1e-4`.

On the locked 27B model, all 16 native token-sequence pairs and all 16 HTTP
output pairs matched. HTTP TPOT P50 was 26.85 ms with the upstream policy and
27.01 ms with automatic dispatch. The profiled fused Q4_K calls had 17,408
or 5,120 output rows and both stayed at four warps. The measured
improvement applies to the selected synthetic shapes.

Build the variant and compare it with the upstream launch policy using the
same engine library in both arms.

```bash
python3 scripts/build_mmvq_experiment.py --shape-dispatch --jobs 4
python3 scripts/perf003_bench.py --model "$GGUF_MODEL" \
  --modes reference shape-dispatch --rounds 4 --repeat 1 --warmup 2 \
  --output-dir results/raw/q4k-shape-native
python3 scripts/perf003_bench.py --http --model "$GGUF_MODEL" \
  --modes reference shape-dispatch --rounds 4 --repeat 1 --warmup 2 \
  --output-dir results/raw/q4k-shape-http
```

Compile [the sweep driver](../../benchmarks/q4k_shapes.cpp) against the
GGML headers and libraries from that build. In the pinned CUDA container,
mount the patched llama.cpp source at `/llama` and the experiment build at
`/build`. Run the driver in a fresh CUDA process per warp setting.

```bash
g++ -std=c++17 -O2 -I/llama/ggml/include q4k_shapes.cpp -L/build/bin \
  -Wl,--allow-shlib-undefined -lggml -lggml-base -lggml-cpu -lggml-cuda -o q4k-shapes
export LD_LIBRARY_PATH=/build/bin GGML_CUDA_DISABLE_GRAPHS=1
KVHARBOR_CUDA_Q4K_WARPS=4 ./q4k-shapes
KVHARBOR_CUDA_Q4K_WARPS=2 ./q4k-shapes
KVHARBOR_CUDA_Q4K_WARPS=auto ./q4k-shapes --check 8192 4096
```
