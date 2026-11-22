#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

namespace {
void Require(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

void FillWeights(ggml_tensor* tensor, std::mt19937& random) {
  const std::int64_t k = tensor->ne[0];
  const std::int64_t rows = tensor->ne[1];
  const std::int64_t distinct_rows = std::min<std::int64_t>(rows, 128);
  const std::size_t row_bytes = ggml_row_size(tensor->type, k);
  std::uniform_real_distribution<float> distribution(-0.04f, 0.04f);
  std::vector<float> input(k * distinct_rows);
  std::generate(input.begin(), input.end(),
                [&] { return distribution(random); });
  std::vector<std::uint8_t> packed(ggml_nbytes(tensor));
  const auto written = ggml_quantize_chunk(
      tensor->type, input.data(), packed.data(), 0, distinct_rows, k, nullptr);
  Require(written == distinct_rows * row_bytes, "unexpected quantized size");
  for (std::int64_t first = distinct_rows; first < rows;
       first += distinct_rows) {
    const auto count = std::min(distinct_rows, rows - first);
    std::copy_n(packed.data(), count * row_bytes,
                packed.data() + first * row_bytes);
  }
  ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
}

std::vector<float> RunShape(ggml_backend_t backend, std::int64_t rows,
                            std::int64_t k, int samples) {
  Require(rows > 0 && rows <= 32768 && k > 0 && k <= 5120 && k % 256 == 0,
          "unsupported sweep shape");
  const ggml_init_params params{1024 * 1024, nullptr, true};
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params),
                                                              ggml_free);
  Require(context != nullptr, "could not allocate tensor metadata");
  auto* ctx = context.get();
  auto* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
  auto* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, rows);
  auto* up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, rows);
  auto* bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, 1);
  auto* up_result = ggml_mul_mat(ctx, up, input);
  auto* gate_result = ggml_mul_mat(ctx, gate, input);
  auto* output = ggml_add(
      ctx, ggml_glu_split(ctx, gate_result, up_result, GGML_GLU_OP_SWIGLU),
      bias);
  auto* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, output);
  std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
      buffer(ggml_backend_alloc_ctx_tensors(ctx, backend),
             ggml_backend_buffer_free);
  Require(buffer != nullptr, "could not allocate device tensors");

  std::mt19937 random(12345);
  FillWeights(gate, random);
  FillWeights(up, random);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<float> values(k);
  std::generate(values.begin(), values.end(),
                [&] { return distribution(random); });
  ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
  values.assign(rows, 0.0f);
  ggml_backend_tensor_set(bias, values.data(), 0, ggml_nbytes(bias));

  const auto compute = [&] {
    Require(
        ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS,
        "graph execution failed");
  };
  const int warmups = samples == 0 ? 1 : 20;
  for (int step = 0; step < warmups; ++step) {
    compute();
  }
  ggml_backend_synchronize(backend);
  constexpr int kIterations = 100;
  for (int sample = 0; sample < samples; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    for (int step = 0; step < kIterations; ++step) {
      compute();
    }
    ggml_backend_synchronize(backend);
    const auto elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin);
    std::printf("%lld,%lld,%d,%.6f\n", static_cast<long long>(rows),
                static_cast<long long>(k), sample,
                elapsed.count() / kIterations);
    std::fflush(stdout);
  }
  ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
  Require(std::all_of(values.begin(), values.end(),
                      [](float v) { return std::isfinite(v); }),
          "non-finite output");
  return values;
}

void CheckShape(ggml_backend_t cuda, std::int64_t rows, std::int64_t columns) {
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
      ggml_backend_cpu_init(), ggml_backend_free);
  Require(cpu != nullptr, "CPU backend initialization failed");
  ggml_backend_cpu_set_n_threads(cpu.get(), 4);
  const auto actual = RunShape(cuda, rows, columns, 0);
  const auto reference = RunShape(cpu.get(), rows, columns, 0);
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double delta = static_cast<double>(actual[i]) - reference[i];
    squared_error += delta * delta;
    squared_reference += static_cast<double>(reference[i]) * reference[i];
  }
  Require(squared_reference > 0.0, "zero reference norm");
  const double nmse = squared_error / squared_reference;
  std::printf("%lld,%lld,%.8g\n", static_cast<long long>(rows),
              static_cast<long long>(columns), nmse);
  Require(nmse <= 1e-4, "normalized mean squared error exceeded 1e-4");
}
}  // namespace

int main(int argc, char** argv) {
  Require(argc == 1 || argc == 4,
          "usage: q4k-shapes [rows columns samples | --check rows columns]");
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
      ggml_backend_cuda_init(0), ggml_backend_free);
  Require(backend != nullptr, "CUDA backend initialization failed");
  if (argc == 4 && std::string_view(argv[1]) == "--check") {
    std::puts("rows,columns,nmse");
    CheckShape(backend.get(), std::atoll(argv[2]), std::atoll(argv[3]));
    return 0;
  }
  std::puts("rows,columns,sample,graph_us");
  if (argc == 4) {
    const int samples = std::atoi(argv[3]);
    Require(samples >= 1 && samples <= 20, "samples must be between 1 and 20");
    RunShape(backend.get(), std::atoll(argv[1]), std::atoll(argv[2]), samples);
    return 0;
  }
  for (const auto& shape : std::vector<std::pair<std::int64_t, std::int64_t>>{
           {256, 2304},
           {257, 2304},
           {1024, 3072},
           {2048, 4096},
           {4096, 4096},
           {5120, 5120},
           {8192, 4096},
           {8192, 5120},
           {12288, 4096},
           {12288, 5120},
           {16384, 4096},
           {16384, 5120},
           {17408, 4096},
           {17408, 5120},
           {20480, 5120},
           {24576, 5120},
           {32768, 5120},
       }) {
    RunShape(backend.get(), shape.first, shape.second, 5);
  }
}
