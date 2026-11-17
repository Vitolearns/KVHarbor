#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "kvharbor/serving/runtime.h"

using kvharbor::serving::Clock;
using kvharbor::serving::ElapsedMs;
using kvharbor::serving::Factory;
using kvharbor::serving::Failure;
using kvharbor::serving::FindField;
using kvharbor::serving::Json;
using kvharbor::serving::MakeFactory;
using kvharbor::serving::ParseSamplingMode;
using kvharbor::serving::Request;
using kvharbor::serving::Runtime;
using kvharbor::serving::RuntimeConfig;
using kvharbor::serving::StringField;
using kvharbor::serving::Task;

namespace {
int BenchmarkError(std::string_view message) {
  std::cerr << "runtime benchmark: " << message << '\n';
  return 1;
}
}  // namespace

int main(int argc, char** argv) {
  RuntimeConfig config;
  std::string input, output;
  int warmup = 1, repeat = 1;
  bool fake = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--reuse-context") {
      config.reuse_context = true;
      continue;
    }
    if (arg == "--test-backend") {
      fake = true;
      continue;
    }
    int* numeric = nullptr;
    if (arg == "--context") numeric = &config.context;
    if (arg == "--threads") numeric = &config.threads;
    if (arg == "--gpu-layers") numeric = &config.gpu_layers;
    if (arg == "--repeat") numeric = &repeat;
    if (arg == "--warmup") numeric = &warmup;
    if (numeric == nullptr && arg != "--model" && arg != "--model-id" &&
        arg != "--max-output-rows" && arg != "--sampling" && arg != "--input" &&
        arg != "--output")
      return BenchmarkError("unknown option: " + std::string(arg));
    if (++i >= argc) return BenchmarkError("missing option value");
    const char* value = argv[i];
    if (numeric != nullptr) {
      char* end = nullptr;
      errno = 0;
      const long parsed = std::strtol(value, &end, 10);
      if (end == value || errno == ERANGE ||
          parsed < std::numeric_limits<int>::min() ||
          parsed > std::numeric_limits<int>::max())
        return BenchmarkError("invalid integer for " + std::string(arg));
      *numeric = static_cast<int>(parsed);
      continue;
    }
    if (arg == "--model")
      config.model = value;
    else if (arg == "--model-id")
      config.model_id = value;
    else if (arg == "--max-output-rows") {
      const std::string_view rows = value;
      if (rows != "0" && rows != "1")
        return BenchmarkError("--max-output-rows requires 0 or 1");
      config.max_output_rows = rows == "1" ? 1 : 0;
    } else if (arg == "--sampling") {
      const auto mode = ParseSamplingMode(value);
      if (!mode)
        return BenchmarkError(
            "sampling mode must be legacy, chain, direct or backend");
      config.sampling = *mode;
    } else if (arg == "--input")
      input = value;
    else if (arg == "--output")
      output = value;
  }
  if ((!fake && config.model.empty()) || input.empty() || output.empty() ||
      warmup < 0 || repeat < 1)
    return BenchmarkError(
        "requires --model, --input and --output; --reuse-context enables "
        "reuse");
  std::ifstream jobs(input);
  if (!jobs) return BenchmarkError("cannot read workload");
  std::vector<Json> rows;
  for (std::string line; std::getline(jobs, line);) {
    if (line.empty()) continue;
    auto row = Json::parse(line, nullptr, false);
    if (row.is_discarded()) return BenchmarkError("invalid JSON workload row");
    rows.push_back(std::move(row));
  }
  if (rows.empty()) return BenchmarkError("empty workload");
  auto created = MakeFactory(config, fake);
  if (const auto* failure = std::get_if<Failure>(&created))
    return BenchmarkError(failure->message);
  auto worker =
      std::get<std::unique_ptr<Factory>>(created)->CreateWorker(Task::kLlm);
  if (const auto* failure = std::get_if<Failure>(&worker))
    return BenchmarkError(failure->message);
  const auto& runtime = std::get<std::unique_ptr<Runtime>>(worker);
  std::ofstream out(output);
  if (!out) return BenchmarkError("cannot write result");
  auto invoke = [&](const Json& body, std::string id, bool measured) -> int {
    Request r;
    r.body = body;
    r.id = std::move(id);
    r.deadline = Clock::now() + std::chrono::minutes(3);
    auto start = Clock::now();
    double ttft = -1.0;
    std::string text;
    bool invalid_chunk = false;
    auto res = runtime->Generate(r, [&](std::string data) {
      if (ttft < 0) ttft = ElapsedMs(start);
      if (!data.starts_with("data: ")) {
        invalid_chunk = true;
        return false;
      }
      const auto chunk =
          Json::parse(std::string_view(data).substr(6), nullptr, false);
      const auto* choices = FindField(chunk, "choices");
      const auto* delta = choices && choices->is_array() && !choices->empty()
                              ? FindField((*choices)[0], "delta")
                              : nullptr;
      const auto* content = delta ? StringField(*delta, "content") : nullptr;
      if (!content) {
        invalid_chunk = true;
        return false;
      }
      text += *content;
      return true;
    });
    if (invalid_chunk)
      return BenchmarkError("invalid SSE content from runtime");
    if (!res.diagnostic.empty()) return BenchmarkError(res.diagnostic);
    if (res.outcome == "bad_request") return BenchmarkError(res.error);
    Json row = {{"id", r.id},
                {"measured", measured},
                {"outcome", res.outcome},
                {"error", res.error},
                {"runtime", runtime->Info()},
                {"ttft_ms", ttft},
                {"wall_ms", ElapsedMs(start)},
                {"timings", res.timings},
                {"usage", res.usage},
                {"details", res.details},
                {"text", text}};
    out << row.dump() << '\n' << std::flush;
    if (res.outcome != "completed")
      return BenchmarkError("generation did not complete");
    return 0;
  };
  for (int i = 0; i < warmup; ++i)
    if (invoke(rows[i % rows.size()], "warmup-" + std::to_string(i), false))
      return 1;
  for (int run = 0; run < repeat; ++run)
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (invoke(rows[i], std::to_string(run) + "-" + std::to_string(i), true))
        return 1;
  return 0;
}
