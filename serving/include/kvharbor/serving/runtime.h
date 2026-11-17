#ifndef KVHARBOR_SERVING_INCLUDE_KVHARBOR_SERVING_RUNTIME_H_
#define KVHARBOR_SERVING_INCLUDE_KVHARBOR_SERVING_RUNTIME_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kvharbor::serving {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Emit = std::function<bool(std::string)>;

enum class Task { kLlm, kAsr, kTts };
const char* Name(Task task);

// The legacy sampler materializes candidates per token. The chain sampler
// reuses its buffer. direct scans borrowed logits; backend uses the pinned
// engine's sampler graph.
enum class SamplingMode { kLegacy, kChain, kDirect, kBackend };
const char* Name(SamplingMode mode);
std::optional<SamplingMode> ParseSamplingMode(std::string_view value);
// Match the reference greedy sampler: first maximum, including its NaN
// behavior.
std::optional<std::int32_t> GreedyArgmax(std::span<const float> logits);
const char* GreedyImplementation();

struct Failure {
  enum class Kind { kInvalidRequest, kBackend };

  Kind kind;
  std::string message;
};

template <typename T>
using ValueOrFailure = std::variant<T, Failure>;

Failure InvalidRequest(std::string message);
Failure BackendError(std::string message);

const Json* FindField(const Json& body, const char* name);
std::optional<int> IntegerFieldOr(const Json& body, const char* name,
                                  int fallback);
std::optional<double> NumberFieldOr(const Json& body, const char* name,
                                    double fallback);
std::optional<bool> BooleanFieldOr(const Json& body, const char* name,
                                   bool fallback);
const std::string* StringField(const Json& body, const char* name);

struct Request {
  Task task = Task::kLlm;
  std::string id;
  Json body;
  std::string audio;
  Clock::time_point received = Clock::now();
  Clock::time_point deadline = received + std::chrono::seconds(60);
  std::shared_ptr<std::atomic<bool>> cancelled =
      std::make_shared<std::atomic<bool>>(false);
};

struct Result {
  std::string outcome = "completed";
  std::string error;
  std::string diagnostic;
  Json usage = Json::object();
  Json timings = Json::object();
  Json details = Json::object();
};

Result FailedRequest(Failure failure);

enum class EngineSchedule { kOff, kFifo, kDecodeFirst };

struct RuntimeConfig {
  std::string model;
  std::string model_id = "Qwen3.8-27B-Q4_K_M";
  int context = 2048;
  int threads = 4;
  int gpu_layers = 99;
  int batch = 512;
  std::uint32_t max_output_rows = 0;
  bool reuse_context = false;
  SamplingMode sampling = SamplingMode::kDirect;
  std::string asr_dir;
  std::string tts_dir;
  EngineSchedule engine_schedule = EngineSchedule::kOff;
  int prefill_chunk = 0;
};

class Runtime {
 public:
  virtual ~Runtime() = default;
  virtual Result Generate(const Request& request, const Emit& emit) = 0;
  virtual Json Info() const = 0;
};

class Factory {
 public:
  virtual ~Factory() = default;
  virtual ValueOrFailure<std::unique_ptr<Runtime>> CreateWorker(Task task) = 0;
  virtual bool Available(Task task) const = 0;
};

ValueOrFailure<std::unique_ptr<Factory>> MakeFactory(
    const RuntimeConfig& config, bool test_backend);
ValueOrFailure<std::unique_ptr<Factory>> MakeLlamaFactory(
    const RuntimeConfig& config);
ValueOrFailure<std::unique_ptr<Runtime>> MakeAsrRuntime(
    const RuntimeConfig& config);
ValueOrFailure<std::unique_ptr<Runtime>> MakeTtsRuntime(
    const RuntimeConfig& config);

// Preserves partial UTF-8 bytes across token boundaries; invalid complete input
// fails.
ValueOrFailure<std::string> TakeUtf8(std::string& pending, bool final = false);
std::string SseContent(const std::string& id, const std::string& text);
struct Waveform {
  int sample_rate;
  std::vector<float> samples;
};

ValueOrFailure<Waveform> ReadPcm16Wav(const std::string& wav);
std::string EncodePcm16(const float* samples, std::size_t count);
double ElapsedMs(Clock::time_point begin, Clock::time_point end = Clock::now());
}  // namespace kvharbor::serving

#endif  // KVHARBOR_SERVING_INCLUDE_KVHARBOR_SERVING_RUNTIME_H_
