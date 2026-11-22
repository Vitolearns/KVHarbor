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
// A text worker emits UTF-8 deltas, a TTS worker emits PCM16, and ASR returns
// its transcript in Result. The HTTP transport owns SSE framing.
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

struct TextMessage {
  std::string role;
  std::string content;
};

struct TextInput {
  int max_tokens;
  bool apply_chat_template;
  std::vector<TextMessage> messages;
};

struct AsrInput {
  std::string wav;
};

struct TtsInput {
  std::string text;
  double speed;
  int voice;
};

struct TestInput {
  int delay_ms;
  int max_tokens;
  bool inject_failure;
  std::optional<int> chunk_bytes;
};

using RequestInput = std::variant<std::monostate, TextInput, AsrInput, TtsInput,
                                  TestInput, Failure>;

struct Request {
  Task task = Task::kLlm;
  std::string id;
  RequestInput input;
  Clock::time_point received = Clock::now();
  Clock::time_point deadline = received + std::chrono::seconds(60);
  std::shared_ptr<std::atomic<bool>> cancelled =
      std::make_shared<std::atomic<bool>>(false);
};

template <typename Input>
ValueOrFailure<const Input*> ReadRequestInput(const Request& request) {
  if (const auto* input = std::get_if<Input>(&request.input)) return input;
  if (const auto* failure = std::get_if<Failure>(&request.input))
    return *failure;
  return BackendError("worker received an unexpected request input type");
}

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
enum class TextBackend { kLlama, kEdgeLlm, kNone };
std::optional<TextBackend> ParseTextBackend(std::string_view value);

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
  TextBackend text_backend = TextBackend::kLlama;
  std::string edge_engine_dir;
  std::string edge_checkpoint_dir;
  bool model_chat_template = false;
  bool edge_cuda_graph = true;
};

ValueOrFailure<TextInput> ParseTextInput(const Json& body,
                                         const RuntimeConfig& config);
RequestInput ParseTextRequest(const Json& body, const RuntimeConfig& config,
                              bool test_backend);
RequestInput ParseTtsRequest(const Json& body);

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

using TextWorkerBuilder =
    std::function<ValueOrFailure<std::unique_ptr<Runtime>>()>;
ValueOrFailure<TextWorkerBuilder> LoadLlamaBackend(const RuntimeConfig& config);
ValueOrFailure<TextWorkerBuilder> LoadEdgeBackend(const RuntimeConfig& config);

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
struct Waveform {
  int sample_rate;
  std::vector<float> samples;
};

ValueOrFailure<Waveform> ReadPcm16Wav(const std::string& wav);
std::string EncodePcm16(const float* samples, std::size_t count);
double ElapsedMs(Clock::time_point begin, Clock::time_point end = Clock::now());
}  // namespace kvharbor::serving

#endif  // KVHARBOR_SERVING_INCLUDE_KVHARBOR_SERVING_RUNTIME_H_
