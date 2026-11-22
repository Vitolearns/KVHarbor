#include "kvharbor/serving/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace kvharbor::serving {
std::optional<TextBackend> ParseTextBackend(std::string_view value) {
  if (value == "llama.cpp") return TextBackend::kLlama;
  if (value == "tensorrt-edge-llm") return TextBackend::kEdgeLlm;
  if (value == "none") return TextBackend::kNone;
  return std::nullopt;
}

const char* Name(Task task) {
  switch (task) {
    case Task::kLlm:
      return "llm";
    case Task::kAsr:
      return "asr";
    case Task::kTts:
      return "tts";
  }
  std::abort();
}

const char* Name(SamplingMode mode) {
  switch (mode) {
    case SamplingMode::kLegacy:
      return "legacy";
    case SamplingMode::kChain:
      return "chain";
    case SamplingMode::kDirect:
      return "direct";
    case SamplingMode::kBackend:
      return "backend";
  }
  std::abort();
}

std::optional<SamplingMode> ParseSamplingMode(std::string_view value) {
  for (auto mode : {SamplingMode::kLegacy, SamplingMode::kChain,
                    SamplingMode::kDirect, SamplingMode::kBackend})
    if (value == Name(mode)) return mode;
  return std::nullopt;
}

Failure InvalidRequest(std::string message) {
  return {Failure::Kind::kInvalidRequest, std::move(message)};
}

Failure BackendError(std::string message) {
  return {Failure::Kind::kBackend, std::move(message)};
}

Result FailedRequest(Failure failure) {
  Result result;
  if (failure.kind == Failure::Kind::kInvalidRequest) {
    result.outcome = "bad_request";
    result.error = std::move(failure.message);
  } else {
    result.outcome = "failed";
    result.error = "native execution failed";
    result.diagnostic = std::move(failure.message);
  }
  return result;
}

const Json* FindField(const Json& body, const char* name) {
  const auto field = body.find(name);
  return field == body.end() ? nullptr : &*field;
}

std::optional<double> NumberFieldOr(const Json& body, const char* name,
                                    double fallback) {
  const auto* field = FindField(body, name);
  if (field == nullptr) return fallback;
  if (field->is_number()) return field->get<double>();
  return std::nullopt;
}

std::optional<int> IntegerFieldOr(const Json& body, const char* name,
                                  int fallback) {
  const auto* field = FindField(body, name);
  if (field == nullptr) return fallback;
  if (field->is_number_unsigned()) {
    const auto number = field->get<std::uint64_t>();
    if (number > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      return std::nullopt;
    return static_cast<int>(number);
  }
  if (!field->is_number_integer()) return std::nullopt;
  const auto number = field->get<std::int64_t>();
  if (number < std::numeric_limits<int>::min() ||
      number > std::numeric_limits<int>::max())
    return std::nullopt;
  return static_cast<int>(number);
}

std::optional<bool> BooleanFieldOr(const Json& body, const char* name,
                                   bool fallback) {
  const auto* field = FindField(body, name);
  if (field == nullptr) return fallback;
  if (!field->is_boolean()) return std::nullopt;
  return field->get<bool>();
}

const std::string* StringField(const Json& body, const char* name) {
  const auto* field = FindField(body, name);
  return field == nullptr ? nullptr : field->get_ptr<const std::string*>();
}

ValueOrFailure<TextInput> ParseTextInput(const Json& body,
                                         const RuntimeConfig& config) {
  const auto tokens = IntegerFieldOr(body, "max_tokens", 64);
  const auto temperature = NumberFieldOr(body, "temperature", 0.0);
  if (!body.is_object() || !tokens || !temperature)
    return InvalidRequest("invalid request fields");
  if (*tokens < 1 || *tokens >= config.context || *temperature != 0.0)
    return InvalidRequest(
        "native adapter requires temperature=0 and a bounded max_tokens");
  if (body.contains("model")) {
    const auto* model = StringField(body, "model");
    if (!model) return InvalidRequest("invalid request fields");
    if (*model != config.model_id)
      return InvalidRequest("requested model is not loaded");
  }
  TextInput input{*tokens, !body.contains("prompt"), {}};
  if (!input.apply_chat_template) {
    const auto* prompt = StringField(body, "prompt");
    if (!prompt || prompt->empty() || prompt->size() > 1024 * 1024)
      return InvalidRequest("prompt or output budget is invalid");
    input.messages.push_back({"user", *prompt});
    return input;
  }
  const auto* messages = FindField(body, "messages");
  if (!messages || !messages->is_array() || messages->empty() ||
      messages->size() > 64)
    return InvalidRequest("messages must contain 1 to 64 text messages");
  for (const auto& message : *messages) {
    const auto* role = StringField(message, "role");
    const auto* content = StringField(message, "content");
    if (!role || !content) return InvalidRequest("invalid request fields");
    if (*role != "system" && *role != "user" && *role != "assistant")
      return InvalidRequest("only system, user and assistant roles supported");
    input.messages.push_back({*role, *content});
  }
  return input;
}

RequestInput ParseTextRequest(const Json& body, const RuntimeConfig& config,
                              bool test_backend) {
  if (!test_backend) {
    auto parsed = ParseTextInput(body, config);
    if (auto* input = std::get_if<TextInput>(&parsed)) return std::move(*input);
    return std::get<Failure>(std::move(parsed));
  }
  const auto delay = IntegerFieldOr(body, "test_delay_ms", 5);
  const auto tokens = IntegerFieldOr(body, "max_tokens", 8);
  const auto injected_failure = BooleanFieldOr(body, "test_throw", false);
  if (!body.is_object() || !delay || !tokens || !injected_failure)
    return InvalidRequest("invalid request fields");
  if (*delay < 0 || *delay > 2000 || *tokens < 1 || *tokens > 2048)
    return InvalidRequest("invalid test workload");
  return TestInput{*delay, *tokens, *injected_failure,
                   IntegerFieldOr(body, "test_chunk_bytes", 1)};
}

RequestInput ParseTtsRequest(const Json& body) {
  const auto* text = StringField(body, "input");
  const auto speed = NumberFieldOr(body, "speed", 1.0);
  const auto voice = IntegerFieldOr(body, "voice", 2);
  if (!text || !speed || !voice)
    return InvalidRequest("invalid request fields");
  return TtsInput{*text, *speed, *voice};
}

namespace {
std::int32_t GreedyScalar(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i)
    if (logits[i] > logits[best]) best = i;
  return static_cast<std::int32_t>(best);
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) std::int32_t GreedyAvx2(
    std::span<const float> logits) {
  // The reference keeps index zero when its logit is NaN.
  if (std::isnan(logits[0])) return 0;
  auto values = _mm256_set1_ps(logits[0]);
  auto indices = _mm256_setzero_si256();
  const auto lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  std::size_t i = 0;
  for (; i + 8 <= logits.size(); i += 8) {
    const auto input = _mm256_loadu_ps(logits.data() + i);
    const auto greater = _mm256_cmp_ps(input, values, _CMP_GT_OQ);
    values = _mm256_blendv_ps(values, input, greater);
    const auto positions =
        _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(i)), lanes);
    indices =
        _mm256_blendv_epi8(indices, positions, _mm256_castps_si256(greater));
  }
  alignas(32) float maxima[8];
  alignas(32) std::int32_t positions[8];
  _mm256_store_ps(maxima, values);
  _mm256_store_si256(reinterpret_cast<__m256i*>(positions), indices);
  std::int32_t best = 0;
  float maximum = logits[0];
  // Lane order differs from token order: explicitly choose the lowest index
  // on equal maxima, preserving greedy behavior across lanes and tails.
  for (int lane = 0; lane < 8; ++lane) {
    if (maxima[lane] > maximum ||
        (maxima[lane] == maximum && positions[lane] < best)) {
      maximum = maxima[lane];
      best = positions[lane];
    }
  }
  for (; i < logits.size(); ++i)
    if (logits[i] > maximum) {
      maximum = logits[i];
      best = static_cast<std::int32_t>(i);
    }
  return best;
}
#endif

struct GreedyKernel {
  std::int32_t (*select)(std::span<const float>);
  const char* name;
};
const GreedyKernel& GetGreedyKernel() {
  static const auto kernel = [] {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) return GreedyKernel{GreedyAvx2, "avx2"};
#endif
    return GreedyKernel{GreedyScalar, "scalar"};
  }();
  return kernel;
}
}  // namespace

std::optional<std::int32_t> GreedyArgmax(std::span<const float> logits) {
  if (logits.empty() ||
      logits.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    return std::nullopt;
  return GetGreedyKernel().select(logits);
}
const char* GreedyImplementation() { return GetGreedyKernel().name; }

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

ValueOrFailure<std::string> TakeUtf8(std::string& pending, bool final) {
  std::size_t i = 0;
  while (i < pending.size()) {
    unsigned char c = pending[i];
    std::size_t n = c < 0x80                 ? 1
                    : c >= 0xc2 && c <= 0xdf ? 2
                    : c >= 0xe0 && c <= 0xef ? 3
                    : c >= 0xf0 && c <= 0xf4 ? 4
                                             : 0;
    if (!n) return BackendError("invalid UTF-8 model output");
    if (i + n > pending.size()) break;
    for (std::size_t j = 1; j < n; ++j)
      if ((static_cast<unsigned char>(pending[i + j]) & 0xc0) != 0x80)
        return BackendError("invalid UTF-8 continuation");
    if (n >= 3) {
      auto second = static_cast<unsigned char>(pending[i + 1]);
      if ((c == 0xe0 && second < 0xa0) || (c == 0xed && second >= 0xa0) ||
          (c == 0xf0 && second < 0x90) || (c == 0xf4 && second >= 0x90))
        return BackendError("invalid UTF-8 scalar");
    }
    i += n;
  }
  auto text = pending.substr(0, i);
  pending.erase(0, i);
  if (final && !pending.empty()) {
    text += "\xef\xbf\xbd";
    pending.clear();
  }
  return text;
}

namespace {
std::uint16_t u16(const char* p) {
  return static_cast<unsigned char>(p[0]) |
         (static_cast<unsigned char>(p[1]) << 8);
}
std::uint32_t u32(const char* p) {
  return u16(p) | (static_cast<std::uint32_t>(u16(p + 2)) << 16);
}
class FakeRuntime final : public Runtime {
 public:
  explicit FakeRuntime(Task task) : task_(task) {}
  Json Info() const override {
    return {{"backend", "test-double"}, {"task", Name(task_)}};
  }
  Result Generate(const Request& r, const Emit& emit) override {
    Result result;
    const auto parsed = ReadRequestInput<TestInput>(r);
    if (const auto* failure = std::get_if<Failure>(&parsed))
      return FailedRequest(*failure);
    const auto& input = *std::get<const TestInput*>(parsed);
    if (input.inject_failure)
      return FailedRequest(BackendError("injected backend failure"));
    auto started = Clock::now();
    int produced = 0;
    for (int i = 0; i < input.max_tokens; ++i) {
      auto until = Clock::now() + std::chrono::milliseconds(input.delay_ms);
      while (Clock::now() < until && !r.cancelled->load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (r.cancelled->load() || Clock::now() >= r.deadline) {
        result.outcome = "cancelled";
        break;
      }
      const auto chunk_bytes = input.chunk_bytes;
      if (!chunk_bytes)
        return FailedRequest(InvalidRequest("invalid request fields"));
      if (*chunk_bytes < 1 || *chunk_bytes > 32768)
        return FailedRequest(InvalidRequest("invalid test chunk size"));
      if (!emit(std::string(*chunk_bytes, 'x'))) {
        result.outcome = "cancelled";
        break;
      }
      ++produced;
    }
    result.usage = {{"prompt_tokens", 1},
                    {"completion_tokens", produced},
                    {"total_tokens", produced + 1}};
    result.timings = {{"generation_ms", ElapsedMs(started)}};
    result.details = {{"test_double", true}};
    return result;
  }

 private:
  Task task_;
};
class FakeFactory final : public Factory {
 public:
  bool Available(Task task) const override { return task == Task::kLlm; }
  ValueOrFailure<std::unique_ptr<Runtime>> CreateWorker(Task task) override {
    return std::unique_ptr<Runtime>(std::make_unique<FakeRuntime>(task));
  }
};

class NativeFactory final : public Factory {
 public:
  NativeFactory(RuntimeConfig config, TextWorkerBuilder text)
      : config_(std::move(config)), text_(std::move(text)) {}

  bool Available(Task task) const override {
    if (task == Task::kLlm) return static_cast<bool>(text_);
    return task == Task::kAsr ? !config_.asr_dir.empty()
                              : !config_.tts_dir.empty();
  }

  ValueOrFailure<std::unique_ptr<Runtime>> CreateWorker(Task task) override {
    if (!Available(task)) return BackendError("task runtime is unavailable");
    if (task == Task::kLlm) return text_();
    if (task == Task::kAsr) return MakeAsrRuntime(config_);
    return MakeTtsRuntime(config_);
  }

 private:
  RuntimeConfig config_;
  TextWorkerBuilder text_;
};
}  // namespace

ValueOrFailure<Waveform> ReadPcm16Wav(const std::string& wav) {
  if (wav.size() < 12 || wav.compare(0, 4, "RIFF") || wav.compare(8, 4, "WAVE"))
    return InvalidRequest("expected RIFF/WAVE PCM16 mono audio");
  std::size_t declared = static_cast<std::size_t>(u32(wav.data() + 4)) + 8;
  if (declared != wav.size()) return InvalidRequest("WAV RIFF size mismatch");
  bool format = false;
  int sample_rate = 0;
  std::size_t audio_pos = 0, audio_size = 0;
  for (std::size_t pos = 12; pos + 8 <= declared;) {
    std::size_t size = u32(wav.data() + pos + 4);
    pos += 8;
    if (size > declared - pos) return InvalidRequest("truncated WAV chunk");
    if (wav.compare(pos - 8, 4, "fmt ") == 0) {
      if (format || size < 16 || u16(wav.data() + pos) != 1 ||
          u16(wav.data() + pos + 2) != 1 || u16(wav.data() + pos + 14) != 16 ||
          u16(wav.data() + pos + 12) != 2)
        return InvalidRequest("WAV must be PCM16 mono");
      sample_rate = static_cast<int>(u32(wav.data() + pos + 4));
      if (sample_rate < 8000 || sample_rate > 48000 ||
          u32(wav.data() + pos + 8) !=
              static_cast<std::uint32_t>(sample_rate * 2))
        return InvalidRequest("invalid WAV sample rate or byte rate");
      format = true;
    } else if (wav.compare(pos - 8, 4, "data") == 0) {
      if (audio_pos)
        return InvalidRequest("multiple WAV data chunks unsupported");
      audio_pos = pos;
      audio_size = size;
    }
    if (size + (size & 1) > declared - pos)
      return InvalidRequest("truncated WAV padding");
    pos += size + (size & 1);
  }
  if (!format || !audio_pos || !audio_size || (audio_size & 1))
    return InvalidRequest("missing WAV format/data");
  if (audio_size / 2 > static_cast<std::size_t>(sample_rate) * 60)
    return InvalidRequest("audio duration exceeds 60 seconds");
  std::vector<float> samples(audio_size / 2);
  for (std::size_t i = 0; i < samples.size(); ++i)
    samples[i] =
        static_cast<std::int16_t>(u16(wav.data() + audio_pos + 2 * i)) /
        32768.0f;
  return Waveform{sample_rate, std::move(samples)};
}

std::string EncodePcm16(const float* samples, std::size_t count) {
  std::string result(count * 2, '\0');
  for (std::size_t i = 0; i < count; ++i) {
    float s =
        std::isfinite(samples[i]) ? std::clamp(samples[i], -1.0f, 1.0f) : 0.0f;
    auto v = static_cast<std::int16_t>(
        std::lrint(s * (s < 0 ? 32768.0f : 32767.0f)));
    auto u = static_cast<std::uint16_t>(v);
    result[2 * i] = static_cast<char>(u & 255);
    result[2 * i + 1] = static_cast<char>(u >> 8);
  }
  return result;
}

ValueOrFailure<std::unique_ptr<Factory>> MakeFactory(
    const RuntimeConfig& config, bool test_backend) {
  if (test_backend) {
    if (config.text_backend != TextBackend::kLlama)
      return InvalidRequest("--test-backend cannot select a native backend");
    return std::unique_ptr<Factory>(std::make_unique<FakeFactory>());
  }
  ValueOrFailure<TextWorkerBuilder> builder = TextWorkerBuilder{};
  switch (config.text_backend) {
    case TextBackend::kLlama:
      if (config.model.empty())
        return InvalidRequest("llama.cpp requires --model");
      builder = LoadLlamaBackend(config);
      break;
    case TextBackend::kEdgeLlm:
      if (config.edge_engine_dir.empty() || config.edge_checkpoint_dir.empty())
        return InvalidRequest(
            "Edge-LLM requires --engine-dir and --checkpoint-dir");
      builder = LoadEdgeBackend(config);
      break;
    case TextBackend::kNone:
      if (config.asr_dir.empty() && config.tts_dir.empty())
        return InvalidRequest("enable at least one text or speech runtime");
      break;
  }
  if (auto* failure = std::get_if<Failure>(&builder))
    return std::move(*failure);
  return std::unique_ptr<Factory>(std::make_unique<NativeFactory>(
      config, std::move(std::get<TextWorkerBuilder>(builder))));
}
ValueOrFailure<std::unique_ptr<Factory>> MakeLlamaFactory(
    const RuntimeConfig& config) {
  auto native = config;
  native.text_backend = TextBackend::kLlama;
  return MakeFactory(native, false);
}
#ifndef KVHARBOR_SERVING_LLAMA
ValueOrFailure<TextWorkerBuilder> LoadLlamaBackend(const RuntimeConfig&) {
  return BackendError(
      "runtime built without llama.cpp; rebuild with "
      "KVHARBOR_SERVING_LLAMA=ON");
}
#endif
#ifndef KVHARBOR_SERVING_EDGE_LLM
ValueOrFailure<TextWorkerBuilder> LoadEdgeBackend(const RuntimeConfig&) {
  return BackendError("runtime built without TensorRT Edge-LLM");
}
#endif
#ifndef KVHARBOR_SERVING_AUDIO
ValueOrFailure<std::unique_ptr<Runtime>> MakeAsrRuntime(const RuntimeConfig&) {
  return BackendError("ASR runtime not linked");
}
ValueOrFailure<std::unique_ptr<Runtime>> MakeTtsRuntime(const RuntimeConfig&) {
  return BackendError("TTS runtime not linked");
}
#endif
}  // namespace kvharbor::serving
