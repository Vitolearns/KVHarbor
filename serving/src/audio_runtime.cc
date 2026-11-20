#include <sherpa-onnx/c-api/c-api.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "kvharbor/serving/runtime.h"

namespace kvharbor::serving {
namespace {
bool IsStopped(const Request& r) {
  return r.cancelled->load() || Clock::now() >= r.deadline;
}

template <std::size_t N>
ValueOrFailure<std::array<std::string, N>> ModelAssets(
    const std::string& root, const std::array<const char*, N>& names) {
  const std::string prefix =
      root.empty() || root.back() == '/' ? root : root + '/';
  std::array<std::string, N> paths;
  for (std::size_t i = 0; i < N; ++i) {
    paths[i] = prefix + names[i];
    struct stat metadata {};
    if (::stat(paths[i].c_str(), &metadata) != 0)
      return BackendError("missing speech model asset: " + paths[i]);
  }
  return paths;
}

class AsrRuntime final : public Runtime {
  std::string model_id_;
  std::unique_ptr<const SherpaOnnxOfflineRecognizer,
                  decltype(&SherpaOnnxDestroyOfflineRecognizer)>
      engine_{nullptr, SherpaOnnxDestroyOfflineRecognizer};

 public:
  AsrRuntime(const std::array<std::string, 4>& assets, std::string model_id)
      : model_id_(std::move(model_id)) {
    SherpaOnnxOfflineRecognizerConfig config{};
    config.model_config.num_threads = 2;
    config.model_config.provider = "cpu";
    auto& q = config.model_config.qwen3_asr;
    q.conv_frontend = assets[0].c_str();
    q.encoder = assets[1].c_str();
    q.decoder = assets[2].c_str();
    q.tokenizer = assets[3].c_str();
    q.max_total_len = 512;
    q.max_new_tokens = 128;
    q.temperature = 1e-6f;
    q.top_p = 0.8f;
    q.seed = 42;
    engine_.reset(SherpaOnnxCreateOfflineRecognizer(&config));
  }
  bool IsReady() const { return engine_ != nullptr; }
  Json Info() const override {
    return {{"backend", "sherpa-onnx"}, {"model", model_id_},
            {"device", "cpu"},          {"threads", 2},
            {"streaming", false},       {"cancel_mode", "after_native_decode"}};
  }
  Result Generate(const Request& r, const Emit&) override {
    Result result;
    auto phase = Clock::now();
    const auto parsed = ReadRequestInput<AsrInput>(r);
    if (const auto* failure = std::get_if<Failure>(&parsed))
      return FailedRequest(*failure);
    auto decoded = ReadPcm16Wav(std::get<const AsrInput*>(parsed)->wav);
    if (const auto* failure = std::get_if<Failure>(&decoded))
      return FailedRequest(*failure);
    const auto& [rate, audio] = std::get<Waveform>(decoded);
    const double seconds = static_cast<double>(audio.size()) / rate;
    if (seconds > 30)
      return FailedRequest(InvalidRequest(
          "ASR currently accepts at most 30 seconds of PCM16 mono WAV"));
    result.timings["wav_parse_ms"] = ElapsedMs(phase);
    if (IsStopped(r)) {
      result.outcome = "cancelled";
      return result;
    }
    using Stream = std::unique_ptr<const SherpaOnnxOfflineStream,
                                   decltype(&SherpaOnnxDestroyOfflineStream)>;
    Stream stream(SherpaOnnxCreateOfflineStream(engine_.get()),
                  SherpaOnnxDestroyOfflineStream);
    if (!stream)
      return FailedRequest(BackendError("failed to allocate ASR stream"));
    phase = Clock::now();
    SherpaOnnxAcceptWaveformOffline(stream.get(), rate, audio.data(),
                                    static_cast<int32_t>(audio.size()));
    result.timings["accept_waveform_ms"] = ElapsedMs(phase);
    if (IsStopped(r)) {
      result.outcome = "cancelled";
      return result;
    }
    phase = Clock::now();
    // The offline C API is blocking. Admission retains this worker lease
    // even if the HTTP request times out before native execution returns.
    SherpaOnnxDecodeOfflineStream(engine_.get(), stream.get());
    const double decode = ElapsedMs(phase);
    result.timings["decode_ms"] = decode;
    result.details = {{"audio_seconds", seconds},
                      {"rtf", decode / 1000 / seconds},
                      {"sample_rate", rate}};
    if (IsStopped(r)) {
      result.outcome = "cancelled";
      return result;
    }
    using Output =
        std::unique_ptr<const SherpaOnnxOfflineRecognizerResult,
                        decltype(&SherpaOnnxDestroyOfflineRecognizerResult)>;
    Output output(SherpaOnnxGetOfflineStreamResult(stream.get()),
                  SherpaOnnxDestroyOfflineRecognizerResult);
    if (!output || !output->text)
      return FailedRequest(BackendError("ASR returned no result"));
    result.details["text"] = output->text;
    return result;
  }
};

class TtsRuntime final : public Runtime {
  std::unique_ptr<const SherpaOnnxOfflineTts,
                  decltype(&SherpaOnnxDestroyOfflineTts)>
      engine_{nullptr, SherpaOnnxDestroyOfflineTts};
  int sample_rate_ = 0, speakers_ = 0;
  struct Callback {
    const Request& request;
    const Emit& emit;
    Clock::time_point start;
    std::size_t samples = 0;
    double first_ms = -1;
    bool failed = false;
    int rate = 0;
    static int32_t Deliver(const float* samples, int32_t count, float,
                           void* user) noexcept {
      auto& self = *static_cast<Callback*>(user);
      if (IsStopped(self.request)) return 0;
      if (!samples || count < 0 ||
          self.samples + static_cast<std::size_t>(count) >
              static_cast<std::size_t>(self.rate) * 60) {
        self.failed = true;
        return 0;
      }
      for (int32_t i = 0; i < count;) {
        const int32_t n = std::min<int32_t>(4096, count - i);
        if (!self.emit(EncodePcm16(samples + i, n))) return 0;
        if (self.first_ms < 0) self.first_ms = ElapsedMs(self.start);
        self.samples += n;
        i += n;
      }
      return 1;
    }
  };

 public:
  explicit TtsRuntime(const std::array<std::string, 8>& assets) {
    const auto lexicon = assets[3] + "," + assets[4];
    const auto rules = assets[5] + "," + assets[6] + "," + assets[7];
    SherpaOnnxOfflineTtsConfig config{};
    config.model.num_threads = 2;
    config.model.provider = "cpu";
    config.model.kokoro.model = assets[0].c_str();
    config.model.kokoro.voices = assets[1].c_str();
    config.model.kokoro.tokens = assets[2].c_str();
    config.model.kokoro.lexicon = lexicon.c_str();
    config.model.kokoro.length_scale = 1.0f;
    config.rule_fsts = rules.c_str();
    config.max_num_sentences = 1;
    config.silence_scale = 0.2f;
    engine_.reset(SherpaOnnxCreateOfflineTts(&config));
    if (!engine_) return;
    sample_rate_ = SherpaOnnxOfflineTtsSampleRate(engine_.get());
    speakers_ = SherpaOnnxOfflineTtsNumSpeakers(engine_.get());
  }
  std::optional<Failure> InitializationFailure() const {
    if (!engine_) return BackendError("failed to initialize Kokoro");
    if (sample_rate_ <= 0 || speakers_ <= 0)
      return BackendError("invalid TTS model metadata");
    return std::nullopt;
  }
  Json Info() const override {
    return {{"backend", "sherpa-onnx"},
            {"model", "Kokoro-v1.0-zh-en"},
            {"device", "cpu"},
            {"threads", 2},
            {"sample_rate", sample_rate_},
            {"speakers", speakers_},
            {"delivery", "sentence_callback_pcm"},
            {"model_native_streaming", false}};
  }
  Result Generate(const Request& r, const Emit& emit) override {
    const auto parsed = ReadRequestInput<TtsInput>(r);
    if (const auto* failure = std::get_if<Failure>(&parsed))
      return FailedRequest(*failure);
    const auto& [text, speed, voice] = *std::get<const TtsInput*>(parsed);
    if (text.empty() || text.size() > 600 || !std::isfinite(speed) ||
        speed < 0.6 || speed > 1.6 || voice < 0 || voice >= speakers_)
      return FailedRequest(InvalidRequest("invalid TTS text, speed or voice"));
    Result result;
    if (IsStopped(r)) {
      result.outcome = "cancelled";
      return result;
    }
    auto begin = Clock::now();
    Callback cb{r, emit, begin, 0, -1, false, sample_rate_};
    SherpaOnnxGenerationConfig config{};
    config.sid = voice;
    config.speed = static_cast<float>(speed);
    config.silence_scale = 0.2f;
    using Audio =
        std::unique_ptr<const SherpaOnnxGeneratedAudio,
                        decltype(&SherpaOnnxDestroyOfflineTtsGeneratedAudio)>;
    Audio audio(
        SherpaOnnxOfflineTtsGenerateWithConfig(engine_.get(), text.c_str(),
                                               &config, Callback::Deliver, &cb),
        SherpaOnnxDestroyOfflineTtsGeneratedAudio);
    if (cb.failed)
      return FailedRequest(BackendError("TTS returned invalid waveform"));
    if (IsStopped(r)) {
      result.outcome = "cancelled";
      return result;
    }
    if (!audio || audio->n <= 0 || audio->sample_rate != sample_rate_)
      return FailedRequest(BackendError("TTS generated no valid audio"));
    if (cb.samples == 0 &&
        !Callback::Deliver(audio->samples, audio->n, 1.0f, &cb)) {
      result.outcome = "cancelled";
      return result;
    }
    result.timings = {{"generation_ms", ElapsedMs(begin)},
                      {"first_pcm_ms", cb.first_ms}};
    result.details = {
        {"audio_seconds", static_cast<double>(cb.samples) / sample_rate_},
        {"samples_sent", cb.samples},
        {"sample_rate", sample_rate_}};
    if (cb.samples != static_cast<std::size_t>(audio->n))
      result.outcome = "cancelled";
    return result;
  }
};
}  // namespace
ValueOrFailure<std::unique_ptr<Runtime>> MakeAsrRuntime(
    const RuntimeConfig& c) {
  auto assets = ModelAssets(
      c.asr_dir, std::array{"conv_frontend.onnx", "encoder.int8.onnx",
                            "decoder.int8.onnx", "tokenizer"});
  if (const auto* failure = std::get_if<Failure>(&assets)) return *failure;
  std::string model_id = "Qwen3-ASR";
  std::ifstream marker(c.asr_dir + "/.kvharbor-asset.json");
  if (marker) {
    const auto metadata = Json::parse(marker, nullptr, false);
    const auto* id = StringField(metadata, "id");
    if (!id || id->empty()) return BackendError("invalid ASR asset marker");
    model_id = *id;
  }
  auto runtime =
      std::make_unique<AsrRuntime>(std::get<0>(assets), std::move(model_id));
  if (!runtime->IsReady())
    return BackendError("failed to initialize Qwen3-ASR");
  return std::unique_ptr<Runtime>(std::move(runtime));
}
ValueOrFailure<std::unique_ptr<Runtime>> MakeTtsRuntime(
    const RuntimeConfig& c) {
  auto assets = ModelAssets(
      c.tts_dir, std::array{"model.onnx", "voices.bin", "tokens.txt",
                            "lexicon-us-en.txt", "lexicon-zh.txt",
                            "date-zh.fst", "number-zh.fst", "phone-zh.fst"});
  if (const auto* failure = std::get_if<Failure>(&assets)) return *failure;
  auto runtime = std::make_unique<TtsRuntime>(std::get<0>(assets));
  if (auto failure = runtime->InitializationFailure())
    return std::move(*failure);
  return std::unique_ptr<Runtime>(std::move(runtime));
}
}  // namespace kvharbor::serving
