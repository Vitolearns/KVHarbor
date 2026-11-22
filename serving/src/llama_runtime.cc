#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "engine_schedule.h"
#include "kvharbor/serving/runtime.h"

namespace kvharbor::serving {
namespace {
struct ModelDeleter {
  void operator()(llama_model* p) const {
    if (p) llama_model_free(p);
  }
};
struct ContextDeleter {
  void operator()(llama_context* p) const {
    if (p) llama_free(p);
  }
};
struct SamplerDeleter {
  void operator()(llama_sampler* p) const {
    if (p) llama_sampler_free(p);
  }
};
using Model = std::shared_ptr<llama_model>;
using Context = std::unique_ptr<llama_context, ContextDeleter>;
using Sampler = std::unique_ptr<llama_sampler, SamplerDeleter>;

Sampler MakeGreedyChain() {
  auto params = llama_sampler_chain_default_params();
  params.no_perf = true;
  Sampler chain(llama_sampler_chain_init(params));
  Sampler greedy(llama_sampler_init_greedy());
  if (!chain || !greedy) return nullptr;
  llama_sampler_chain_add(chain.get(), greedy.release());
  return chain;
}

ValueOrFailure<std::string> FormatPrompt(const TextInput& input,
                                         const llama_model* model,
                                         bool model_template) {
  if (!input.apply_chat_template) return input.messages.front().content;
  std::vector<llama_chat_message> view;
  view.reserve(input.messages.size());
  for (const auto& msg : input.messages)
    view.push_back({msg.role.data(), msg.content.data()});
  const char* tmpl =
      model_template ? llama_model_chat_template(model, nullptr) : "chatml";
  if (!tmpl) return BackendError("GGUF has no default chat template");
  std::string result(4096, '\0');
  int n = llama_chat_apply_template(tmpl, view.data(), view.size(), true,
                                    result.data(), result.size());
  if (n <= 0 || n > 1024 * 1024)
    return InvalidRequest("chat template output exceeds limit");
  if (static_cast<std::size_t>(n) > result.size()) {
    result.resize(n);
    int again = llama_chat_apply_template(tmpl, view.data(), view.size(), true,
                                          result.data(), result.size());
    if (again != n) return BackendError("chat template changed on retry");
  }
  result.resize(n);
  if (!model_template) result += "<think>\n\n</think>\n\n";
  return result;
}

ValueOrFailure<std::vector<llama_token>> Tokenize(const llama_vocab* vocab,
                                                  const std::string& prompt,
                                                  int limit) {
  if (prompt.empty() || prompt.size() > 1024 * 1024 || limit < 1)
    return InvalidRequest("prompt or output budget is invalid");
  // One tokenizer pass in the common case; the negative return carries the
  // required size.
  std::vector<llama_token> out(limit);
  auto n = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()),
                          out.data(), limit, true, true);
  if (n <= 0 || n > limit)
    return InvalidRequest("prompt plus max_tokens exceeds context");
  out.resize(n);
  return out;
}

ValueOrFailure<std::string> RenderToken(const llama_vocab* vocab,
                                        llama_token token) {
  std::array<char, 128> small{};
  int n =
      llama_token_to_piece(vocab, token, small.data(), small.size(), 0, false);
  if (n >= 0) return std::string(small.data(), n);
  if (n < -65536) return BackendError("token piece too large");
  std::string piece(-n, '\0');
  n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
  if (n < 0 || static_cast<std::size_t>(n) > piece.size())
    return BackendError("token rendering failed");
  piece.resize(n);
  return piece;
}

class LlamaRuntime final : public Runtime {
  Model model_;
  RuntimeConfig config_;
  std::shared_ptr<EngineDispatch> dispatch_;
  const llama_vocab* vocab_;
  // A bound sampler must outlive every context referencing it (reverse
  // destruction).
  Sampler backend_sampler_;
  Context retained_;
  const std::atomic<bool>* active_cancel_ = nullptr;
  std::size_t contexts_created_ = 0;

  static bool abort(void* data) {
    auto& self = *static_cast<LlamaRuntime*>(data);
    return self.active_cancel_ &&
           self.active_cancel_->load(std::memory_order_relaxed);
  }

  ValueOrFailure<Context> CreateContext() {
    auto params = llama_context_default_params();
    params.n_ctx = config_.context;
    params.n_batch = std::min(config_.batch, config_.context);
    params.n_ubatch = params.n_batch;
    params.n_seq_max = 1;
    // This adapter requests only the last logits row in each decode call.
    params.n_outputs_max = config_.max_output_rows;
    params.n_threads = config_.threads;
    params.n_threads_batch = config_.threads;
    params.abort_callback = abort;
    params.abort_callback_data = this;
    params.no_perf = false;
    llama_sampler_seq_config binding{0, nullptr};
    if (config_.sampling == SamplingMode::kBackend) {
      if (!backend_sampler_) {
        backend_sampler_ = MakeGreedyChain();
        if (!backend_sampler_)
          return BackendError("greedy chain allocation failed");
      }
      binding.sampler = backend_sampler_.get();
      params.samplers = &binding;
      params.n_samplers = 1;
    }
    Context ctx(llama_init_from_model(model_.get(), params));
    if (!ctx ||
        llama_n_ctx_seq(ctx.get()) < static_cast<unsigned>(config_.context))
      return BackendError("llama context allocation failed");
    ++contexts_created_;
    return ctx;
  }

 public:
  LlamaRuntime(Model model, RuntimeConfig config,
               std::shared_ptr<EngineDispatch> dispatch)
      : model_(std::move(model)),
        config_(std::move(config)),
        dispatch_(std::move(dispatch)),
        vocab_(llama_model_get_vocab(model_.get())) {}

  Json Info() const override {
    return {
        {"backend", "llama.cpp"},
        {"chat_template",
         config_.model_chat_template ? "model" : "qwen3-nonthinking"},
        {"model", config_.model_id},
        {"context", config_.context},
        {"threads", config_.threads},
        {"gpu_layers", config_.gpu_layers},
        {"batch", config_.batch},
        {"sequences_per_context", 1},
        {"prefill_chunk", config_.prefill_chunk
                              ? config_.prefill_chunk
                              : std::min(config_.batch, config_.context)},
        {"engine_schedule",
         config_.engine_schedule == EngineSchedule::kOff    ? "off"
         : config_.engine_schedule == EngineSchedule::kFifo ? "fifo"
                                                            : "decode-first"},
        {"context_policy", config_.reuse_context ? "reuse" : "fresh"},
        {"max_output_rows_requested", config_.max_output_rows},
        {"sampling_mode", Name(config_.sampling)},
        {"vocab_size", llama_vocab_n_tokens(vocab_)},
        {"sampling_implementation", config_.sampling == SamplingMode::kDirect
                                        ? GreedyImplementation()
                                        : "llama.cpp"}};
  }

  Result Generate(const Request& r, const Emit& emit) override {
    Result result;
    const auto begin = Clock::now();
    const auto parsed = ReadRequestInput<TextInput>(r);
    if (const auto* failure = std::get_if<Failure>(&parsed))
      return FailedRequest(*failure);
    const auto& input = *std::get<const TextInput*>(parsed);
    const int max_tokens = input.max_tokens;
    auto prompt =
        FormatPrompt(input, model_.get(), config_.model_chat_template);
    if (const auto* failure = std::get_if<Failure>(&prompt))
      return FailedRequest(*failure);
    auto tokenized = Tokenize(vocab_, std::get<std::string>(prompt),
                              config_.context - max_tokens);
    if (const auto* failure = std::get_if<Failure>(&tokenized))
      return FailedRequest(*failure);
    auto& tokens = std::get<std::vector<llama_token>>(tokenized);
    result.timings["template_tokenize_ms"] = ElapsedMs(begin);
    if (r.cancelled->load() || Clock::now() >= r.deadline) {
      result.outcome = "cancelled";
      return result;
    }

    // A worker lease covers context use through native completion. Abort data
    // never outlives r.
    struct CancelScope {
      const std::atomic<bool>*& slot;
      ~CancelScope() { slot = nullptr; }
    } scope{active_cancel_};
    active_cancel_ = r.cancelled.get();
    Context fresh;
    auto phase = Clock::now();
    bool reused = config_.reuse_context && static_cast<bool>(retained_);
    if (!reused) {
      auto created = CreateContext();
      if (const auto* failure = std::get_if<Failure>(&created))
        return FailedRequest(*failure);
      auto& destination = config_.reuse_context ? retained_ : fresh;
      destination = std::move(std::get<Context>(created));
    }
    auto* ctx = config_.reuse_context ? retained_.get() : fresh.get();
    result.timings["context_create_ms"] = reused ? 0.0 : ElapsedMs(phase);
    phase = Clock::now();
    if (reused) {
      // Reset both attention KV and recurrent state; no cross-request prompt
      // cache.
      llama_memory_clear(llama_get_memory(ctx), true);
      llama_synchronize(ctx);
    }
    result.timings["context_reset_ms"] = ElapsedMs(phase);
    result.details = {
        {"context_reused", reused},
        {"contexts_created_by_worker", contexts_created_},
        {"sampling_mode", Name(config_.sampling)},
        {"vocab_size", llama_vocab_n_tokens(vocab_)},
        {"sampling_implementation", config_.sampling == SamplingMode::kDirect
                                        ? GreedyImplementation()
                                        : "llama.cpp"}};
    if (config_.sampling == SamplingMode::kBackend)
      llama_sampler_reset(backend_sampler_.get());
    auto cancelled = [&] {
      return r.cancelled->load(std::memory_order_relaxed) ||
             Clock::now() >= r.deadline;
    };
    EngineDispatch::Stats dispatch_stats;
    auto decode = [&](llama_token* input, int count,
                      EngineDispatch::Phase work) -> std::optional<int> {
      if (cancelled()) return std::nullopt;
      auto on_engine = [&] {
        auto code = llama_decode(ctx, llama_batch_get_one(input, count));
        llama_synchronize(ctx);
        return code;
      };
      int rc;
      if (dispatch_) {
        auto scheduled =
            dispatch_->Run(work, this, dispatch_stats, cancelled, on_engine);
        if (!scheduled) return std::nullopt;
        rc = *scheduled;
      } else {
        rc = on_engine();
      }
      if (cancelled() || rc == 2) return std::nullopt;
      if (rc != 0) {
        // A failed context is discarded instead of being leased to another
        // request.
        retained_.reset();
      }
      return rc;
    };
    auto decode_failure = [](int rc) {
      return FailedRequest(BackendError("llama_decode failed with status " +
                                        std::to_string(rc)));
    };
    phase = Clock::now();
    const std::size_t batch = llama_n_batch(ctx);
    const std::size_t prefill_chunk =
        config_.prefill_chunk
            ? std::min<std::size_t>(config_.prefill_chunk, batch)
            : batch;
    for (std::size_t offset = 0; offset < tokens.size();) {
      auto count = std::min(prefill_chunk, tokens.size() - offset);
      const auto status =
          decode(tokens.data() + offset, static_cast<int>(count),
                 EngineDispatch::Phase::kPrefill);
      if (!status) {
        result.outcome = "cancelled";
        return result;
      }
      if (*status != 0) return decode_failure(*status);
      offset += count;
    }
    result.timings["prefill_ms"] = ElapsedMs(phase);
    Sampler sampler;
    if (config_.sampling == SamplingMode::kLegacy)
      sampler.reset(llama_sampler_init_greedy());
    else if (config_.sampling == SamplingMode::kChain) {
      sampler = MakeGreedyChain();
      if (!sampler)
        return FailedRequest(BackendError("greedy chain allocation failed"));
    }
    if ((config_.sampling == SamplingMode::kLegacy ||
         config_.sampling == SamplingMode::kChain) &&
        !sampler)
      return FailedRequest(BackendError("greedy sampler allocation failed"));
    const auto n_vocab = llama_vocab_n_tokens(vocab_);
    if (n_vocab <= 0)
      return FailedRequest(BackendError("invalid model vocabulary"));
    std::size_t sample_calls = 0, backend_samples = 0;
    std::vector<llama_token> generated;
    std::string pending;
    double sampling_ms = 0.0, decode_ms = 0.0;
    for (int i = 0; i < max_tokens; ++i) {
      if (cancelled()) {
        result.outcome = "cancelled";
        break;
      }
      phase = Clock::now();
      llama_token token;
      if (config_.sampling == SamplingMode::kDirect) {
        const auto* logits = llama_get_logits_ith(ctx, -1);
        if (!logits)
          return FailedRequest(
              BackendError("missing logits for direct greedy sampling"));
        const auto selected =
            GreedyArgmax({logits, static_cast<std::size_t>(n_vocab)});
        if (!selected)
          return FailedRequest(BackendError("invalid greedy logits span"));
        token = *selected;
      } else if (config_.sampling == SamplingMode::kBackend) {
        token = llama_get_sampled_token_ith(ctx, -1);
        if (token == LLAMA_TOKEN_NULL)
          return FailedRequest(BackendError(
              "backend sampling unavailable; explicit mode never silently "
              "falls back"));
        llama_sampler_accept(backend_sampler_.get(), token);
        ++backend_samples;
      } else
        token = llama_sampler_sample(sampler.get(), ctx, -1);
      ++sample_calls;
      sampling_ms += ElapsedMs(phase);
      if (llama_vocab_is_eog(vocab_, token)) break;
      generated.push_back(token);
      auto piece = RenderToken(vocab_, token);
      if (const auto* failure = std::get_if<Failure>(&piece))
        return FailedRequest(*failure);
      pending += std::get<std::string>(piece);
      auto output = TakeUtf8(pending);
      if (const auto* failure = std::get_if<Failure>(&output))
        return FailedRequest(*failure);
      const auto& text = std::get<std::string>(output);
      if (!text.empty()) {
        if (!result.timings.contains("native_ttft_ms"))
          result.timings["native_ttft_ms"] = ElapsedMs(begin);
        if (!emit(text)) {
          result.outcome = "cancelled";
          break;
        }
      }
      if (i + 1 == max_tokens) break;
      phase = Clock::now();
      const auto status = decode(&token, 1, EngineDispatch::Phase::kDecode);
      if (!status) {
        result.outcome = "cancelled";
        break;
      }
      if (*status != 0) return decode_failure(*status);
      decode_ms += ElapsedMs(phase);
    }
    if (result.outcome == "completed") {
      auto output = TakeUtf8(pending, true);
      if (const auto* failure = std::get_if<Failure>(&output))
        return FailedRequest(*failure);
      const auto& tail = std::get<std::string>(output);
      if (!tail.empty() && !emit(tail)) result.outcome = "cancelled";
    }
    result.usage = {{"prompt_tokens", tokens.size()},
                    {"completion_tokens", generated.size()},
                    {"total_tokens", tokens.size() + generated.size()}};
    result.details["token_ids"] = generated;
    result.details["sampling_calls"] = sample_calls;
    result.details["backend_sampled_tokens"] = backend_samples;
    result.details["cpu_candidate_rows_constructed"] =
        (config_.sampling == SamplingMode::kLegacy ||
         config_.sampling == SamplingMode::kChain)
            ? sample_calls * n_vocab
            : 0;
    result.details["finish_reason"] =
        generated.size() == static_cast<std::size_t>(max_tokens) ? "length"
                                                                 : "stop";
    result.timings["sampling_ms"] = sampling_ms;
    result.timings["decode_ms"] = decode_ms;
    result.timings["generation_ms"] = ElapsedMs(begin);
    // Include per-request context destruction in the measured worker occupancy.
    phase = Clock::now();
    fresh.reset();
    result.timings["context_free_ms"] = ElapsedMs(phase);
    if (dispatch_) {
      result.details["engine_schedule"] = {
          {"prefill_calls", dispatch_stats.prefill_calls},
          {"decode_calls", dispatch_stats.decode_calls},
          {"decode_ahead_of_prefill", dispatch_stats.decode_ahead_of_prefill},
          {"forced_prefill", dispatch_stats.forced_prefill},
          {"decode_after_other_prefill",
           dispatch_stats.decode_after_other_prefill},
          {"prefill_after_other_decode",
           dispatch_stats.prefill_after_other_decode},
          {"wait_us", dispatch_stats.wait_us},
      };
    }
    return result;
  }
};

}  // namespace
ValueOrFailure<TextWorkerBuilder> LoadLlamaBackend(
    const RuntimeConfig& config) {
  if (config.context < 128 || config.context > 32768 || config.batch < 1 ||
      config.threads < 1)
    return InvalidRequest("invalid native runtime configuration");
  if (config.prefill_chunk < 0 || config.prefill_chunk > config.batch)
    return InvalidRequest("prefill chunk cannot exceed batch size");
  static std::once_flag backend;
  std::call_once(backend, [] { ggml_backend_load_all(); });
  auto params = llama_model_default_params();
  params.n_gpu_layers = config.gpu_layers;
  params.load_mode = LLAMA_LOAD_MODE_MMAP;
  Model model(llama_model_load_from_file(config.model.c_str(), params),
              ModelDeleter{});
  if (!model || !llama_model_has_decoder(model.get()) ||
      llama_model_has_encoder(model.get()))
    return BackendError("failed to load decoder-only GGUF");
  std::shared_ptr<EngineDispatch> dispatch;
  if (config.engine_schedule != EngineSchedule::kOff)
    dispatch = std::make_shared<EngineDispatch>(config.engine_schedule ==
                                                EngineSchedule::kDecodeFirst);
  return TextWorkerBuilder(
      [model = std::move(model), config,
       dispatch]() -> ValueOrFailure<std::unique_ptr<Runtime>> {
        return std::unique_ptr<Runtime>(
            std::make_unique<LlamaRuntime>(model, config, dispatch));
      });
}
}  // namespace kvharbor::serving
