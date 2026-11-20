#include <cuda_runtime_api.h>
#include <tensorrt_edgellm/common/trtUtils.h>
#include <tensorrt_edgellm/runtime/llmInferenceRuntime.h>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>
#include <tensorrt_edgellm/runtime/streaming.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "edge_sdk_boundary.h"
#include "kvharbor/serving/runtime.h"

namespace kvharbor::serving {
namespace {
namespace rt = trt_edgellm::rt;

struct CudaStream {
  cudaStream_t value = nullptr;
  ~CudaStream() {
    if (value) cudaStreamDestroy(value);
  }
};

// Declared after the producer future so unwinding cancels before joining it.
struct CancelOnExit {
  std::shared_ptr<rt::StreamChannel> channel;
  ~CancelOnExit() {
    if (channel) channel->cancel();
  }
};

struct FenceOnExit {
  cudaStream_t stream;
  ~FenceOnExit() {
    if (stream) cudaStreamSynchronize(stream);
  }
};

class EdgeRuntime final : public Runtime {
 public:
  explicit EdgeRuntime(RuntimeConfig config) : config_(std::move(config)) {}

  std::optional<Failure> Initialize() {
    plugin_ = trt_edgellm::loadEdgellmPluginLib();
    if (!plugin_)
      return BackendError("could not load TensorRT Edge-LLM plugin");
    auto status =
        cudaStreamCreateWithFlags(&stream_.value, cudaStreamNonBlocking);
    if (status != cudaSuccess) return BackendError(cudaGetErrorString(status));
    rt::DeploymentOptions options{};
    options.engineDir = config_.edge_engine_dir;
    options.checkpointDir = config_.edge_checkpoint_dir;
    engine_ = std::make_unique<rt::LLMInferenceRuntime>(std::move(options),
                                                        stream_.value);
    graph_ = config_.edge_cuda_graph &&
             engine_->captureDecodingCUDAGraph(stream_.value);
    return std::nullopt;
  }

  Json Info() const override {
    return {{"backend", "tensorrt-edge-llm"}, {"sdk_version", "0.12.0"},
            {"model", config_.model_id},      {"context", config_.context},
            {"chat_template", "checkpoint"},  {"cuda_graph", graph_}};
  }

  Result Generate(const Request& request, const Emit& emit) override {
    const auto parsed = ReadRequestInput<TextInput>(request);
    if (const auto* failure = std::get_if<Failure>(&parsed))
      return FailedRequest(*failure);
    const auto& input = *std::get<const TextInput*>(parsed);
    auto result =
        EdgeSdkCall<Result>([&] { return Run(input, request, emit); });
    if (auto* failure = std::get_if<Failure>(&result))
      return FailedRequest(*failure);
    return std::move(std::get<Result>(result));
  }

 private:
  Result Run(const TextInput& input, const Request& request, const Emit& emit) {
    Result result;
    const auto begin = Clock::now();
    auto stopped = [&] {
      return request.cancelled->load(std::memory_order_relaxed) ||
             Clock::now() >= request.deadline;
    };
    if (stopped()) {
      result.outcome = "cancelled";
      return result;
    }
    rt::LLMGenerationRequest native;
    native.maxGenerateLength = input.max_tokens;
    native.topK = 1;
    native.applyChatTemplate = input.apply_chat_template;
    auto& row = native.requests.emplace_back();
    for (const auto& entry : input.messages) {
      auto& message = row.messages.emplace_back();
      message.role = entry.role;
      message.contents.push_back({"text", entry.content});
    }
    const auto counts = engine_->countPromptTokens(native);
    if (counts.size() != 1 || counts.front() < 1 ||
        counts.front() > config_.context - input.max_tokens)
      return FailedRequest(
          InvalidRequest("prompt plus max_tokens exceeds context"));
    auto channel = rt::StreamChannel::create();
    native.streamChannels.push_back(channel);
    rt::LLMGenerationResponse response;
    FenceOnExit fence{stream_.value};
    auto producer = std::async(std::launch::async, [&] {
      return EdgeSdkCall<bool>([&] {
        return engine_->handleRequest(native, response, stream_.value);
      });
    });
    CancelOnExit guard{channel};
    bool cancelled = false;
    for (;;) {
      if (stopped()) {
        cancelled = true;
        break;
      }
      auto chunk = channel->waitPop(std::chrono::milliseconds(5));
      if (!chunk) {
        if (producer.wait_for(std::chrono::milliseconds(0)) !=
            std::future_status::ready)
          continue;
        // The producer may have published its last chunk after waitPop timed
        // out.
        chunk = channel->tryPop();
        if (!chunk) break;
      }
      if (!chunk->text.empty()) {
        if (!result.timings.contains("native_ttft_ms"))
          result.timings["native_ttft_ms"] = ElapsedMs(begin);
        if (!emit(std::move(chunk->text))) {
          cancelled = true;
          break;
        }
      }
      if (chunk->finished) break;
    }
    if (cancelled) channel->cancel();
    auto ok = producer.get();
    guard.channel.reset();
    const auto status = cudaStreamSynchronize(stream_.value);
    fence.stream = nullptr;
    if (status != cudaSuccess)
      return FailedRequest(BackendError(cudaGetErrorString(status)));
    if (auto* failure = std::get_if<Failure>(&ok))
      return FailedRequest(*failure);
    if (cancelled) {
      result.outcome = "cancelled";
      if (response.finishReasons.size() == 1)
        result.details["engine_finish_reason"] =
            rt::finishReasonName(response.finishReasons.front());
      if (response.outputIds.size() == 1)
        result.details["engine_output_tokens"] =
            response.outputIds.front().size();
      return result;
    }
    if (!std::get<bool>(ok) || response.finishReasons.size() != 1 ||
        response.outputIds.size() != 1 || response.inputTokenCounts.size() != 1)
      return FailedRequest(
          BackendError("Edge-LLM did not complete the request"));
    const auto reason = response.finishReasons.front();
    if (reason != rt::FinishReason::kEndId &&
        reason != rt::FinishReason::kLength &&
        reason != rt::FinishReason::kStopWords)
      return FailedRequest(BackendError(rt::finishReasonName(reason)));
    auto prompt = response.inputTokenCounts.front();
    auto generated = response.outputIds.front().size();
    result.usage = {{"prompt_tokens", prompt},
                    {"completion_tokens", generated},
                    {"total_tokens", prompt + generated}};
    if (!response.reusedInputTokenCounts.empty())
      result.details["reused_prompt_tokens"] =
          response.reusedInputTokenCounts.front();
    result.details["finish_reason"] =
        reason == rt::FinishReason::kLength ? "length" : "stop";
    result.timings["generation_ms"] = ElapsedMs(begin);
    return result;
  }

  RuntimeConfig config_;
  // Destruction order keeps the stream and plugin alive until the engine is
  // gone.
  std::unique_ptr<void, trt_edgellm::DlDeleter> plugin_;
  CudaStream stream_;
  std::unique_ptr<rt::LLMInferenceRuntime> engine_;
  bool graph_ = false;
};
}  // namespace

ValueOrFailure<TextWorkerBuilder> LoadEdgeBackend(const RuntimeConfig& config) {
  return TextWorkerBuilder([config]() {
    return EdgeSdkCall<std::unique_ptr<Runtime>>(
        [&]() -> ValueOrFailure<std::unique_ptr<Runtime>> {
          auto worker = std::make_unique<EdgeRuntime>(config);
          if (auto failure = worker->Initialize()) return *failure;
          return std::unique_ptr<Runtime>(std::move(worker));
        });
  });
}
}  // namespace kvharbor::serving
