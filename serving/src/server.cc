#include <algorithm>
#include <array>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/version.hpp>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "kvharbor/serving/runtime.h"

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using Tcp = net::ip::tcp;
using Error = boost::system::error_code;
using std::chrono_literals::operator""s;
using kvharbor::serving::BooleanFieldOr;
using kvharbor::serving::Clock;
using kvharbor::serving::ElapsedMs;
using kvharbor::serving::EngineSchedule;
using kvharbor::serving::Factory;
using kvharbor::serving::Failure;
using kvharbor::serving::InvalidRequest;
using kvharbor::serving::Json;
using kvharbor::serving::MakeFactory;
using kvharbor::serving::Name;
using kvharbor::serving::ParseSamplingMode;
using kvharbor::serving::Request;
using kvharbor::serving::Result;
using kvharbor::serving::Runtime;
using kvharbor::serving::RuntimeConfig;
using kvharbor::serving::Task;
using kvharbor::serving::ValueOrFailure;
static_assert(BOOST_VERSION >= 108300,
              "KVHarbor serving requires Boost 1.83 or newer");

namespace {
struct Options {
  RuntimeConfig runtime;
  std::string bind = "127.0.0.1";
  int port = 8082, workers = 1, queue = 8, connections = 128;
  int timeout_ms = 60000, buffer_bytes = 65536, write_ms = 3000;
  bool fake = false, earliest_deadline = false;
};

std::optional<int> ParseInteger(std::string_view value, int lo, int hi) {
  int n = 0;
  auto [end, ec] =
      std::from_chars(value.data(), value.data() + value.size(), n);
  if (ec != std::errc{} || end != value.data() + value.size() || n < lo ||
      n > hi)
    return std::nullopt;
  return n;
}

ValueOrFailure<Options> ParseOptions(int argc, char** argv) {
  Options o;
  struct NumericOption {
    std::string_view name;
    int& destination;
    int min;
    int max;
  };
  int output_rows = 0;
  const std::array numbers{
      NumericOption{"--context", o.runtime.context, 128, 32768},
      NumericOption{"--threads", o.runtime.threads, 1, 128},
      NumericOption{"--gpu-layers", o.runtime.gpu_layers, 0, 999},
      NumericOption{"--batch", o.runtime.batch, 1, 4096},
      NumericOption{"--prefill-chunk", o.runtime.prefill_chunk, 1, 4096},
      NumericOption{"--max-output-rows", output_rows, 0, 1},
      NumericOption{"--port", o.port, 1, 65535},
      NumericOption{"--workers", o.workers, 1, 16},
      NumericOption{"--queue", o.queue, 0, 256},
      NumericOption{"--max-connections", o.connections, 1, 4096},
      NumericOption{"--timeout-ms", o.timeout_ms, 10, 300000},
      NumericOption{"--stream-buffer-bytes", o.buffer_bytes, 1024, 1024 * 1024},
      NumericOption{"--write-timeout-ms", o.write_ms, 10, 30000},
  };
  struct TextOption {
    std::string_view name;
    std::string& destination;
  };
  const std::array strings{
      TextOption{"--model", o.runtime.model},
      TextOption{"--model-id", o.runtime.model_id},
      TextOption{"--asr-model-dir", o.runtime.asr_dir},
      TextOption{"--tts-model-dir", o.runtime.tts_dir},
      TextOption{"--bind", o.bind},
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (argument == "--reuse-context") {
      o.runtime.reuse_context = true;
      continue;
    }
    if (argument == "--test-backend") {
      o.fake = true;
      continue;
    }
    const auto number = std::find_if(numbers.begin(), numbers.end(),
                                     [argument](const NumericOption& entry) {
                                       return entry.name == argument;
                                     });
    const auto text = std::find_if(
        strings.begin(), strings.end(),
        [argument](const TextOption& entry) { return entry.name == argument; });
    if (number == numbers.end() && text == strings.end() &&
        argument != "--engine-schedule" && argument != "--sampling" &&
        argument != "--policy")
      return InvalidRequest("unknown option: " + std::string(argument));
    if (++i >= argc)
      return InvalidRequest("missing value for " + std::string(argument));
    const std::string_view value = argv[i];
    if (number != numbers.end()) {
      const auto n = ParseInteger(value, number->min, number->max);
      if (!n) return InvalidRequest("numeric option out of range");
      number->destination = *n;
    } else if (text != strings.end()) {
      text->destination = value;
    } else if (argument == "--engine-schedule") {
      if (value == "fifo")
        o.runtime.engine_schedule = EngineSchedule::kFifo;
      else if (value == "decode-first")
        o.runtime.engine_schedule = EngineSchedule::kDecodeFirst;
      else if (value == "off")
        o.runtime.engine_schedule = EngineSchedule::kOff;
      else
        return InvalidRequest(
            "engine schedule must be off, fifo or decode-first");
    } else if (argument == "--sampling") {
      const auto mode = ParseSamplingMode(value);
      if (!mode)
        return InvalidRequest(
            "sampling mode must be legacy, chain, direct or backend");
      o.runtime.sampling = *mode;
    } else {
      if (value != "fifo" && value != "deadline")
        return InvalidRequest("policy must be fifo or deadline");
      o.earliest_deadline = value == "deadline";
    }
  }
  if (o.runtime.prefill_chunk > o.runtime.batch)
    return InvalidRequest("prefill chunk cannot exceed batch size");
  if (!o.fake && o.runtime.model.empty())
    return InvalidRequest("--model is required");
  o.runtime.max_output_rows = static_cast<std::uint32_t>(output_rows);
  return o;
}

// The reactor owns the timer. Producers may wait for capacity, and cancellation
// wakes them. Bytes remain charged until the corresponding socket write
// finishes.
class Pipe : public std::enable_shared_from_this<Pipe> {
  net::any_io_executor executor_;
  net::steady_timer signal_;
  std::mutex mutex_;
  std::condition_variable space_;
  std::deque<std::shared_ptr<const std::string>> chunks_;
  std::size_t bytes_ = 0, peak_ = 0, cap_;
  bool stopped_ = false, saturated_ = false;
  std::chrono::milliseconds wait_limit_;

 public:
  Pipe(net::io_context& io, std::size_t cap,
       std::chrono::milliseconds wait_limit)
      : executor_(io.get_executor()),
        signal_(io),
        cap_(cap),
        wait_limit_(wait_limit) {}
  void notify() { signal_.cancel(); }
  net::awaitable<void> wait() {
    signal_.expires_at(Clock::time_point::max());
    Error ec;
    co_await signal_.async_wait(net::redirect_error(net::use_awaitable, ec));
  }
  bool put(std::string chunk, const Request& r) {
    std::unique_lock lock(mutex_);
    if (chunk.empty()) return !stopped_;
    if (chunk.size() > cap_) {
      saturated_ = true;
      return false;
    }
    auto available = [&] {
      return stopped_ || r.cancelled->load() || chunk.size() <= cap_ - bytes_;
    };
    if (!space_.wait_until(lock,
                           std::min(r.deadline, Clock::now() + wait_limit_),
                           available)) {
      saturated_ = Clock::now() < r.deadline;
      return false;
    }
    if (stopped_ || r.cancelled->load() || Clock::now() >= r.deadline)
      return false;
    bool wake = chunks_.empty();
    bytes_ += chunk.size();
    peak_ = std::max(peak_, bytes_);
    chunks_.push_back(std::make_shared<const std::string>(std::move(chunk)));
    lock.unlock();
    if (wake)
      net::post(executor_, [self = shared_from_this()] { self->notify(); });
    return true;
  }
  std::shared_ptr<const std::string> front() {
    std::lock_guard lock(mutex_);
    return chunks_.empty() ? nullptr : chunks_.front();
  }
  void consumed() {
    std::lock_guard lock(mutex_);
    if (chunks_.empty()) return;
    bytes_ -= chunks_.front()->size();
    chunks_.pop_front();
    space_.notify_one();
  }
  void stop() {
    {
      std::lock_guard lock(mutex_);
      stopped_ = true;
      chunks_.clear();
      bytes_ = 0;
    }
    space_.notify_all();
    notify();
  }
  std::size_t peak() {
    std::lock_guard lock(mutex_);
    return peak_;
  }
  bool saturated() {
    std::lock_guard lock(mutex_);
    return saturated_;
  }
};

struct Job {
  Request request;
  std::shared_ptr<Pipe> pipe;
  std::optional<Clock::time_point> started;
  std::optional<Result> result;
  std::string stop_reason;
};

struct Slot {
  Task task;
  std::unique_ptr<Runtime> runtime;
  std::shared_ptr<Job> job;
};

ValueOrFailure<std::vector<Slot>> PrepareSlots(const Options& options,
                                               Factory& factory) {
  std::vector<Slot> slots;
  for (const auto task : {Task::kLlm, Task::kAsr, Task::kTts}) {
    if (!factory.Available(task)) continue;
    const int count = task == Task::kLlm ? options.workers : 1;
    for (int i = 0; i < count; ++i) {
      auto worker = factory.CreateWorker(task);
      if (const auto* failure = std::get_if<Failure>(&worker)) return *failure;
      slots.push_back({task,
                       std::move(std::get<std::unique_ptr<Runtime>>(worker)),
                       nullptr});
    }
  }
  return slots;
}

class Admission {
  net::io_context& io_;
  const Options& options_;
  std::vector<Slot> slots_;
  net::thread_pool pool_;
  std::array<std::deque<std::shared_ptr<Job>>, 3> queues_;
  std::map<std::string, std::uint64_t> outcomes_;
  std::uint64_t accepted_ = 0, started_ = 0, rejected_ = 0, releases_ = 0,
                serial_ = 0;
  std::size_t peak_buffer_ = 0;
  bool draining_ = false;
  Json runtime_info_ = Json::object();

  void complete(const std::shared_ptr<Job>& job, Result result) {
    if (job->result) std::abort();
    if (!job->stop_reason.empty())
      result.outcome = job->stop_reason;
    else if (job->pipe->saturated())
      result.outcome = "backpressure";
    else if (result.outcome == "cancelled" &&
             Clock::now() >= job->request.deadline)
      result.outcome = "deadline";
    ++outcomes_[result.outcome];
    peak_buffer_ = std::max(peak_buffer_, job->pipe->peak());
    auto details = result.details;
    details.erase("token_ids");
    details.erase("text");
    Json record = {
        {"id", job->request.id},
        {"task", Name(job->request.task)},
        {"outcome", result.outcome},
        {"queue_ms", job->started
                         ? ElapsedMs(job->request.received, *job->started)
                         : ElapsedMs(job->request.received)},
        {"backend_finished_ms", ElapsedMs(job->request.received)},
        {"slot_released", bool(job->started)},
        {"peak_buffer_bytes", job->pipe->peak()},
        {"usage", result.usage},
        {"timings", result.timings},
        {"details", details}};
    std::cout << record.dump() << '\n' << std::flush;
    job->result = std::move(result);
    job->pipe->notify();
  }

  void start(std::size_t index, std::shared_ptr<Job> job) {
    auto& slot = slots_[index];
    slot.job = job;
    job->started = Clock::now();
    ++started_;
    job->pipe->notify();
    net::post(pool_, [this, index, job] {
      auto result =
          slots_[index].runtime->Generate(job->request, [&](std::string chunk) {
            return job->pipe->put(std::move(chunk), job->request);
          });
      if (!result.diagnostic.empty())
        std::cerr << "runtime request " << job->request.id << ": "
                  << result.diagnostic << '\n';
      net::post(io_, [this, index, job, result = std::move(result)]() mutable {
        slots_[index].job.reset();
        ++releases_;
        complete(job, std::move(result));
        dispatch();
        if (draining_ && active() == 0 && on_drained) on_drained();
      });
    });
  }

  void dispatch() {
    if (draining_) return;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].job) continue;
      auto& queue = queues_[static_cast<int>(slots_[i].task)];
      while (!queue.empty()) {
        auto it = options_.earliest_deadline
                      ? std::min_element(queue.begin(), queue.end(),
                                         [](const auto& a, const auto& b) {
                                           return a->request.deadline <
                                                  b->request.deadline;
                                         })
                      : queue.begin();
        auto job = std::move(*it);
        queue.erase(it);
        if (Clock::now() >= job->request.deadline) {
          Result res;
          res.outcome = "deadline";
          complete(job, std::move(res));
          continue;
        }
        start(i, std::move(job));
        break;
      }
    }
  }

 public:
  std::function<void()> on_drained;
  Admission(net::io_context& io, const Options& options,
            std::vector<Slot> slots)
      : io_(io),
        options_(options),
        slots_(std::move(slots)),
        pool_(slots_.size()) {
    for (const auto& slot : slots_)
      if (!runtime_info_.contains(Name(slot.task)))
        runtime_info_[Name(slot.task)] = slot.runtime->Info();
  }
  ~Admission() { pool_.join(); }
  bool supports(Task task) const { return runtime_info_.contains(Name(task)); }
  const Json& Info() const { return runtime_info_; }
  std::string next_id() { return "kvh-" + std::to_string(++serial_); }
  std::size_t active() const {
    return std::count_if(slots_.begin(), slots_.end(),
                         [](const auto& s) { return bool(s.job); });
  }
  bool submit(const std::shared_ptr<Job>& job) {
    if (draining_) {
      ++rejected_;
      return false;
    }
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].task == job->request.task && !slots_[i].job) {
        ++accepted_;
        start(i, job);
        return true;
      }
    }
    auto& queue = queues_[static_cast<int>(job->request.task)];
    if (queue.size() >= static_cast<std::size_t>(options_.queue)) {
      ++rejected_;
      return false;
    }
    ++accepted_;
    queue.push_back(job);
    return true;
  }
  void cancel(const std::shared_ptr<Job>& job, std::string reason) {
    if (!job) return;
    if (job->stop_reason.empty()) job->stop_reason = std::move(reason);
    job->request.cancelled->store(true);
    job->pipe->stop();
    if (job->result || job->started) return;
    auto& queue = queues_[static_cast<int>(job->request.task)];
    auto it = std::find(queue.begin(), queue.end(), job);
    if (it != queue.end()) {
      queue.erase(it);
      Result res;
      res.outcome = job->stop_reason;
      complete(job, std::move(res));
    }
  }
  void shutdown() {
    draining_ = true;
    for (auto& q : queues_)
      while (!q.empty()) {
        auto job = q.front();
        cancel(job, "shutdown");
      }
    for (auto& slot : slots_)
      if (slot.job) cancel(slot.job, "shutdown");
    if (active() == 0 && on_drained) on_drained();
  }
  Json snapshot() const {
    Json tasks = Json::object();
    for (auto t : {Task::kLlm, Task::kAsr, Task::kTts}) {
      std::size_t busy = 0, limit = 0;
      for (const auto& s : slots_)
        if (s.task == t) {
          ++limit;
          busy += bool(s.job);
        }
      tasks[Name(t)] = {{"active", busy},
                        {"slots", limit},
                        {"queued", queues_[static_cast<int>(t)].size()}};
    }
    return {{"accepted", accepted_},
            {"started", started_},
            {"rejected", rejected_},
            {"slot_releases", releases_},
            {"active", active()},
            {"outcomes", outcomes_},
            {"peak_buffer_bytes", peak_buffer_},
            {"tasks", tasks},
            {"draining", draining_}};
  }
  std::string metrics() const {
    auto s = snapshot();
    std::string text;
    for (auto key : {"accepted", "started", "rejected", "slot_releases",
                     "active", "peak_buffer_bytes"})
      text += "kvharbor_" + std::string(key) + " " + s.at(key).dump() + "\n";
    for (auto& [key, value] : s.at("outcomes").items())
      text += "kvharbor_backend_outcomes_total{outcome=\"" + key + "\"} " +
              value.dump() + "\n";
    for (auto& [task, value] : s.at("tasks").items())
      for (auto key : {"active", "queued", "slots"})
        text += "kvharbor_task_" + std::string(key) + "{task=\"" + task +
                "\"} " + value.at(key).dump() + "\n";
    return text;
  }
};

struct Client {
  beast::tcp_stream stream;
  net::steady_timer deadline;
  std::shared_ptr<Job> job;
  std::array<char, 1> watch{};
  bool closed = false, header_sent = false;
  Client(Tcp::socket socket, net::io_context& io)
      : stream(std::move(socket)), deadline(io) {}
  void close() {
    if (closed) return;
    closed = true;
    deadline.cancel();
    Error ec;
    stream.socket().close(ec);
  }
};

net::awaitable<void> reply(Client& client, http::status status,
                           std::string body,
                           std::string type = "application/json") {
  http::response<http::string_body> message{status, 11};
  message.set(http::field::content_type, type);
  message.keep_alive(false);
  if (client.job) message.set("X-Request-Id", client.job->request.id);
  message.body() = std::move(body);
  message.prepare_payload();
  client.stream.expires_after(3s);
  Error ec;
  co_await http::async_write(client.stream, message,
                             net::redirect_error(net::use_awaitable, ec));
}

std::string error_body(const std::string& text) {
  return Json{{"error", {{"message", text}}}}.dump();
}

net::awaitable<void> serve(std::shared_ptr<Client> client, net::io_context& io,
                           Admission& admission, const Options& options) {
  auto& stream = client->stream;
  beast::flat_buffer read_buffer(2 * 1024 * 1024 + 16384);
  http::request_parser<http::string_body> parser;
  parser.header_limit(16384);
  parser.body_limit(2 * 1024 * 1024);
  stream.expires_after(5s);
  Error ec;
  co_await http::async_read(stream, read_buffer, parser,
                            net::redirect_error(net::use_awaitable, ec));
  if (ec) {
    if (ec == http::error::body_limit)
      co_await reply(*client, http::status::payload_too_large,
                     error_body("request body exceeds 2 MiB"));
    co_return;
  }
  auto req = parser.release();
  if (req.method() == http::verb::get) {
    std::string body, type = "application/json";
    auto status = http::status::ok;
    if (req.target() == "/health")
      body = Json{{"status", "ok"}, {"runtimes", admission.Info()}}.dump();
    else if (req.target() == "/state")
      body = admission.snapshot().dump();
    else if (req.target() == "/metrics") {
      body = admission.metrics();
      type = "text/plain; version=0.0.4";
    } else if (req.target() == "/v1/models")
      body = Json{{"object", "list"},
                  {"data", Json::array({{{"id", options.runtime.model_id},
                                         {"object", "model"}}})}}
                 .dump();
    else {
      status = http::status::not_found;
      body = error_body("unknown endpoint");
    }
    co_await reply(*client, status, std::move(body), std::move(type));
    co_return;
  }
  Task task = Task::kLlm;
  if (req.target() == "/v1/audio/transcriptions")
    task = Task::kAsr;
  else if (req.target() == "/v1/audio/speech")
    task = Task::kTts;
  else if (req.target() != "/v1/chat/completions" &&
           req.target() != "/v1/completions") {
    co_await reply(*client, http::status::not_found,
                   error_body("unknown endpoint"));
    co_return;
  }
  if (req.method() != http::verb::post) {
    co_await reply(*client, http::status::method_not_allowed,
                   error_body("expected POST"));
    co_return;
  }
  if (!admission.supports(task)) {
    co_await reply(*client, http::status::service_unavailable,
                   error_body("task runtime is unavailable"));
    co_return;
  }
  auto job = std::make_shared<Job>();
  job->request.task = task;
  job->request.id = admission.next_id();
  job->pipe = std::make_shared<Pipe>(
      io, options.buffer_bytes, std::chrono::milliseconds(options.write_ms));
  int timeout = options.timeout_ms;
  bool valid = true;
  const auto timeout_header = req.find("X-Request-Timeout-Ms");
  if (timeout_header != req.end()) {
    const auto value = ParseInteger(
        {timeout_header->value().data(), timeout_header->value().size()}, 10,
        300000);
    valid = value.has_value();
    if (value) timeout = std::min(timeout, *value);
  }
  job->request.deadline =
      job->request.received + std::chrono::milliseconds(timeout);
  if (valid) {
    if (task == Task::kAsr) {
      valid = req[http::field::content_type].starts_with("audio/wav");
      if (valid) job->request.audio = std::move(req.body());
    } else {
      job->request.body = Json::parse(req.body(), nullptr, false);
      valid = job->request.body.is_object();
      if (valid && task == Task::kLlm)
        valid =
            BooleanFieldOr(job->request.body, "stream", false).value_or(false);
    }
  }
  if (!valid) {
    co_await reply(*client, http::status::bad_request,
                   error_body("invalid body, content type or deadline"));
    co_return;
  }
  if (!admission.submit(job)) {
    co_await reply(*client, http::status::too_many_requests,
                   error_body("task queue is full"));
    co_return;
  }
  client->job = job;
  client->deadline.expires_at(job->request.deadline);
  client->deadline.async_wait([client, job, &admission](Error e) {
    if (e || client->closed) return;
    admission.cancel(job, "deadline");
    if (client->header_sent) client->close();
  });
  // There is no HTTP pipelining on this connection. An EOF/reset (or extra
  // input) cancels the model request independently of blocked response writes.
  stream.socket().async_read_some(
      net::buffer(client->watch),
      [client, job, &admission](Error e, std::size_t) {
        if (!client->closed && e != net::error::operation_aborted) {
          admission.cancel(job, "client_disconnected");
          client->close();
        }
      });

  while (!client->closed && !job->result && !job->request.cancelled->load() &&
         (task == Task::kAsr || !job->pipe->front()))
    co_await job->pipe->wait();
  if (client->closed) co_return;
  if (job->request.cancelled->load() ||
      (job->result && job->result->outcome != "completed")) {
    bool timeout = job->stop_reason == "deadline" ||
                   (job->result && job->result->outcome == "deadline");
    auto status = timeout ? http::status::request_timeout
                  : job->result && job->result->outcome == "bad_request"
                      ? http::status::bad_request
                      : http::status::bad_gateway;
    co_await reply(*client, status,
                   error_body(timeout       ? "request deadline exceeded"
                              : job->result ? job->result->error
                                            : "request cancelled"));
    co_return;
  }
  if (task == Task::kAsr) {
    auto body =
        Json{{"id", job->request.id},
             {"text", job->result->details.at("text")},
             {"timings", job->result->timings},
             {"audio_seconds", job->result->details.at("audio_seconds")}}
            .dump();
    co_await reply(*client, http::status::ok, std::move(body));
    co_return;
  }

  http::response<http::empty_body> header{http::status::ok, 11};
  header.keep_alive(false);
  header.chunked(true);
  header.set(http::field::content_type,
             task == Task::kLlm ? "text/event-stream" : "audio/pcm");
  header.set(http::field::cache_control, "no-cache");
  header.set("X-Request-Id", job->request.id);
  if (task == Task::kTts) {
    header.set("X-Audio-Sample-Rate",
               admission.Info().at("tts").at("sample_rate").dump());
    header.set("X-Audio-Encoding", "pcm_s16le_mono");
  }
  http::response_serializer<http::empty_body> serializer(header);
  client->header_sent = true;
  stream.expires_after(std::chrono::milliseconds(options.write_ms));
  co_await http::async_write_header(
      stream, serializer, net::redirect_error(net::use_awaitable, ec));
  if (ec) {
    admission.cancel(job, "client_write_failed");
    co_return;
  }
  while (!client->closed && !job->request.cancelled->load()) {
    if (auto chunk = job->pipe->front()) {
      stream.expires_after(std::chrono::milliseconds(options.write_ms));
      auto wire = http::make_chunk(net::buffer(*chunk));
      co_await net::async_write(stream, wire,
                                net::redirect_error(net::use_awaitable, ec));
      if (ec) {
        admission.cancel(job, "client_write_failed");
        co_return;
      }
      job->pipe->consumed();
    } else if (job->result)
      break;
    else
      co_await job->pipe->wait();
  }
  if (client->closed || job->request.cancelled->load()) co_return;
  if (task == Task::kLlm) {
    std::string tail;
    if (job->result->outcome == "completed") {
      auto finish = job->result->details.value("finish_reason", "stop");
      tail = "data: " +
             Json{{"id", job->request.id},
                  {"choices", Json::array({{{"index", 0},
                                            {"delta", Json::object()},
                                            {"finish_reason", finish}}})},
                  {"usage", job->result->usage}}
                 .dump() +
             "\n\ndata: [DONE]\n\n";
    } else
      tail = "data: " +
             error_body(job->result->error.empty() ? job->result->outcome
                                                   : job->result->error) +
             "\n\n";
    stream.expires_after(std::chrono::milliseconds(options.write_ms));
    auto wire = http::make_chunk(net::buffer(tail));
    co_await net::async_write(stream, wire,
                              net::redirect_error(net::use_awaitable, ec));
    if (ec) co_return;
  } else if (job->result->outcome != "completed")
    co_return;  // Truncated PCM never claims normal completion.
  stream.expires_after(std::chrono::milliseconds(options.write_ms));
  auto last = http::make_chunk_last();
  co_await net::async_write(stream, last,
                            net::redirect_error(net::use_awaitable, ec));
}
int ServerError(std::string_view message) {
  std::cerr << "kvharbor-server: " << message << '\n';
  return 1;
}
}  // namespace

int main(int argc, char** argv) {
  auto parsed = ParseOptions(argc, argv);
  if (const auto* failure = std::get_if<Failure>(&parsed))
    return ServerError(failure->message);
  const auto& options = std::get<Options>(parsed);
  auto created = MakeFactory(options.runtime, options.fake);
  if (const auto* failure = std::get_if<Failure>(&created))
    return ServerError(failure->message);
  auto slots =
      PrepareSlots(options, *std::get<std::unique_ptr<Factory>>(created));
  if (const auto* failure = std::get_if<Failure>(&slots))
    return ServerError(failure->message);
  net::io_context io(1);
  auto work = net::make_work_guard(io);
  Admission admission(io, options,
                      std::move(std::get<std::vector<Slot>>(slots)));
  Error socket_error;
  const auto address = net::ip::make_address(options.bind, socket_error);
  if (socket_error) return ServerError(socket_error.message());
  const Tcp::endpoint endpoint(address,
                               static_cast<unsigned short>(options.port));
  Tcp::acceptor acceptor(io);
  acceptor.open(endpoint.protocol(), socket_error);
  if (!socket_error)
    acceptor.set_option(Tcp::acceptor::reuse_address(true), socket_error);
  if (!socket_error) acceptor.bind(endpoint, socket_error);
  if (!socket_error)
    acceptor.listen(net::socket_base::max_listen_connections, socket_error);
  if (socket_error) return ServerError(socket_error.message());
  using Clients = std::map<Client*, std::shared_ptr<Client>>;
  Clients clients;
  net::signal_set signals(io);
  signals.add(SIGINT, socket_error);
  if (!socket_error) signals.add(SIGTERM, socket_error);
  if (socket_error) return ServerError(socket_error.message());
  admission.on_drained = [&] { work.reset(); };
  signals.async_wait([&](Error e, int) {
    if (e) return;
    Error ignored;
    acceptor.close(ignored);
    admission.shutdown();
    for (auto& [_, c] : clients) c->close();
  });
  auto serve_client =
      [&](std::shared_ptr<Client> client) -> net::awaitable<void> {
    struct Finish {
      Clients& clients;
      Admission& admission;
      Client& client;
      bool completed = false;

      ~Finish() {
        if (!completed) admission.cancel(client.job, "internal_error");
        client.close();
        clients.erase(&client);
      }
    } finish{clients, admission, *client};
    co_await serve(client, io, admission, options);
    finish.completed = true;
  };
  auto accept = [&]() -> net::awaitable<void> {
    while (acceptor.is_open()) {
      Error e;
      auto socket = co_await acceptor.async_accept(
          net::redirect_error(net::use_awaitable, e));
      if (e) {
        if (e != net::error::operation_aborted && acceptor.is_open()) {
          std::cerr << "accept loop failed: " << e.message() << '\n';
          Error ignored;
          signals.cancel(ignored);
          admission.shutdown();
          for (auto& [_, client] : clients) client->close();
        }
        break;
      }
      if (clients.size() >= static_cast<std::size_t>(options.connections))
        continue;
      Error ignored;
      socket.set_option(Tcp::no_delay(true), ignored);
      auto client = std::make_shared<Client>(std::move(socket), io);
      clients[client.get()] = client;
      net::co_spawn(io, serve_client(std::move(client)), net::detached);
    }
  };
  net::co_spawn(io, accept(), net::detached);
  std::cout << Json{{"listening",
                     options.bind + ":" + std::to_string(options.port)},
                    {"runtimes", admission.Info()},
                    {"policy", options.earliest_deadline ? "deadline" : "fifo"},
                    {"queue_per_task", options.queue},
                    {"stream_buffer_bytes", options.buffer_bytes}}
                   .dump()
            << '\n'
            << std::flush;
  io.run();
  std::cout << Json{{"shutdown", admission.snapshot()}}.dump() << '\n';
  return 0;
}
