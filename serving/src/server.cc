#include <algorithm>
#include <array>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/version.hpp>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "admission.h"
#include "kvharbor/serving/runtime.h"

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using Tcp = net::ip::tcp;
using Error = boost::system::error_code;
using std::chrono_literals::operator""s;
using kvharbor::serving::Admission;
using kvharbor::serving::AsrInput;
using kvharbor::serving::BooleanFieldOr;
using kvharbor::serving::Clock;
using kvharbor::serving::EngineSchedule;
using kvharbor::serving::Factory;
using kvharbor::serving::Failure;
using kvharbor::serving::InvalidRequest;
using kvharbor::serving::Job;
using kvharbor::serving::Json;
using kvharbor::serving::MakeFactory;
using kvharbor::serving::ParseSamplingMode;
using kvharbor::serving::ParseTextRequest;
using kvharbor::serving::ParseTtsRequest;
using kvharbor::serving::Pipe;
using kvharbor::serving::PrepareSlots;
using kvharbor::serving::RuntimeConfig;
using kvharbor::serving::Slot;
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
      TextOption{"--engine-dir", o.runtime.edge_engine_dir},
      TextOption{"--checkpoint-dir", o.runtime.edge_checkpoint_dir},
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
    if (argument == "--model-chat-template" ||
        argument == "--edge-no-cuda-graph") {
      if (argument == "--model-chat-template")
        o.runtime.model_chat_template = true;
      else
        o.runtime.edge_cuda_graph = false;
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
        argument != "--policy" && argument != "--text-backend")
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
    } else if (argument == "--text-backend") {
      auto backend = kvharbor::serving::ParseTextBackend(value);
      if (!backend) return InvalidRequest("unknown text backend");
      o.runtime.text_backend = *backend;
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
  o.runtime.max_output_rows = static_cast<std::uint32_t>(output_rows);
  return o;
}

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

std::string SseContent(const std::string& id, const std::string& text) {
  return "data: " +
         Json{{"id", id},
              {"object", "chat.completion.chunk"},
              {"choices", Json::array({{{"index", 0},
                                        {"delta", {{"content", text}}},
                                        {"finish_reason", nullptr}}})}}
             .dump() +
         "\n\n";
}

std::string Metrics(Json s) {
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
      body = admission.Snapshot().dump();
    else if (req.target() == "/metrics") {
      body = Metrics(admission.Snapshot());
      type = "text/plain; version=0.0.4";
    } else if (req.target() == "/v1/models") {
      auto models = Json::array();
      if (admission.Supports(Task::kLlm))
        models.push_back(
            {{"id", options.runtime.model_id}, {"object", "model"}});
      body = Json{{"object", "list"}, {"data", std::move(models)}}.dump();
    } else {
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
  if (!admission.Supports(task)) {
    co_await reply(*client, http::status::service_unavailable,
                   error_body("task runtime is unavailable"));
    co_return;
  }
  auto job = std::make_shared<Job>();
  job->request.task = task;
  job->request.id = admission.NextId();
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
      if (valid) job->request.input = AsrInput{std::move(req.body())};
    } else {
      const auto body = Json::parse(req.body(), nullptr, false);
      valid = body.is_object();
      if (valid && task == Task::kLlm)
        valid = BooleanFieldOr(body, "stream", false).value_or(false);
      // Keep input errors on the job so admission and terminal accounting
      // still follow the same path as an error from the worker.
      if (valid)
        job->request.input =
            task == Task::kLlm
                ? ParseTextRequest(body, options.runtime, options.fake)
                : ParseTtsRequest(body);
    }
  }
  if (!valid) {
    co_await reply(*client, http::status::bad_request,
                   error_body("invalid body, content type or deadline"));
    co_return;
  }
  if (task == Task::kLlm)
    job->encode_chunk = [id = job->request.id](std::string text) {
      return SseContent(id, text);
    };
  if (!admission.Submit(job)) {
    co_await reply(*client, http::status::too_many_requests,
                   error_body("task queue is full"));
    co_return;
  }
  client->job = job;
  client->deadline.expires_at(job->request.deadline);
  client->deadline.async_wait([client, job, &admission](Error e) {
    if (e || client->closed) return;
    admission.Cancel(job, "deadline");
    if (client->header_sent) client->close();
  });
  // There is no HTTP pipelining on this connection. An EOF/reset (or extra
  // input) cancels the model request independently of blocked response writes.
  stream.socket().async_read_some(
      net::buffer(client->watch),
      [client, job, &admission](Error e, std::size_t) {
        if (!client->closed && e != net::error::operation_aborted) {
          admission.Cancel(job, "client_disconnected");
          client->close();
        }
      });

  while (!client->closed && !job->result && !job->request.cancelled->load() &&
         (task == Task::kAsr || !job->pipe->Front()))
    co_await job->pipe->Wait();
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
    admission.Cancel(job, "client_write_failed");
    co_return;
  }
  while (!client->closed && !job->request.cancelled->load()) {
    if (auto chunk = job->pipe->Front()) {
      stream.expires_after(std::chrono::milliseconds(options.write_ms));
      auto wire = http::make_chunk(net::buffer(*chunk));
      co_await net::async_write(stream, wire,
                                net::redirect_error(net::use_awaitable, ec));
      if (ec) {
        admission.Cancel(job, "client_write_failed");
        co_return;
      }
      job->pipe->Consume();
    } else if (job->result)
      break;
    else
      co_await job->pipe->Wait();
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
  auto slots = PrepareSlots(
      options.workers, *std::get<std::unique_ptr<Factory>>(created));
  if (const auto* failure = std::get_if<Failure>(&slots))
    return ServerError(failure->message);
  net::io_context io(1);
  auto work = net::make_work_guard(io);
  Admission admission(io, options.queue, options.earliest_deadline,
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
    admission.Shutdown();
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
        if (!completed) admission.Cancel(client.job, "internal_error");
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
          admission.Shutdown();
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
  std::cout << Json{{"shutdown", admission.Snapshot()}}.dump() << '\n';
  return 0;
}
