#ifndef KVHARBOR_SERVING_SRC_ADMISSION_H_
#define KVHARBOR_SERVING_SRC_ADMISSION_H_

#include <array>
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/thread_pool.hpp>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "kvharbor/serving/runtime.h"

namespace kvharbor::serving {
namespace net = boost::asio;

// The reactor owns the timer. Producers may wait for capacity, and cancellation
// wakes them. Bytes remain charged until the consumer has written the chunk.
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
       std::chrono::milliseconds wait_limit);
  void Notify();
  net::awaitable<void> Wait();
  bool Put(std::string chunk, const Request& r);
  std::shared_ptr<const std::string> Front();
  void Consume();
  void Stop();
  std::size_t PeakBytes();
  bool Saturated();
};

struct Job {
  Request request;
  std::shared_ptr<Pipe> pipe;
  // Optional transport framing, applied before charging bytes to the buffer.
  std::function<std::string(std::string)> encode_chunk;
  std::optional<Clock::time_point> started;
  std::optional<Result> result;
  std::string stop_reason;
};

struct Slot {
  Task task;
  std::unique_ptr<Runtime> runtime;
  std::shared_ptr<Job> job;
};

ValueOrFailure<std::vector<Slot>> PrepareSlots(int text_workers,
                                               Factory& factory);

// The reactor owns the queues and slot state. Workers post completion only
// after the runtime returns and has finished using its native context.
class Admission {
  net::io_context& io_;
  std::size_t queue_limit_;
  bool earliest_deadline_;
  std::vector<Slot> slots_;
  net::thread_pool pool_;
  std::array<std::deque<std::shared_ptr<Job>>, 3> queues_;
  std::map<std::string, std::uint64_t> outcomes_;
  std::uint64_t accepted_ = 0, started_ = 0, rejected_ = 0, releases_ = 0,
                serial_ = 0;
  std::size_t peak_buffer_ = 0;
  bool draining_ = false;
  Json runtime_info_ = Json::object();

  void Complete(const std::shared_ptr<Job>& job, Result result);
  void Start(std::size_t index, std::shared_ptr<Job> job);
  void Dispatch();

 public:
  std::function<void()> on_drained;
  Admission(net::io_context& io, std::size_t queue_limit,
            bool earliest_deadline, std::vector<Slot> slots);
  ~Admission();
  bool Supports(Task task) const;
  const Json& Info() const;
  std::string NextId();
  std::size_t Active() const;
  bool Submit(const std::shared_ptr<Job>& job);
  void Cancel(const std::shared_ptr<Job>& job, std::string reason);
  void Shutdown();
  Json Snapshot() const;
};

}  // namespace kvharbor::serving

#endif  // KVHARBOR_SERVING_SRC_ADMISSION_H_
