#include "admission.h"

#include <algorithm>
#include <boost/asio/post.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/system/error_code.hpp>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <variant>

namespace kvharbor::serving {

Pipe::Pipe(net::io_context& io, std::size_t cap,
           std::chrono::milliseconds wait_limit)
    : executor_(io.get_executor()),
      signal_(io),
      cap_(cap),
      wait_limit_(wait_limit) {}

void Pipe::Notify() { signal_.cancel(); }

net::awaitable<void> Pipe::Wait() {
  signal_.expires_at(Clock::time_point::max());
  boost::system::error_code ec;
  co_await signal_.async_wait(net::redirect_error(net::use_awaitable, ec));
}

bool Pipe::Put(std::string chunk, const Request& r) {
  std::unique_lock lock(mutex_);
  if (chunk.empty()) return !stopped_;
  if (chunk.size() > cap_) {
    saturated_ = true;
    return false;
  }
  auto available = [&] {
    return stopped_ || r.cancelled->load() || chunk.size() <= cap_ - bytes_;
  };
  if (!space_.wait_until(lock, std::min(r.deadline, Clock::now() + wait_limit_),
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
    net::post(executor_, [self = shared_from_this()] { self->Notify(); });
  return true;
}

std::shared_ptr<const std::string> Pipe::Front() {
  std::lock_guard lock(mutex_);
  return chunks_.empty() ? nullptr : chunks_.front();
}

void Pipe::Consume() {
  std::lock_guard lock(mutex_);
  if (chunks_.empty()) return;
  bytes_ -= chunks_.front()->size();
  chunks_.pop_front();
  space_.notify_one();
}

void Pipe::Stop() {
  {
    std::lock_guard lock(mutex_);
    stopped_ = true;
    chunks_.clear();
    bytes_ = 0;
  }
  space_.notify_all();
  Notify();
}

std::size_t Pipe::PeakBytes() {
  std::lock_guard lock(mutex_);
  return peak_;
}

bool Pipe::Saturated() {
  std::lock_guard lock(mutex_);
  return saturated_;
}

ValueOrFailure<std::vector<Slot>> PrepareSlots(int text_workers,
                                               Factory& factory) {
  std::vector<Slot> slots;
  for (const auto task : {Task::kLlm, Task::kAsr, Task::kTts}) {
    if (!factory.Available(task)) continue;
    const int count = task == Task::kLlm ? text_workers : 1;
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

Admission::Admission(net::io_context& io, std::size_t queue_limit,
                     bool earliest_deadline, std::vector<Slot> slots)
    : io_(io),
      queue_limit_(queue_limit),
      earliest_deadline_(earliest_deadline),
      slots_(std::move(slots)),
      pool_(slots_.size()) {
  for (const auto& slot : slots_)
    if (!runtime_info_.contains(Name(slot.task)))
      runtime_info_[Name(slot.task)] = slot.runtime->Info();
}

Admission::~Admission() { pool_.join(); }

bool Admission::Supports(Task task) const {
  return runtime_info_.contains(Name(task));
}

const Json& Admission::Info() const { return runtime_info_; }

std::string Admission::NextId() { return "kvh-" + std::to_string(++serial_); }

std::size_t Admission::Active() const {
  return std::count_if(slots_.begin(), slots_.end(),
                       [](const auto& s) { return bool(s.job); });
}

bool Admission::Submit(const std::shared_ptr<Job>& job) {
  if (draining_) {
    ++rejected_;
    return false;
  }
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].task == job->request.task && !slots_[i].job) {
      ++accepted_;
      Start(i, job);
      return true;
    }
  }
  auto& queue = queues_[static_cast<int>(job->request.task)];
  if (queue.size() >= queue_limit_) {
    ++rejected_;
    return false;
  }
  ++accepted_;
  queue.push_back(job);
  return true;
}

void Admission::Cancel(const std::shared_ptr<Job>& job, std::string reason) {
  if (!job) return;
  if (job->stop_reason.empty()) job->stop_reason = std::move(reason);
  job->request.cancelled->store(true);
  job->pipe->Stop();
  if (job->result || job->started) return;
  auto& queue = queues_[static_cast<int>(job->request.task)];
  auto it = std::find(queue.begin(), queue.end(), job);
  if (it != queue.end()) {
    queue.erase(it);
    Result res;
    res.outcome = job->stop_reason;
    Complete(job, std::move(res));
  }
}

void Admission::Shutdown() {
  draining_ = true;
  for (auto& q : queues_)
    while (!q.empty()) {
      auto job = q.front();
      Cancel(job, "shutdown");
    }
  for (auto& slot : slots_)
    if (slot.job) Cancel(slot.job, "shutdown");
  if (Active() == 0 && on_drained) on_drained();
}

Json Admission::Snapshot() const {
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
          {"active", Active()},
          {"outcomes", outcomes_},
          {"peak_buffer_bytes", peak_buffer_},
          {"tasks", tasks},
          {"draining", draining_}};
}

void Admission::Complete(const std::shared_ptr<Job>& job, Result result) {
  if (job->result) std::abort();
  if (!job->stop_reason.empty())
    result.outcome = job->stop_reason;
  else if (job->pipe->Saturated())
    result.outcome = "backpressure";
  else if (result.outcome == "cancelled" &&
           Clock::now() >= job->request.deadline)
    result.outcome = "deadline";
  ++outcomes_[result.outcome];
  peak_buffer_ = std::max(peak_buffer_, job->pipe->PeakBytes());
  auto details = result.details;
  details.erase("token_ids");
  details.erase("text");
  Json record = {{"id", job->request.id},
                 {"task", Name(job->request.task)},
                 {"outcome", result.outcome},
                 {"queue_ms",
                  job->started ? ElapsedMs(job->request.received, *job->started)
                               : ElapsedMs(job->request.received)},
                 {"backend_finished_ms", ElapsedMs(job->request.received)},
                 {"slot_released", bool(job->started)},
                 {"peak_buffer_bytes", job->pipe->PeakBytes()},
                 {"usage", result.usage},
                 {"timings", result.timings},
                 {"details", details}};
  std::cout << record.dump() << '\n' << std::flush;
  job->result = std::move(result);
  job->pipe->Notify();
}

void Admission::Start(std::size_t index, std::shared_ptr<Job> job) {
  auto& slot = slots_[index];
  slot.job = job;
  job->started = Clock::now();
  ++started_;
  job->pipe->Notify();
  net::post(pool_, [this, index, job] {
    auto result =
        slots_[index].runtime->Generate(job->request, [&](std::string chunk) {
          if (job->encode_chunk) chunk = job->encode_chunk(std::move(chunk));
          return job->pipe->Put(std::move(chunk), job->request);
        });
    if (!result.diagnostic.empty())
      std::cerr << "runtime request " << job->request.id << ": "
                << result.diagnostic << '\n';
    net::post(io_, [this, index, job, result = std::move(result)]() mutable {
      slots_[index].job.reset();
      ++releases_;
      Complete(job, std::move(result));
      Dispatch();
      if (draining_ && Active() == 0 && on_drained) on_drained();
    });
  });
}

void Admission::Dispatch() {
  if (draining_) return;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].job) continue;
    auto& queue = queues_[static_cast<int>(slots_[i].task)];
    while (!queue.empty()) {
      auto it = earliest_deadline_
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
        Complete(job, std::move(res));
        continue;
      }
      Start(i, std::move(job));
      break;
    }
  }
}

}  // namespace kvharbor::serving
