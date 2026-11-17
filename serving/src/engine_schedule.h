#ifndef KVHARBOR_SERVING_SRC_ENGINE_SCHEDULE_H_
#define KVHARBOR_SERVING_SRC_ENGINE_SCHEDULE_H_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace kvharbor::serving {

// Serializes engine calls across text contexts. Socket writes and sampling stay
// with each request's worker.
class EngineDispatch {
 public:
  enum class Phase { kPrefill, kDecode };

  struct Stats {
    unsigned prefill_calls = 0;
    unsigned decode_calls = 0;
    unsigned decode_ahead_of_prefill = 0;
    unsigned forced_prefill = 0;
    unsigned decode_after_other_prefill = 0;
    unsigned prefill_after_other_decode = 0;
    std::uint64_t wait_us = 0;
  };

  explicit EngineDispatch(bool prefer_decode) : prefer_decode_(prefer_decode) {}

  template <class Cancelled, class Operation>
  std::optional<int> Run(Phase phase, const void* worker, Stats& stats,
                         Cancelled cancelled, Operation operation) {
    const auto arrived = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex_);
    Ticket ticket{phase};
    waiters_.push_back(&ticket);
    while (true) {
      if (cancelled()) {
        waiters_.erase(std::find(waiters_.begin(), waiters_.end(), &ticket));
        ready_.notify_all();
        return std::nullopt;
      }
      if (!active_ && NextWaiting() == &ticket) break;
      // Cancellation and deadlines are shared with the HTTP reactor.
      ready_.wait_for(lock, std::chrono::milliseconds(4));
    }

    const bool other_worker = previous_worker_ && previous_worker_ != worker;
    auto prefill = FirstWaiting(Phase::kPrefill);
    if (phase == Phase::kDecode) {
      ++stats.decode_calls;
      if (prefill != waiters_.end()) {
        if (waiters_.front()->phase == Phase::kPrefill)
          ++stats.decode_ahead_of_prefill;
        ++decode_streak_;
      } else {
        decode_streak_ = 0;
      }
      if (other_worker && previous_phase_ == Phase::kPrefill)
        ++stats.decode_after_other_prefill;
    } else {
      ++stats.prefill_calls;
      if (prefer_decode_ && decode_streak_ >= kMaxDecodeStreak &&
          FirstWaiting(Phase::kDecode) != waiters_.end())
        ++stats.forced_prefill;
      decode_streak_ = 0;
      if (other_worker && previous_phase_ == Phase::kDecode)
        ++stats.prefill_after_other_decode;
    }
    previous_worker_ = worker;
    previous_phase_ = phase;
    stats.wait_us += std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - arrived)
                         .count();
    waiters_.erase(std::find(waiters_.begin(), waiters_.end(), &ticket));
    active_ = true;
    lock.unlock();

    struct Release {
      EngineDispatch& dispatch;
      ~Release() {
        std::lock_guard lock(dispatch.mutex_);
        dispatch.active_ = false;
        dispatch.ready_.notify_all();
      }
    } release{*this};
    return operation();
  }

 private:
  struct Ticket {
    Phase phase;
  };
  using Waiters = std::deque<Ticket*>;
  static constexpr unsigned kMaxDecodeStreak = 4;

  Waiters::iterator FirstWaiting(Phase phase) {
    return std::find_if(
        waiters_.begin(), waiters_.end(),
        [phase](const Ticket* ticket) { return ticket->phase == phase; });
  }

  Ticket* NextWaiting() {
    if (prefer_decode_) {
      const auto prefill = FirstWaiting(Phase::kPrefill);
      if (decode_streak_ >= kMaxDecodeStreak && prefill != waiters_.end())
        return *prefill;
      const auto decode = FirstWaiting(Phase::kDecode);
      if (decode != waiters_.end()) return *decode;
    }
    return waiters_.front();
  }

  const bool prefer_decode_;
  std::mutex mutex_;
  std::condition_variable ready_;
  Waiters waiters_;
  bool active_ = false;
  unsigned decode_streak_ = 0;
  const void* previous_worker_ = nullptr;
  Phase previous_phase_ = Phase::kPrefill;
};

}  // namespace kvharbor::serving

#endif  // KVHARBOR_SERVING_SRC_ENGINE_SCHEDULE_H_
