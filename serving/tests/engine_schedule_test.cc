#include "engine_schedule.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <latch>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

using kvharbor::serving::EngineDispatch;
using Phase = EngineDispatch::Phase;

namespace {
void Check(bool condition, const char* message) {
  if (condition) return;
  std::cerr << message << '\n';
  std::exit(EXIT_FAILURE);
}

void CheckOrder(bool prefer_decode, const std::vector<int>& expected) {
  EngineDispatch dispatch(prefer_decode);
  std::latch occupied(1), release(1);
  EngineDispatch::Stats initial;
  std::jthread owner([&] {
    dispatch.Run(
        Phase::kPrefill, &initial, initial, [] { return false; },
        [&] {
          occupied.count_down();
          release.wait();
          return 0;
        });
  });
  occupied.wait();

  std::array<EngineDispatch::Stats, 6> stats{};
  std::vector<int> order;
  std::mutex order_mutex;
  std::vector<std::jthread> queued;
  std::array<std::latch, 6> registered{std::latch(1), std::latch(1),
                                       std::latch(1), std::latch(1),
                                       std::latch(1), std::latch(1)};
  std::array<bool, 6> announced{};
  for (int i = 0; i < 6; ++i) {
    queued.emplace_back([&, i] {
      dispatch.Run(
          i == 0 ? Phase::kPrefill : Phase::kDecode, &stats[i], stats[i],
          [&] {
            if (!announced[i]) {
              announced[i] = true;
              registered[i].count_down();
            }
            return false;
          },
          [&] {
            std::lock_guard lock(order_mutex);
            order.push_back(i);
            return 0;
          });
    });
    registered[i].wait();
  }
  release.count_down();
  queued.clear();
  owner.join();
  Check(order == expected, "dispatch order differs from policy");
  if (prefer_decode) {
    Check(stats[0].forced_prefill == 1, "prefill bound was not measured");
    Check(stats[1].decode_ahead_of_prefill == 1,
          "decode did not bypass older prefill");
  }
}

void CheckWaitingCancellation() {
  EngineDispatch dispatch(true);
  std::latch occupied(1), release(1), registered(1);
  EngineDispatch::Stats running_stats, cancelled_stats;
  std::jthread owner([&] {
    dispatch.Run(
        Phase::kPrefill, &running_stats, running_stats, [] { return false; },
        [&] {
          occupied.count_down();
          release.wait();
          return 0;
        });
  });
  occupied.wait();
  std::atomic<bool> cancelled{false};
  std::optional<int> result;
  bool announced = false;
  std::jthread waiter([&] {
    result = dispatch.Run(
        Phase::kDecode, &cancelled_stats, cancelled_stats,
        [&] {
          if (!announced) {
            announced = true;
            registered.count_down();
          }
          return cancelled.load();
        },
        [] { return 12; });
  });
  registered.wait();
  cancelled.store(true);
  waiter.join();
  release.count_down();
  owner.join();
  Check(!result && cancelled_stats.decode_calls == 0,
        "cancelled waiter ran on the engine");
  const auto next = dispatch.Run(
      Phase::kDecode, &cancelled_stats, cancelled_stats, [] { return false; },
      [] { return 42; });
  Check(next == 42, "cancelled waiter blocked a later dispatch");
}

void CheckFailureRelease() {
  EngineDispatch dispatch(true);
  EngineDispatch::Stats stats;
  const auto failed = dispatch.Run(
      Phase::kPrefill, &stats, stats, [] { return false; }, [] { return -1; });
  Check(failed == -1, "engine error status was lost");
  const auto next = dispatch.Run(
      Phase::kDecode, &stats, stats, [] { return false; }, [] { return 42; });
  Check(next == 42, "engine failure retained the permit");
}
}  // namespace

int main() {
  CheckOrder(false, {0, 1, 2, 3, 4, 5});
  CheckOrder(true, {1, 2, 3, 4, 0, 5});
  CheckWaitingCancellation();
  CheckFailureRelease();
}
