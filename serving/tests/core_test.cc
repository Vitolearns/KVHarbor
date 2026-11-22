#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "kvharbor/serving/runtime.h"

using kvharbor::serving::EncodePcm16;
using kvharbor::serving::Failure;
using kvharbor::serving::GreedyArgmax;
using kvharbor::serving::IntegerFieldOr;
using kvharbor::serving::Json;
using kvharbor::serving::NumberFieldOr;
using kvharbor::serving::ParseSamplingMode;
using kvharbor::serving::ReadPcm16Wav;
using kvharbor::serving::TakeUtf8;
using kvharbor::serving::Waveform;

namespace {
void Check(bool pass) {
  if (pass) return;
  std::cerr << "serving core check failed\n";
  std::exit(EXIT_FAILURE);
}
}  // namespace

int main() {
  Check(GreedyArgmax(std::array{-4.0f, -1.0f, -1.0f, -9.0f}) == 1);
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  Check(GreedyArgmax(std::array{-infinity, infinity, infinity}) == 1);
  Check(GreedyArgmax(std::array{0.0f, -0.0f}) == 0);
  Check(GreedyArgmax(std::array{nan, 10.0f, 20.0f}) == 0);
  Check(GreedyArgmax(std::array{1.0f, nan, 2.0f}) == 2);
  Check(!GreedyArgmax({}));
  Check(!ParseSamplingMode("invalid"));

  kvharbor::serving::RuntimeConfig config;
  config.model_id = "qwen";
  config.context = 128;
  const Json chat = {
      {"model", "qwen"},
      {"max_tokens", 8},
      {"messages", Json::array({{{"role", "user"}, {"content", "hello"}}})}};
  auto parsed = kvharbor::serving::ParseTextInput(chat, config);
  const auto& text = std::get<kvharbor::serving::TextInput>(parsed);
  Check(text.apply_chat_template && text.max_tokens == 8 &&
        text.messages.front().content == "hello");
  const Json raw = {{"prompt", "hello"}, {"max_tokens", 8}};
  Check(!std::get<kvharbor::serving::TextInput>(
             kvharbor::serving::ParseTextInput(raw, config))
             .apply_chat_template);
  for (const auto& [key, value] : std::vector<std::pair<std::string, Json>>{
           {"temperature", 1},
           {"model", "other"},
           {"max_tokens", 128},
           {"messages", Json::array({{{"role", "tool"}, {"content", "x"}}})}}) {
    auto invalid_chat = chat;
    invalid_chat[key] = value;
    Check(std::holds_alternative<Failure>(
        kvharbor::serving::ParseTextInput(invalid_chat, config)));
  }

  // Queued requests must own their text after the wire JSON is destroyed.
  kvharbor::serving::Request request;
  {
    Json body = chat;
    body["messages"][0]["content"] = std::string(512, 'x');
    request.input = kvharbor::serving::ParseTextRequest(body, config, false);
    body["messages"][0]["content"] = "replaced";
  }
  auto queued_input =
      kvharbor::serving::ReadRequestInput<kvharbor::serving::TextInput>(
          request);
  auto output = std::get<const kvharbor::serving::TextInput*>(queued_input);
  Check(output->messages.front().content == std::string(512, 'x'));

  Check(IntegerFieldOr(Json::object(), "max_tokens", 64) == 64);
  const int minimum = std::numeric_limits<int>::min();
  const int maximum = std::numeric_limits<int>::max();
  Check(IntegerFieldOr(Json{{"max_tokens", minimum}}, "max_tokens", 0) ==
        minimum);
  Check(
      IntegerFieldOr(Json{{"max_tokens", static_cast<std::uint64_t>(maximum)}},
                     "max_tokens", 0) == maximum);
  for (const Json& invalid :
       {Json(true), Json(false), Json(1.9), Json(2.0), Json("2"), Json(nullptr),
        Json(std::numeric_limits<std::uint64_t>::max()),
        Json(static_cast<std::int64_t>(minimum) - 1)}) {
    Check(!IntegerFieldOr(Json{{"max_tokens", invalid}}, "max_tokens", 64));
  }
  Check(NumberFieldOr(Json{{"speed", 1.25}}, "speed", 1.0) == 1.25);
  Check(!NumberFieldOr(Json{{"speed", true}}, "speed", 1.0));
  Check(!NumberFieldOr(Json{{"temperature", false}}, "temperature", 0.0));
  // Exercise vector-lane ties, unaligned spans and every short scalar-tail
  // size.
  std::vector<float> tied(34, -1.0f);
  tied[8] = 5.0f;
  tied[9] = 5.0f;
  tied[33] = 5.0f;
  Check(GreedyArgmax(std::span<const float>(tied).subspan(1)) == 7);
  tied[33] = 6.0f;
  Check(GreedyArgmax(std::span<const float>(tied).subspan(1)) == 32);
  tied[18] = nan;
  tied[19] = infinity;
  tied[28] = infinity;
  Check(GreedyArgmax(std::span<const float>(tied).subspan(1)) == 18);
  tied[1] = nan;
  Check(GreedyArgmax(std::span<const float>(tied).subspan(1)) == 0);
  std::uint32_t seed = 42;
  for (std::size_t size = 1; size <= 257; ++size) {
    std::vector<float> data(size + 1);
    for (auto& value : data) {
      seed = seed * 1664525u + 1013904223u;
      value = static_cast<float>(seed % 997) - 498.0f;
    }
    auto values = std::span<const float>(data).subspan(1);
    Check(GreedyArgmax(values) ==
          std::max_element(values.begin(), values.end()) - values.begin());
  }

  std::string pending = "\xe4\xbd";
  Check(std::get<std::string>(TakeUtf8(pending)).empty());
  pending +=
      "\xa0"
      "A";
  Check(std::get<std::string>(TakeUtf8(pending)) == "你A" && pending.empty());
  pending = "\xe4";
  Check(std::get<std::string>(TakeUtf8(pending, true)) == "\xef\xbf\xbd" &&
        pending.empty());
  pending = "\xed\xa0\x80";
  auto invalid = TakeUtf8(pending);
  const auto* failure = std::get_if<Failure>(&invalid);
  Check(failure && failure->message == "invalid UTF-8 scalar");

  std::string wav = "RIFF";
  auto append = [&](std::uint32_t v, int bytes) {
    while (bytes--) {
      wav.push_back(static_cast<char>(v & 255));
      v >>= 8;
    }
  };
  append(40, 4);
  wav += "WAVEfmt ";
  append(16, 4);
  append(1, 2);
  append(1, 2);
  append(16000, 4);
  append(32000, 4);
  append(2, 2);
  append(16, 2);
  wav += "data";
  append(4, 4);
  append(0x8000, 2);
  append(0x7fff, 2);
  auto decoded = ReadPcm16Wav(wav);
  const auto* audio = std::get_if<Waveform>(&decoded);
  Check(audio && audio->sample_rate == 16000 && audio->samples.size() == 2 &&
        audio->samples[0] == -1.0f && audio->samples[1] > 0.99f);
  Check(
      EncodePcm16(audio->samples.data(), audio->samples.size()).substr(0, 2) ==
      wav.substr(44, 2));
  Check(std::holds_alternative<Failure>(
      ReadPcm16Wav(wav.substr(0, wav.size() - 1))));
}
