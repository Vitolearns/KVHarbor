#ifndef KVHARBOR_SERVING_SRC_EDGE_SDK_BOUNDARY_H_
#define KVHARBOR_SERVING_SRC_EDGE_SDK_BOUNDARY_H_

#include <exception>
#include <utility>

#include "kvharbor/serving/runtime.h"

namespace kvharbor::serving {
// The vendor C++ API can throw from construction and inference. Translate
// failures here, including on the thread that calls handleRequest.
template <typename T, typename Fn>
ValueOrFailure<T> EdgeSdkCall(Fn&& call) {
  try {
    return std::forward<Fn>(call)();
  } catch (const std::exception& error) {
    return BackendError(error.what());
  } catch (...) {
    return BackendError("TensorRT Edge-LLM raised an unknown exception");
  }
}
}  // namespace kvharbor::serving

#endif  // KVHARBOR_SERVING_SRC_EDGE_SDK_BOUNDARY_H_
