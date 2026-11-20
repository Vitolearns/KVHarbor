#!/usr/bin/env bash
# Build and run the C++ server against the pinned CUDA runtime.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mode=${1:-}
if [[ -z $mode ]]; then
  printf '%s\n' 'Usage: bash scripts/native_serving.sh {build|serve|supervise|bench} [args...]' >&2
  exit 2
fi
shift
readarray -t pinned < <(python3 -c 'import json; p=json.load(open("configs/runtime/llama.json")); print(p["build_image"]); print(p["revision"])')
image=${pinned[0]}
revision=${pinned[1]}
build=build-serving-cuda
audio=OFF
if [[ ${KVHARBOR_AUDIO:-0} == 1 ]]; then
  build=build-serving-cuda-audio
  audio=ON
fi
common=(--pull=never --user "$(id -u):$(id -g)"
  --mount "type=bind,src=$root,dst=/workspace,readonly"
  --mount "type=bind,src=$root/build-llama-cuda,dst=/build,readonly")
edge=OFF
cuda_arch=
if [[ -n ${KVHARBOR_EDGE_SDK:-} ]]; then
  sdk=$(realpath -e -- "$KVHARBOR_EDGE_SDK")
  common+=(--mount "type=bind,src=$sdk,dst=/edge-sdk,readonly"
    --env EDGELLM_PLUGIN_PATH=/edge-sdk/lib/libNvInfer_edgellm_plugin.so)
  edge=ON
  cuda_arch=${KVHARBOR_CUDA_ARCHITECTURES:-120}
  build+=-edge
fi
build=$(realpath -m -- "${KVHARBOR_SERVING_BUILD_DIR:-$build}")
if [[ $mode == build ]]; then
  test -f build-llama-cuda/kvharbor-build.json
  test -f /usr/include/boost/beast.hpp
  test -f /usr/include/nlohmann/json.hpp
  mkdir -p "$build"
  docker run --rm "${common[@]}" --network none \
    --mount "type=bind,src=$build,dst=/native" \
    --mount type=bind,src=/usr/include/boost,dst=/usr/include/boost,readonly \
    --mount type=bind,src=/usr/include/nlohmann,dst=/usr/include/nlohmann,readonly \
    --entrypoint bash "$image" -c '
      set -euo pipefail
      arch_args=()
      if [[ -n $4 ]]; then arch_args+=("-DCMAKE_CUDA_ARCHITECTURES=$4"); fi
      cmake -S /workspace/serving -B /native -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DKVHARBOR_SERVING_LLAMA=ON -DKVHARBOR_LLAMA_BUILD=/build \
        -DKVHARBOR_LLAMA_SOURCE="/workspace/.cache/llama.cpp-$1" \
        -DKVHARBOR_SERVING_AUDIO="$2" -DKVHARBOR_SHERPA_ROOT=/workspace/.cache/kvharbor/sherpa \
        -DKVHARBOR_SERVING_EDGE_LLM="$3" -DCMAKE_PREFIX_PATH=/edge-sdk \
        "${arch_args[@]}"
      cmake --build /native --parallel 2
    ' bash "$revision" "$audio" "$edge" "$cuda_arch"
  docker run --rm "${common[@]}" --gpus all --network none \
    --mount "type=bind,src=$build,dst=/native" \
    --entrypoint bash "$image" -c '
      export LD_LIBRARY_PATH="/build/bin:/workspace/.cache/kvharbor/sherpa/lib:${LD_LIBRARY_PATH:-}"
      ctest --test-dir /native --output-on-failure
    '
  exit
fi
backend=llama.cpp
cli=("$@")
for ((i=0; i<${#cli[@]}; i++)); do
  if [[ ${cli[i]} == --text-backend ]]; then
    backend=${cli[i+1]:?missing text backend}
  fi
done
model_args=()
case "$backend" in
  llama.cpp)
    model=$(realpath -e -- "${GGUF_MODEL:?set GGUF_MODEL to the verified GGUF file}")
    common+=(--mount "type=bind,src=$model,dst=/models/llm.gguf,readonly")
    model_args=(--model /models/llm.gguf);;
  tensorrt-edge-llm)
    if [[ $edge != ON ]]; then
      echo 'Set KVHARBOR_EDGE_SDK to the installed TensorRT Edge-LLM SDK.' >&2
      exit 2
    fi;;
  none) ;;
  *) echo "Unknown text backend: $backend" >&2; exit 2;;
esac
out=${KVHARBOR_OUTPUT_DIR:-$root/results/raw}
mkdir -p "$out"
out=$(realpath -- "$out")
args=("${common[@]}" --gpus all --stop-timeout 30
  --mount "type=bind,src=$build,dst=/native,readonly"
  --mount "type=bind,src=$out,dst=/run-output")
case "$mode" in
  serve) bin=kvharbor-server; args+=(--rm --network host --name "${KVHARBOR_NATIVE_NAME:-kvharbor-native-server}");;
  supervise) bin=kvharbor-server; args+=(--detach --init --restart on-failure:3
    --label kvharbor.role=acceptance -e KVHARBOR_SUPERVISED=1
    --network host --name "${KVHARBOR_NATIVE_NAME:-kvharbor-acceptance}");;
  bench) bin=kvharbor-runtime-bench; args+=(--rm --network none);;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2;;
esac
exec docker run "${args[@]}" --entrypoint bash "$image" -c '
  export LD_LIBRARY_PATH="/build/bin:/workspace/.cache/kvharbor/sherpa/lib:${LD_LIBRARY_PATH:-}"
  if [[ ${KVHARBOR_SUPERVISED:-0} == 1 ]]; then
    printf "%s\n" "$$" > /tmp/kvharbor-server.pid
  fi
  exec "$@"
' bash "/native/$bin" "${model_args[@]}" "$@"
