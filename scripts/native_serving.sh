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
if [[ $mode == build ]]; then
  test -f build-llama-cuda/kvharbor-build.json
  test -f /usr/include/boost/beast.hpp
  test -f /usr/include/nlohmann/json.hpp
  mkdir -p "$build"
  docker run --rm "${common[@]}" --network none \
    --mount "type=bind,src=$root/$build,dst=/native" \
    --mount type=bind,src=/usr/include/boost,dst=/usr/include/boost,readonly \
    --mount type=bind,src=/usr/include/nlohmann,dst=/usr/include/nlohmann,readonly \
    --entrypoint bash "$image" -c '
      set -euo pipefail
      cmake -S /workspace/serving -B /native -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DKVHARBOR_SERVING_LLAMA=ON -DKVHARBOR_LLAMA_BUILD=/build \
        -DKVHARBOR_LLAMA_SOURCE="/workspace/.cache/llama.cpp-$1" \
        -DKVHARBOR_SERVING_AUDIO="$2" -DKVHARBOR_SHERPA_ROOT=/workspace/.cache/kvharbor/sherpa
      cmake --build /native --parallel 2
    ' bash "$revision" "$audio"
  docker run --rm "${common[@]}" --gpus all --network none \
    --mount "type=bind,src=$root/$build,dst=/native" \
    --entrypoint bash "$image" -c '
      export LD_LIBRARY_PATH="/build/bin:/workspace/.cache/kvharbor/sherpa/lib:${LD_LIBRARY_PATH:-}"
      ctest --test-dir /native --output-on-failure
    '
  exit
fi
test -f "${GGUF_MODEL:?set GGUF_MODEL to the verified GGUF file}"
model=$(realpath -- "$GGUF_MODEL")
out=${KVHARBOR_OUTPUT_DIR:-$root/results/raw}
mkdir -p "$out"
out=$(realpath -- "$out")
args=("${common[@]}" --gpus all --stop-timeout 30
  --mount "type=bind,src=$root/$build,dst=/native,readonly"
  --mount "type=bind,src=$model,dst=/models/llm.gguf,readonly"
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
' bash "/native/$bin" --model /models/llm.gguf "$@"
