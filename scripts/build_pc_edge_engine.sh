#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: bash scripts/build_pc_edge_engine.sh ONNX_LLM_DIR [ENGINE_DIR]'
  echo 'Set KVHARBOR_EDGE_SDK to the installed Edge-LLM SDK.'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
onnx=$(realpath -e -- "$1")
sdk=$(realpath -e -- "${KVHARBOR_EDGE_SDK:?Set KVHARBOR_EDGE_SDK to the installed Edge-LLM SDK}")
dest=$(realpath -m -- "${2:-$repo/models/qwen3.5-0.8b-edge}")
if [[ ! -d $onnx || ! -x $sdk/bin/llm_build || ! -r $sdk/lib/libNvInfer_edgellm_plugin.so ]]; then
  echo 'Expected an exported LLM directory and an SDK with llm_build and its plugin.' >&2
  exit 1
fi
if [[ -e $dest || -L $dest ]]; then
  echo "Engine destination already exists: $dest" >&2
  exit 1
fi

image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["build_image"])' \
  "$repo/configs/runtime/edge-llm.json")
extra=()
if [[ -n ${KVHARBOR_TRT_BUILDER_LIB_DIR:-} ]]; then
  resources=$(realpath -e -- "$KVHARBOR_TRT_BUILDER_LIB_DIR")
  [[ -d $resources ]] || { echo "Expected a TensorRT library directory: $resources" >&2; exit 1; }
  extra+=(--mount "type=bind,src=$resources,dst=/trt-builder,readonly")
fi

parent=$(dirname -- "$dest")
mkdir -p -- "$parent"
stage=$(mktemp -d -- "$parent/.edge-engine.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
docker run --rm --network none --gpus all --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$sdk,dst=/edge-sdk,readonly" \
  --mount "type=bind,src=$onnx,dst=/onnx,readonly" \
  --mount "type=bind,src=$stage,dst=/engine" \
  "${extra[@]}" --env EDGELLM_PLUGIN_PATH=/edge-sdk/lib/libNvInfer_edgellm_plugin.so \
  --entrypoint /bin/bash "$image" -c '
    export LD_LIBRARY_PATH="/trt-builder:/edge-sdk/lib:${LD_LIBRARY_PATH:-}"
    exec /edge-sdk/bin/llm_build --onnx-dir /onnx --engine-dir /engine \
      --max-batch-size 1 --max-input-len 1024 --max-kv-cache-capacity 2048
  '
if [[ -z $(find "$stage" -type f -name '*.engine' -size +0c -print -quit) ]]; then
  echo 'The builder did not produce an engine.' >&2
  exit 1
fi
mv -Tn -- "$stage" "$dest"
if [[ -d $stage ]]; then
  echo "Another build created $dest before this build finished." >&2
  exit 1
fi
trap - EXIT
echo "Engine ready: $dest"
