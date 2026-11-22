# TensorRT Edge-LLM deployment

[Serving guide](serving.md) · [简体中文](tensorrt.zh-CN.md)

Complete the [shared CUDA build](serving.md#build-prerequisites). This recipe uses
SM120, SDK 0.12.0 and Qwen3.5-0.8B. Run from the repository root with these paths.

```bash
edge_src="$PWD/.cache/kvharbor/TensorRT-Edge-LLM-v0.12.0"
edge_python="$PWD/.cache/kvharbor/edge-python-v0.12.0"
onnx_dir="$PWD/.cache/qwen3.5-edge-onnx"
export KVHARBOR_EDGE_SDK="$PWD/.cache/kvharbor/edge-sdk-v0.12.0-sm120"
edge_image=$(python3 -c 'import json; print(json.load(open("configs/runtime/edge-llm.json"))["build_image"])')
```

## Prepare SDK and export tools

On first setup, run both preparation groups below. The SDK supplies
`bin/llm_build`, its plugin and the CMake package. The Python environment uses
PyTorch from the pinned image. [Runtime lock](../configs/runtime/edge-llm.json)

<details>
<summary>Fetch the pinned source and build the SDK</summary>

```bash
mkdir -p .cache/kvharbor
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM "$edge_src"
git -C "$edge_src" checkout 513dcd18291e587c80928911a480761352f3c2d8
git -C "$edge_src" submodule update --init --recursive
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/workspace" --workdir /workspace \
  --entrypoint bash "$edge_image" -c '
    set -euo pipefail
    cmake -S .cache/kvharbor/TensorRT-Edge-LLM-v0.12.0 \
      -B .cache/kvharbor/edge-sdk-build-v0.12.0-sm120 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_UNIT_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=/workspace/.cache/kvharbor/edge-sdk-v0.12.0-sm120
    cmake --build .cache/kvharbor/edge-sdk-build-v0.12.0-sm120 --parallel 2
    cmake --install .cache/kvharbor/edge-sdk-build-v0.12.0-sm120
  '
```

</details>

<details>
<summary>Create the ONNX export environment in the image</summary>

The isolated venv uses the SDK's NumPy version. Clear the image-wide pip
constraint while installing its export dependencies.

```bash
mkdir -p "$edge_python"
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$edge_src,dst=/edge-src,readonly" \
  --mount "type=bind,src=$edge_python,dst=/edge-python" \
  --env PYTHONPATH=/edge-src --env PIP_CACHE_DIR=/edge-python/pip-cache \
  --entrypoint bash "$edge_image" -c '
    set -euo pipefail
    unset PIP_CONSTRAINT
    python3 -m venv --system-site-packages /edge-python/venv
    /edge-python/venv/bin/python -m pip install \
      numpy==2.2.6 onnx==1.19.0 onnxscript==0.7.1 onnx-graphsurgeon==0.6.1 \
      transformers==5.17.0 huggingface-hub==1.15.0 safetensors==0.8.0 \
      peft==0.19.1 datasets==5.0.0 tqdm==4.69.0
  '
```

</details>

## Export the text model

```bash
python3 scripts/prepare_pc_model.py Qwen3.5-0.8B
mkdir -p "$onnx_dir"
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$edge_src,dst=/edge-src,readonly" \
  --mount "type=bind,src=$edge_python,dst=/edge-python,readonly" \
  --mount "type=bind,src=$PWD/models/qwen3.5-0.8b,dst=/checkpoint,readonly" \
  --mount "type=bind,src=$onnx_dir,dst=/onnx" \
  --env PYTHONPATH=/edge-src --env HF_HUB_OFFLINE=1 \
  --entrypoint /edge-python/venv/bin/python "$edge_image" \
  -m tensorrt_edgellm.cli export /checkpoint /onnx --skip-visual --skip-audio
```

Outputs are `$onnx_dir/llm/model.onnx`, external weights and tokenizer files.

## Build the engine

The pinned image needs additional SM120 builder libraries. Prepare the package
below, or set `KVHARBOR_TRT_BUILDER_LIB_DIR` to an existing matching directory.

<details>
<summary>Download, verify and extract TensorRT 11.2.1.2 builder libraries</summary>

```bash
trt_builder="$PWD/.cache/kvharbor/trt-builder-11.2.1-sm120"
mkdir -p "$trt_builder"
deb="$trt_builder/libnvinfer11_11.2.1.2-1+cuda13.3_amd64.deb"
curl -fL --retry 3 --retry-all-errors --continue-at - \
  -o "$deb" \
  'https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2404/x86_64/libnvinfer11_11.2.1.2-1+cuda13.3_amd64.deb'
printf '%s  %s\n' \
  4286e4df98c77168724eff2d79bafd2c2bb7939f6665437cfd4bf033d2143ee0 \
  "$deb" | sha256sum --check -
dpkg-deb --fsys-tarfile "$deb" | tar -xf - -C "$trt_builder" \
  ./usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource_sm120.so.11.2.1 \
  ./usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource_ptx.so.11.2.1
export KVHARBOR_TRT_BUILDER_LIB_DIR="$trt_builder/usr/lib/x86_64-linux-gnu"
```

</details>

```bash
bash scripts/build_pc_edge_engine.sh "$onnx_dir/llm" models/qwen3.5-0.8b-edge
```

The engine destination must be new. This build supports one sequence, up to
1,024 input tokens and a 2,048-token KV cache. Use a new second-argument path
for another build.

## Serve and query

```bash
unset KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
bash scripts/native_serving.sh build
bash scripts/native_serving.sh serve --text-backend tensorrt-edge-llm \
  --model-id Qwen3.5-0.8B --context 2048 --workers 1 \
  --max-output-tokens 1024 --timeout-ms 180000 \
  --engine-dir /workspace/models/qwen3.5-0.8b-edge \
  --checkpoint-dir /workspace/models/qwen3.5-0.8b
```

In another terminal, check that `/health` reports `tensorrt-edge-llm`. The
request returns `42` and ends with `[DONE]`. CUDA Graph is enabled by default.

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsSN http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-0.8B","stream":true,"temperature":0,"max_tokens":48,
       "messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}]}'
```
