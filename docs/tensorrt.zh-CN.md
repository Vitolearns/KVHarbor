# TensorRT Edge-LLM 部署

[部署指南](serving.zh-CN.md) · [English](tensorrt.md)

先完成[共用 CUDA 依赖](serving.zh-CN.md#构建准备)。以下流程使用 SM120、SDK 0.12.0 和 Qwen3.5-0.8B。
命令从仓库根目录执行，准备目录由以下变量指定。

```bash
edge_src="$PWD/.cache/kvharbor/TensorRT-Edge-LLM-v0.12.0"
edge_python="$PWD/.cache/kvharbor/edge-python-v0.12.0"
onnx_dir="$PWD/.cache/qwen3.5-edge-onnx"
export KVHARBOR_EDGE_SDK="$PWD/.cache/kvharbor/edge-sdk-v0.12.0-sm120"
edge_image=$(python3 -c 'import json; print(json.load(open("configs/runtime/edge-llm.json"))["build_image"])')
```

## 准备 SDK 与导出环境

首次部署执行下方两组准备命令。SDK 提供 `bin/llm_build`、插件和 CMake 包，
Python 环境使用固定镜像中的 PyTorch。[版本清单](../configs/runtime/edge-llm.json)

<details>
<summary>获取固定版本源码并构建 SDK</summary>

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
<summary>在镜像内创建 ONNX 导出环境</summary>

独立 venv 使用 SDK 的 NumPy 版本，安装时清除镜像全局的 pip 约束。

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

## 导出文本模型

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

输出为 `$onnx_dir/llm/model.onnx`、外部权重及 tokenizer 文件。

## 构建引擎

固定镜像需要额外的 SM120 builder 库。准备下方软件包，或将
`KVHARBOR_TRT_BUILDER_LIB_DIR` 指向已有的匹配库目录。

<details>
<summary>下载、校验并提取 TensorRT 11.2.1.2 构建库</summary>

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

目标引擎目录须尚未创建。此构建支持单序列、最多 1,024 个输入 token 和
2,048 个 token 的 KV 缓存。重新构建时给第二个参数指定新目录。

## 启动与请求

```bash
unset KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
bash scripts/native_serving.sh build
bash scripts/native_serving.sh serve --text-backend tensorrt-edge-llm \
  --model-id Qwen3.5-0.8B --context 2048 --workers 1 \
  --max-output-tokens 1024 --timeout-ms 180000 \
  --engine-dir /workspace/models/qwen3.5-0.8b-edge \
  --checkpoint-dir /workspace/models/qwen3.5-0.8b
```

另开终端发送请求。`/health` 的后端应为 `tensorrt-edge-llm`，输出为 `42`，
SSE 以 `[DONE]` 结束。CUDA Graph 默认开启。

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsSN http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-0.8B","stream":true,"temperature":0,"max_tokens":48,
       "messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}]}'
```
