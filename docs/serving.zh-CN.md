# 部署指南

[项目首页](../README.zh-CN.md) · [English](serving.md)

## 构建准备

参考环境是 Ubuntu 24.04（原生 Linux 或 WSL2）和 RTX 5090 Laptop GPU，
锁定的 CUDA 构建目标是 SM120。主机需要 Bash、Git、curl、Python 3.12+、
可访问 NVIDIA GPU 的 Docker，以及 `libboost-dev` 和 `nlohmann-json3-dev`。
下列命令从仓库根目录执行。

三个启动方式共用同一个 CUDA 镜像和 llama.cpp 构建。先拉取镜像并构建依赖。

```bash
image=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["build_image"])')
docker pull "$image"
python3 scripts/llama_runtime.py build --jobs 4
```

## CUDA 文本生成

创建 `models/` 目录并将[锁定版本的 GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/resolve/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf)保存到其中。
[文本运行时清单](../configs/runtime/llama.json)记录了模型校验和、llama.cpp 版本和 CUDA 镜像。

```bash
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha256=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["model"]["sha256"])')
printf '%s  %s\n' "$model_sha256" "$GGUF_MODEL" | sha256sum --check -
```

校验通过后，构建服务。

```bash
bash scripts/native_serving.sh build
```

以前台模式启动服务。

```bash
bash scripts/native_serving.sh serve --text-backend llama.cpp \
  --workers 1 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

另开终端检查健康状态并发送会话请求。SSE 最后一条记录为 `data: [DONE]`。

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsSN http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Q4_K_M","stream":true,"temperature":0,"max_tokens":48,
       "messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}]}'
```

## TensorRT Edge-LLM 文本生成

先完成[构建准备](#构建准备)，然后安装
[运行时清单](../configs/runtime/edge-llm.json)指定版本的 TensorRT Edge-LLM SDK。
用同一 SDK 为目标 GPU 构建 Qwen3.5-0.8B 引擎，将引擎目录放在
`models/qwen3.5-0.8b-edge`。准备脚本会下载并校验对应的 checkpoint。
启动脚本将仓库挂载到容器内的 `/workspace`。

切换后端前，先停止正在运行的文本服务。

```bash
python3 scripts/prepare_pc_model.py Qwen3.5-0.8B
export KVHARBOR_EDGE_SDK=/absolute/path/to/edge-llm-sdk
bash scripts/native_serving.sh build

bash scripts/native_serving.sh serve --text-backend tensorrt-edge-llm \
  --model-id Qwen3.5-0.8B --context 2048 --workers 1 \
  --engine-dir /workspace/models/qwen3.5-0.8b-edge \
  --checkpoint-dir /workspace/models/qwen3.5-0.8b
```

在 `GET /health` 中确认 `runtimes.llm.backend` 为 `tensorrt-edge-llm`。
复用上方会话请求，将模型改为 `"model":"Qwen3.5-0.8B"`。
CUDA Graph 默认开启，可用 `--edge-no-cuda-graph` 关闭以做对照。

## 启用语音

先完成[构建准备](#构建准备)，再停止占用 8082 端口的服务。
准备脚本按[语音运行时清单](../configs/runtime/audio.json)校验 sherpa-onnx、Qwen3-ASR 和 Kokoro。
纯语音启动使用 `--text-backend none`，无需 GGUF 模型。
如果要同时运行文本和语音，构建和启动时都设置 `KVHARBOR_AUDIO=1`，并给所选的
文本启动命令加上两项语音模型目录参数。

```bash
python3 scripts/prepare_audio_runtime.py
KVHARBOR_AUDIO=1 bash scripts/native_serving.sh build
KVHARBOR_AUDIO=1 bash scripts/native_serving.sh serve --text-backend none \
  --asr-model-dir /workspace/models/audio/qwen3-asr-0.6b-int8 \
  --tts-model-dir /workspace/models/audio/kokoro-v1.0
```

另开终端转写仓库内的 Kokoro WAV，并将合成语音保存为裸 PCM。实际采样率见
`X-Audio-Sample-Rate` 响应头。

```bash
curl -fsS http://127.0.0.1:8082/v1/audio/transcriptions \
  -H 'Content-Type: audio/wav' --data-binary @assets/readme/ferry.wav
curl -fsS http://127.0.0.1:8082/v1/audio/speech \
  -H 'Content-Type: application/json' -d '{"input":"See you at the pier."}' \
  -D speech.headers -o speech.pcm
```

## 实现细节

[NOTICE.md](../NOTICE.md)记录署名与来源。[llama.cpp](../configs/runtime/llama.json)、
[Edge-LLM](../configs/runtime/edge-llm.json)和[语音](../configs/runtime/audio.json)运行时清单
列出上游组件和模型产物。

### Context 与调度

在 llama.cpp 适配器中，每个 worker 独占一个请求，直到原生调用返回。一个 context 只包含一条序列
（`n_seq_max = 1`）。复用 context 保留已分配内存，每次请求前清空注意力 KV 和
循环状态。

`--engine-schedule off` 让 worker 直接调用引擎。启用 FIFO 或 decode-first 后，
`--prefill-chunk` 在独立 context 的实际 prefill 分块和 decode 步骤之间提供调度点。
引擎调用串行交错，一次 `llama_batch` 仍只含当前请求的一条序列，KV 分配沿用
llama.cpp。采样和 socket 输出由请求 worker 处理。

`--policy fifo|deadline` 决定 HTTP 等待队列的顺序。`--engine-schedule fifo|decode-first`
在请求取得 worker 后决定引擎调用顺序。等待调度许可时可取消。开始执行的调用在
同步完成后才归还许可。

### 采样与输出容量

直接贪心 argmax 借用 logits 数组，根据 CPU 能力选择 AVX2 或标量实现，支持
temperature 为零。旧实现与上游 sampler 路径可以用于对照。

单行输出使用 llama.cpp 的输出容量参数。适配层只读单序列调用的最后一行 logits。
将来引入多序列批处理时，需要重新验证这个假设。

`max_tokens` 和音色 ID 等整数字段只接受取值在范围内的 JSON 整数。温度和语速
接受 JSON 数值，布尔值会被拒绝。

### 实测数据与版本

[采样](../docs/experiments/perf-002-sampling.md)、[GPU 分析](../docs/experiments/perf-003-gpu.md)、
[单 worker 容量](../docs/experiments/serving-capacity.md)和[调度实验](../docs/experiments/prefill-decode.md)
各自记录运行配置。历史 llama-server slot 数据对应上游二进制，KVHarbor 的
单 worker 服务数据见单独的报告。

实验保留采集时的源码与二进制标识。后续代码变更不会重写历史数据的版本标识。
样本量、未完成的 SSE 流和端到端退化也记录在报告里。

### 取消与资源归还

reactor 负责请求准入与终态。运行中发生断连或超时时，服务设置共享取消标记、
停止响应管道并唤醒阻塞的生产者。worker 在引擎同步完成后返回，reactor 归还
名额、递增归还计数并写入唯一终态。取消排队中的请求只移除队列条目，开始与
归还计数均保持原值。

`accepted = terminal + active + queued` 校验请求总数。
`started = slot_releases + active` 校验 worker 名额。容量测试脚本检查两条等式，
覆盖超时请求未取得 worker 的情况。

引擎调度的许可与 worker 名额独立。取消等待时移除排队票据，执行中的调用离开
作用域时归还许可。采样与网络背压可继续占用 worker 名额。

### 如何阅读性能数据

采样实验的阶段耗时中位数从每 token 461.42 微秒降到 107.61 微秒。HTTP 完成时间中位数
从 1,632.85 毫秒降到 1,623.92 毫秒。端到端还涵盖排队、prefill、矩阵向量计算、同步和传输。

decode-first 在四组成对实验中降低了 TPOT p95，SLO goodput 持平或下降。首 token
延迟退化的样本也保存在报告中。该策略由参数开启。

## 开发

源码入口包括 [HTTP 传输](../serving/src/server.cc)、[任务准入](../serving/src/admission.cc)
和[运行时接口](../serving/include/kvharbor/serving/runtime.h)。
README 的四张架构图共用[绘图源文件](../scripts/render_readme_architecture.py)。

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
python3 tests/test_stream_bench.py
python3 tests/test_capacity_bench.py
```
