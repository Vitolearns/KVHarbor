# 部署指南

[项目首页](../README.zh-CN.md) · [English](serving.md)

## 构建准备

环境为 Ubuntu 24.04 或 WSL2，需要 Bash、Git、curl、Python 3.12+、
可访问 NVIDIA GPU 的 Docker、`libboost-dev` 和 `nlohmann-json3-dev`。
参考 GPU 为 RTX 5090 Laptop，锁定的构建目标是 SM120。

从仓库根目录执行命令，先构建共用的 CUDA 依赖。
服务以前台模式运行，另开终端发送请求，切换后端前按 Ctrl+C 停止服务。

```bash
image=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["build_image"])')
docker pull "$image"
python3 scripts/llama_runtime.py build --jobs 4
```

## CUDA 文本生成

将锁定版本的 Qwen3.8-27B Q4_K_M GGUF 保存到 `models/`，校验后构建并启动服务。
[模型下载](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/resolve/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf) · [运行时清单](../configs/runtime/llama.json)

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_AUDIO KVHARBOR_SERVING_BUILD_DIR
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha256=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["model"]["sha256"])')
printf '%s  %s\n' "$model_sha256" "$GGUF_MODEL" | sha256sum --check -
bash scripts/native_serving.sh build
bash scripts/native_serving.sh serve --text-backend llama.cpp \
  --workers 1 --context 2048 --batch 512 \
  --threads 4 --gpu-layers 99 --reuse-context --sampling direct --max-output-rows 1 \
  --queue 8 --timeout-ms 8000 --policy fifo
```

检查健康状态并发送请求。完整 SSE 响应以 `data: [DONE]` 结束。

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsSN http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Q4_K_M","stream":true,"temperature":0,"max_tokens":48,
       "messages":[{"role":"user","content":"What is 17 + 25? Reply with the number only."}]}'
```

## TensorRT Edge-LLM 文本生成

按 [TensorRT 部署流程](tensorrt.zh-CN.md)准备 SDK、导出环境和引擎，再启动
Qwen3.5-0.8B。参考引擎的输入上限为 1,024 token，KV 容量为 2,048 token。

## 启用语音

准备脚本校验 sherpa-onnx、Qwen3-ASR 和 Kokoro。
用 `--text-backend none` 单独运行语音。混合服务在构建和启动时设置
`KVHARBOR_AUDIO=1`，并在文本启动命令后加上两项语音模型目录参数。
[运行时清单](../configs/runtime/audio.json)

```bash
unset KVHARBOR_EDGE_SDK KVHARBOR_SERVING_BUILD_DIR
export KVHARBOR_AUDIO=1
python3 scripts/prepare_audio_runtime.py
bash scripts/native_serving.sh build
bash scripts/native_serving.sh serve --text-backend none \
  --asr-model-dir /workspace/models/audio/qwen3-asr-0.6b-int8 \
  --tts-model-dir /workspace/models/audio/kokoro-v1.0
```

转写仓库内的 WAV，或合成 PCM16。输出采样率见 `X-Audio-Sample-Rate`。

```bash
curl -fsS http://127.0.0.1:8082/v1/audio/transcriptions \
  -H 'Content-Type: audio/wav' --data-binary @assets/readme/ferry.wav
curl -fsS http://127.0.0.1:8082/v1/audio/speech \
  -H 'Content-Type: application/json' -d '{"input":"See you at the pier."}' \
  -D speech.headers -o speech.pcm
```

ASR、TTS 与 llama.cpp 混合运行时，先停止上面的纯语音服务，在同一终端设置
GGUF 路径并启动文本后端。复用刚构建的语音版本。

```bash
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
bash scripts/native_serving.sh serve --text-backend llama.cpp --context 2048 \
  --workers 1 --reuse-context --max-output-rows 1 --timeout-ms 180000 \
  --asr-model-dir /workspace/models/audio/qwen3-asr-0.6b-int8 \
  --tts-model-dir /workspace/models/audio/kokoro-v1.0
```

## 交互演示

先准备 GGUF、[Edge 引擎与 SDK](tensorrt.zh-CN.md) 和上文的语音模型，再构建同时支持两个文本后端和语音的服务。

```bash
export KVHARBOR_AUDIO=1
export KVHARBOR_EDGE_SDK="$PWD/.cache/kvharbor/edge-sdk-v0.12.0-sm120"
export KVHARBOR_SERVING_BUILD_DIR="$PWD/build-serving-262k-edge"
bash scripts/native_serving.sh build
bash scripts/demo/serve.sh --llama
```

另开终端执行 `python3 -B scripts/demo/client.py`。启动参数 `--trt` 可直接加载 TensorRT。
默认端口是 18082，也可以在两个终端设置相同的 `KVHARBOR_DEMO_PORT` 来使用其他空闲端口。按 Ctrl+C 停止启动器。

| 客户端输入 | 演示内容 |
| --- | --- |
| 直接提问，或输入 `/engine` | 查看流式回答，或显示当前文本后端。 |
| `/engine trt`、`/engine llama` | 在同一块 GPU 和同一端口重新加载模型。更换引擎后清空文本历史，并读取新的输出上限。 |
| `/tts See you at the pier.`、`/play` | 合成 WAV，播放最近一次合成的音频。 |
| `/asr`、`/asr assets/readme/ferry.wav` | 转写上次合成的音频，或指定的单声道 PCM16 WAV。 |
| `/mix`、`/status` | 同时发起文本、转写和合成请求，或查看各任务的 worker 和队列。 |

llama.cpp 演示默认使用 196,608 个 token 的上下文，单轮最多输出 16,384 个 token。MTP 投机解码每步最多提议 3 个 token，需要 GGUF 包含 MTP 预测层。在 24 GB RTX 5090 Laptop GPU 上，开启 MTP 的代码生成测试最低剩余显存为 10 MiB。TensorRT 的上下文和单轮输出上限分别为 2,048 和 1,024 个 token。日志和 WAV 保存在 `.local/demo/runs/`。

## API

| 接口 | 请求 → 响应 |
| --- | --- |
| `POST /v1/chat/completions`、`POST /v1/completions` | 含 `stream: true`、`temperature: 0` 的 JSON → SSE |
| `POST /v1/audio/transcriptions` | 单声道 PCM16 WAV → JSON 转写 |
| `POST /v1/audio/speech` | JSON 文本 → PCM16，响应头标注采样率 |
| `GET /health`、`/state`、`/metrics`、`/v1/models` | 运行状态、任务占用、Prometheus 指标和已加载模型 |

`max_tokens` 和音色 ID 等整数字段使用 JSON 整数。
`--max-output-tokens` 设置输出上限，`--timeout-ms` 设置请求时限，
两项配置均显示在 `/health` 中。

## 实现细节

<a name="context-与调度"></a>
<a name="采样与输出容量"></a>
<a name="取消与资源归还"></a>

| 机制 | 行为 |
| --- | --- |
| Worker 名额 | 每个 llama.cpp worker 独占一个单序列 context，直到原生执行结束。 |
| `--reuse-context` | 复用内存分配，每次请求前清空注意力 KV 和循环状态。 |
| `--policy fifo\|deadline` | 决定等待中的 HTTP 请求顺序。 |
| `--engine-schedule fifo\|decode-first` | 在独立 context 间串行交错 prefill/decode，等待许可时可取消。 |
| 采样 | 贪心生成。直接 argmax 借用 logits，选择 AVX2 或标量实现。单行输出读取最后一行 logits。 |
| 长上下文 | llama.cpp 最多接受 262,144 个 context token，`--kv-cache-q8` 启用 Q8_0 KV 与 flash attention。Edge-LLM 容量由已构建的引擎决定。 |

断连或超时会设置取消标记、停止响应管道并唤醒阻塞的生产者。
运行中的 worker 等引擎同步完成后归还名额，排队取消则移除队列条目。
请求与 worker 账目分别满足 `accepted = terminal + active + queued`
和 `started = slot_releases + active`。

<a name="实测数据与版本"></a>
<a name="如何阅读性能数据"></a>

每份实验记录对应的源码、二进制、负载和测量范围。
阶段耗时与端到端结果见[采样实验](experiments/perf-002-sampling.md)和
[调度实验](experiments/prefill-decode.md)。

## 开发

完成 [CPU 构建](../README.zh-CN.md#快速开始)后执行以下检查。
源码入口为 [HTTP 传输](../serving/src/server.cc)、[任务准入](../serving/src/admission.cc)
和[运行时接口](../serving/include/kvharbor/serving/runtime.h)。

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
python3 tests/test_stream_bench.py
python3 tests/test_capacity_bench.py
```

[README 绘图源文件](../scripts/render_readme_architecture.py) · [署名与依赖条款](../NOTICE.md)
