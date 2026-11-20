<h1 align="center">KVHarbor</h1>

<p align="center">
  <a href="#快速开始">快速开始</a> · <a href="#运行真实模型">运行真实模型</a> · <a href="#架构">架构</a> · <a href="#单卡实测">单卡实测</a> · <a href="README.md">English</a>
</p>

KVHarbor 用一个 C++20 进程提供文本生成、语音识别和语音合成。文本在 GPU 上运行，
启动时可选 **llama.cpp** 或 **TensorRT Edge-LLM**。**Qwen3-ASR** 和 **Kokoro**
在 CPU 上运行，可以接到同一个服务里，也可以单独启动。

## 快速开始

在 Ubuntu 24.04 或 WSL2 上，用 CPU 测试后端先发一条流式请求。这个例子无需下载模型权重。

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 build-essential cmake libboost-dev nlohmann-json3-dev
git clone https://github.com/Vitolearns/KVHarbor.git
cd KVHarbor
cmake -S serving -B build-serving-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-serving-cpu --parallel 2
./build-serving-cpu/kvharbor-server --test-backend --port 18082
```

服务启动后，在另一个终端发送请求。

```bash
curl -fsSN http://127.0.0.1:18082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":4}'
```

响应包含 4 个内容块，最后一条 SSE 记录是 `data: [DONE]`。用 Ctrl+C 停止服务。

## 运行真实模型

<a name="cuda-文本生成"></a>
<a name="tensorrt-edge-llm-文本生成"></a>
<a name="启用语音"></a>

| 想运行什么 | 引擎 | 启动说明 |
| --- | --- | --- |
| GGUF 文本生成 | llama.cpp（GPU） | [启动 Qwen3.8-27B](docs/serving.zh-CN.md#cuda-文本生成) |
| TensorRT 文本生成 | TensorRT Edge-LLM（GPU） | [启动 Qwen3.5-0.8B](docs/serving.zh-CN.md#tensorrt-edge-llm-文本生成) |
| 语音转写和合成 | sherpa-onnx（CPU） | [启动 Qwen3-ASR 和 Kokoro](docs/serving.zh-CN.md#启用语音) |

## 架构

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/architecture-dark.svg">
  <img src="assets/readme/architecture-light.svg" width="960" alt="HTTP 入口、分任务队列、运行时适配、推理引擎和响应格式">
</picture>

文本、识别和合成各用自己的队列和 worker。客户端断开时，服务取消请求并唤醒等待写出的
worker。已经开始的推理要等原生调用收尾，才能把名额交给下一个请求。
[请求生命周期和引擎细节](docs/serving.zh-CN.md#实现细节)

## 单卡实测

以下数据来自 RTX 5090 Laptop GPU 上的 Qwen3.8-27B Q4_K_M。

- [服务容量实验](docs/experiments/serving-acceptance.md)记录了三档速率下的 1,152 次请求到达，
  并单独验证了过载与恢复。
- [单行 logits 实验](docs/experiments/perf-003-gpu.md)将 CUDA 工作区从 507 MiB 降到
  124 MiB，HTTP 完成延迟中位数增加约 5 ms。

其他实验见 [prefill/decode 交错](docs/experiments/prefill-decode.md)和
[贪心采样](docs/experiments/perf-002-sampling.md)。

## API

| 任务 | 接口 | 请求与响应 |
| --- | --- | --- |
| 文本生成 | `POST /v1/chat/completions`、`POST /v1/completions` | JSON 请求使用 `stream: true` 和 `temperature: 0`，成功时返回 SSE。 |
| 语音识别 | `POST /v1/audio/transcriptions` | 单声道 PCM16 WAV 请求，返回 JSON 转写结果 |
| 语音合成 | `POST /v1/audio/speech` | JSON 请求，返回裸 PCM16，响应头给出采样率 |
| 服务状态 | `GET /health`、`GET /state`、`GET /metrics`、`GET /v1/models` | 运行状态、任务占用、Prometheus 指标和已加载模型 |

## 开发

用快速开始的构建跑服务回归。源码入口和压测脚本检查见[开发说明](docs/serving.zh-CN.md#开发)。

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
```

## 许可

KVHarbor 中受仓库许可覆盖的贡献采用 [PolyForm Noncommercial 1.0.0](LICENSE)。
继承代码、第三方依赖与模型的适用条款见 [NOTICE](NOTICE.md)。
