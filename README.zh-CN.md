<h1 align="center">KVHarbor</h1>

<p align="center"><strong>面向单卡的 C++ 文本与语音推理服务</strong></p>

<p align="center">
  <a href="https://github.com/Vitolearns/KVHarbor/actions/workflows/serving.yml"><img src="https://github.com/Vitolearns/KVHarbor/actions/workflows/serving.yml/badge.svg?branch=main" alt="C++ serving CI"></a>
  <a href="serving/CMakeLists.txt"><img src="assets/readme/badge-cpp.svg" alt="C++20"></a>
  <a href="LICENSE"><img src="assets/readme/badge-license.svg" alt="PolyForm Noncommercial 1.0.0"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> · <a href="#运行真实模型">运行真实模型</a> · <a href="#架构">架构</a> · <a href="#单卡实测">单卡实测</a> · <a href="README.md" lang="en">English</a>
</p>

KVHarbor 是面向单张 NVIDIA GPU 的 C++20 HTTP 推理服务。文本生成由
llama.cpp 在 CUDA 上执行，语音识别和合成由 sherpa-onnx 在 CPU 上执行。
从请求进入队列到 worker 归还，服务层负责准入、截止时间、取消和流式背压。

## 快速开始

在 Ubuntu 24.04 上构建，支持原生 Linux 和 WSL2。示例用 CPU 测试后端跑通一次 HTTP/SSE
请求，只需安装编译依赖。

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 build-essential cmake libboost-dev nlohmann-json3-dev
git clone https://github.com/Vitolearns/KVHarbor.git
cd KVHarbor
cmake -S serving -B build-serving-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-serving-cpu --parallel 2
./build-serving-cpu/kvharbor-server --test-backend --port 18082
```

另开一个终端，发送流式请求。

```bash
curl -fsSN http://127.0.0.1:18082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":4}'
```

请求会返回 4 个占位数据块，并以 `data: [DONE]` 结束。用 Ctrl+C 停止服务。
需要实际生成文本、转写或合成语音时，参见[真实模型运行指南](#运行真实模型)。

## 运行真实模型

参考环境为 Ubuntu 24.04（原生 Linux 或 WSL2）、RTX 5090 Laptop GPU，
CUDA 编译目标为 SM120。主机需要 Bash、Git、curl、Python 3.12+、可访问
NVIDIA GPU 的 Docker，以及 `libboost-dev` 和 `nlohmann-json3-dev`。
以下命令在仓库根目录执行。

### CUDA 文本生成

<details>
<summary>构建、启动服务并发送请求</summary>

[文本运行时清单](configs/runtime/llama.json)记录了权重来源、SHA-256、
llama.cpp 版本和 CUDA 镜像。将 GGUF 下载到 `models/`，核对校验和后构建。

```bash
export GGUF_MODEL="$PWD/models/Qwen3.8-27B-Q4_K_M.gguf"
sha256sum "$GGUF_MODEL"

image=$(python3 -c 'import json; print(json.load(open("configs/runtime/llama.json"))["build_image"])')
docker pull "$image"
python3 scripts/llama_runtime.py build --jobs 4
bash scripts/native_serving.sh build
```

以前台模式启动服务。

```bash
bash scripts/native_serving.sh serve --workers 1 --context 2048 --batch 512 \
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

</details>

### 启用语音

<details>
<summary>准备 Qwen3-ASR 和 Kokoro，再试用语音接口</summary>

停止文本服务并保留 `GGUF_MODEL`。[语音运行时清单](configs/runtime/audio.json)
指定了 sherpa-onnx、Qwen3-ASR 和 Kokoro 的版本与校验和。准备脚本会下载并校验
这些依赖，语音推理在 CPU 上运行。

```bash
python3 scripts/prepare_audio_runtime.py
KVHARBOR_AUDIO=1 bash scripts/native_serving.sh build
KVHARBOR_AUDIO=1 bash scripts/native_serving.sh serve \
  --workers 1 --context 2048 --batch 512 --threads 4 \
  --reuse-context --sampling direct --max-output-rows 1 \
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

</details>

## 架构

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/architecture-dark.svg">
  <img src="assets/readme/architecture-light.svg" width="780" alt="HTTP 请求入口、有界任务队列以及文本和语音 worker。">
</picture>
</p>

- **请求生命周期。** 各任务有独立的有界队列和 worker 名额，流式响应也有缓冲上限。
  排队中的请求可立即取消。已经执行的请求要等后端调用返回并完成同步，才能归还 worker。
- **两级调度。** FIFO 或 deadline 决定队列中的请求何时取得 worker。
  可选的 FIFO 或 decode-first 调度器在 prefill 分块和 decode 的调用边界
  协调多个独立的 llama.cpp context。
- **运行时状态。** 文本 worker 共享模型权重，每个 worker 持有一条独立序列，
  两次请求之间清空 KV 和循环状态。

[Graphviz 拓扑源文件](assets/readme/architecture.dot)

<details>
<summary>实现细节与代码来源</summary>

[NOTICE.md](NOTICE.md)记录了继承材料与署名。[文本](configs/runtime/llama.json)
和[语音](configs/runtime/audio.json)运行时清单列出了上游引擎、模型版本和校验和。

| 模块 | 负责的工作 |
| --- | --- |
| llama.cpp | 分词、模型执行、量化 CUDA kernel 和模型状态分配。KVHarbor 在进程内调用 C API。 |
| sherpa-onnx | ASR 与 TTS。Qwen3-ASR 做离线识别，Kokoro 按句回调音频。 |
| [HTTP 服务](serving/src/server.cc) | 异步连接、有界任务队列与输出缓冲、截止时间、取消和终态统计。 |
| [文本适配层](serving/src/llama_runtime.cc) | worker 独占 context、分块 prefill、贪心采样选择、请求状态重置和实际引擎调用调度。 |
| [引擎调度器](serving/src/engine_schedule.h) | 串行协调多个 worker 的引擎调用。FIFO 按到达顺序执行。decode-first 在 prefill 等待时最多连续放行四次 decode，然后选最早的 prefill。调度许可涵盖 decode 与同步，由作用域守卫归还。 |
| [容量测试脚本](scripts/capacity_bench.py) | 开环到达、客户端排队、SLO goodput、尾延迟、拒绝与未完成流计数、源码与二进制标识、GPU 观测。 |

### Context 与调度

每个文本 worker 独占一个请求，直到原生调用返回。一个 context 只包含一条序列
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

[采样](docs/experiments/perf-002-sampling.md)、[GPU 分析](docs/experiments/perf-003-gpu.md)、
[单 worker 容量](docs/experiments/serving-capacity.md)和[调度实验](docs/experiments/prefill-decode.md)
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

</details>

## 单卡实测

实验使用单张 RTX 5090 Laptop GPU 和 Qwen3.8-27B Q4_K_M。负载、配置和
统计口径见各实验报告。

| 实验 | 实测结果 | 复现材料 |
| --- | --- | --- |
| 服务容量与恢复 | 共 1,152 次测量请求。在 0.4 req/s 下，单 worker 三轮分别有 128/128、126/128、128/128 个请求达标。 | [方法](docs/experiments/serving-acceptance.md) · [逐请求数据](results/experiments/serving-acceptance.json) · [源码快照](results/experiments/serving-acceptance-source.tar.gz) |
| prefill/decode 交错 | 四组成对实验中，decode-first 使每个输出 token 的耗时 p95 降低 9.0%–19.0%，SLO goodput 持平或下降。 | [方法](docs/experiments/prefill-decode.md) · [逐请求数据](results/experiments/prefill-decode.json) |
| 单行 logits 缓冲 | CUDA 工作区从 507.02 MiB 降至 124.02 MiB。HTTP 完成时间中位数增加约 5 ms。 | [GPU 分析](docs/experiments/perf-003-gpu.md) |
| 直接贪心采样 | 采样耗时中位数下降约 76.7%，HTTP 完成时间中位数下降约 0.55%。 | [采样实验](docs/experiments/perf-002-sampling.md) |

容量实验的 SLO 要求请求从到达到首 token ≤ 2 秒，每个输出 token 的耗时
≤ 75 毫秒。在受控崩溃测试中，HTTP 在 7.48 秒内恢复健康。

## API

| 任务 | 接口 | 请求与响应 |
| --- | --- | --- |
| 文本生成 | `POST /v1/chat/completions`、`POST /v1/completions` | JSON 请求，返回 JSON 或 SSE |
| 语音识别 | `POST /v1/audio/transcriptions` | 单声道 PCM16 WAV 请求，返回 JSON 转写结果 |
| 语音合成 | `POST /v1/audio/speech` | JSON 请求，返回裸 PCM16，响应头给出采样率 |
| 健康与指标 | `GET /health`、`GET /state`、`GET /metrics` | 健康状态、任务占用和 Prometheus 指标 |

[语音运行命令](#启用语音) · [Kokoro WAV 样例](assets/readme/ferry.wav)

## 开发

可以从 [HTTP 服务](serving/src/server.cc)、[引擎调度器](serving/src/engine_schedule.h)
或 [llama.cpp 适配层](serving/src/llama_runtime.cc) 开始读。以下测试使用快速
开始里的 CPU 构建。

```bash
ctest --test-dir build-serving-cpu --output-on-failure
python3 serving/tests/test_http.py build-serving-cpu/kvharbor-server
python3 tests/test_stream_bench.py
python3 tests/test_capacity_bench.py
```

## 许可

KVHarbor 中受仓库许可覆盖的贡献采用 [PolyForm Noncommercial 1.0.0](LICENSE)。
继承代码、第三方依赖与模型的适用条款见 [NOTICE](NOTICE.md)。
