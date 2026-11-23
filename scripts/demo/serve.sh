#!/usr/bin/env bash
# Run one mixed text and speech service, dropping the old GPU model on a switch.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd -- "$root"
port=${KVHARBOR_DEMO_PORT:-18082}

usage() {
  cat <<'HELP'
bash scripts/demo/serve.sh [--llama|--trt]
用 KVHARBOR_DEMO_PORT 指定端口，默认 18082。在客户端输入 /engine llama 或 /engine trt 切换。
Ctrl+C 停止演示。日志与生成的音频保存在 .local/demo/runs/。
HELP
}
fail() { printf '%s\n' "$*" >&2; exit 1; }
if (( $# > 1 )); then usage >&2; exit 2; fi
case ${1:---llama} in
  -h|--help) usage; exit 0 ;;
  --llama) engine=llama ;;
  --trt) engine=trt ;;
  *) usage >&2; exit 2 ;;
esac
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || fail '端口必须在 1 至 65535 之间。'
for tool in curl docker flock ss python3; do
  command -v "$tool" >/dev/null || fail "缺少命令 $tool"
done
export KVHARBOR_AUDIO=1
export KVHARBOR_EDGE_SDK=${KVHARBOR_EDGE_SDK:-"$root/.cache/kvharbor/edge-sdk-v0.12.0-sm120"}
export KVHARBOR_SERVING_BUILD_DIR=${KVHARBOR_SERVING_BUILD_DIR:-"$root/build-serving-262k-edge"}
export GGUF_MODEL=${GGUF_MODEL:-"$root/models/Qwen3.8-27B-Q4_K_M.gguf"}
asr_model=models/audio/qwen3-asr-0.6b-int8
tts_model=models/audio/kokoro-v1.0
[[ -x $KVHARBOR_SERVING_BUILD_DIR/kvharbor-server ]] || fail '请按部署指南构建启用文本与语音的演示服务。'
[[ -r $KVHARBOR_EDGE_SDK/lib/libNvInfer_edgellm_plugin.so ]] || fail '找不到 Edge-LLM SDK，请检查 KVHARBOR_EDGE_SDK。'
[[ -d $asr_model && -d $tts_model ]] || fail '请先准备 Qwen3-ASR 和 Kokoro。'
check_engine() {
  if [[ $1 == llama ]]; then
    [[ -s $GGUF_MODEL ]] || { echo "找不到 GGUF：$GGUF_MODEL" >&2; return 1; }
  else
    [[ -s models/qwen3.5-0.8b-edge/llm.engine && -s models/qwen3.5-0.8b/tokenizer.json ]] ||
      { echo '请先按 TensorRT 部署指南准备引擎和 checkpoint。' >&2; return 1; }
  fi
}
check_engine "$engine"
umask 077
runs="$root/.local/demo/runs"
mkdir -p -- "$runs"
# A stable inode lets new launchers share a single advisory lock.
exec 9>"$runs/port-$port.lock"
flock -n 9 || fail "端口 $port 已由另一个演示启动器管理。"
# A container left by an earlier run keeps the port until it is stopped. Only
# stop containers that run this project's server on this exact port.
stop_stale_container() {
  local name args
  while read -r name; do
    [[ -n $name ]] || continue
    args=$(docker inspect --format '{{join .Args " "}}' "$name" 2>/dev/null) || continue
    [[ $args == *kvharbor-server* && $args == *"--port $port"* ]] || continue
    printf '停止占用端口 %s 的残留容器 %s\n' "$port" "$name"
    docker stop -t 15 "$name" >/dev/null 2>&1 || true
  done < <(docker ps --format '{{.Names}}' 2>/dev/null || true)
}
if [[ -n $(ss -ltnH "sport = :$port") ]]; then
  stop_stale_container
  for _ in {1..20}; do
    [[ -n $(ss -ltnH "sport = :$port") ]] || break
    sleep 0.25
  done
fi
listeners=$(ss -ltnH "sport = :$port")
[[ -z $listeners ]] || fail "端口 $port 已被非本项目的进程占用，请选择空闲的 KVHARBOR_DEMO_PORT。"
export KVHARBOR_OUTPUT_DIR
KVHARBOR_OUTPUT_DIR=$(mktemp -d "$runs/serve.XXXXXX")
export KVHARBOR_NATIVE_NAME="kvharbor-demo-${KVHARBOR_OUTPUT_DIR##*/}"
control="$runs/port-$port"
url="http://127.0.0.1:$port"
launcher_pid=
log=
stop_service() {
  if [[ -n $launcher_pid ]]; then
    if kill -0 "$launcher_pid" 2>/dev/null; then
      docker stop -t 15 "$KVHARBOR_NATIVE_NAME" >/dev/null 2>&1 ||
        kill -TERM "$launcher_pid" 2>/dev/null || true
    fi
    wait "$launcher_pid" 2>/dev/null || true
    launcher_pid=
  fi
}
cleanup() {
  local status=$?
  trap - EXIT
  stop_service
  if [[ -p $control ]]; then rm -f -- "$control"; fi
  printf '\n服务已退出。日志目录 %s\n' "$KVHARBOR_OUTPUT_DIR"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Only the supervisor holding the port lock may publish or remove this FIFO.
if [[ -p $control ]]; then rm -- "$control"; fi
mkfifo -- "$control"
exec 8<>"$control"
start_service() {
  local selected=$1 backend model started_at
  local -a args
  if [[ $selected == llama ]]; then
    backend=llama.cpp
    model=Qwen3.8-27B-Q4_K_M
    args=(--context 155648 --max-output-tokens 16384 --batch 512 --threads 4 --gpu-layers 99
          --reuse-context --sampling direct --max-output-rows 1 --kv-cache-q8 --mtp-draft-tokens 3)
  else
    backend=tensorrt-edge-llm
    model=Qwen3.5-0.8B
    args=(--engine-dir /workspace/models/qwen3.5-0.8b-edge --checkpoint-dir /workspace/models/qwen3.5-0.8b
          --context 2048 --max-output-tokens 1024)
  fi
  log="$KVHARBOR_OUTPUT_DIR/$selected.log"
  printf '正在加载 %s，日志 %s\n' "$model" "$log"
  bash scripts/native_serving.sh serve --bind 127.0.0.1 --port "$port" --workers 1 --queue 8 \
    --timeout-ms 180000 --text-backend "$backend" --model-id "$model" "${args[@]}" \
    --asr-model-dir "/workspace/$asr_model" --tts-model-dir "/workspace/$tts_model" \
    8>&- 9>&- >>"$log" 2>&1 &
  launcher_pid=$!
  started_at=$SECONDS
  while kill -0 "$launcher_pid" 2>/dev/null && (( SECONDS - started_at < 120 )); do
    if curl --noproxy '*' -fsS --max-time 2 "$url/health" 2>/dev/null |
      python3 -c 'import json,sys; h=json.load(sys.stdin); r=h["runtimes"]; sys.exit(not (h["status"]=="ok" and r["llm"]["backend"]==sys.argv[1] and "asr" in r and "tts" in r))' "$backend" 2>/dev/null; then
      printf '服务已就绪 %s · %s\n' "$url" "$backend"
      return 0
    fi
    sleep 0.4
  done
  tail -n 12 "$log" >&2
  stop_service
  return 1
}
start_service "$engine" || fail '演示服务启动失败。'
printf '另开 Bash 执行 python3 -B scripts/demo/client.py --port %s\n' "$port"
while kill -0 "$launcher_pid" 2>/dev/null; do
  if ! IFS=' ' read -r -t 1 target token extra <&8; then continue; fi
  [[ -z $extra && $token =~ ^[a-f0-9]{32}$ && ( $target == llama || $target == trt ) ]] || continue
  reply="$runs/reply-$token"
  [[ -p $reply && -O $reply ]] || continue
  answer=ok
  if [[ $target != "$engine" ]]; then
    if ! check_engine "$target"; then
      answer='缺少目标模型，当前引擎仍可用。'
    else
      stop_service
      if start_service "$target"; then
        engine=$target
      elif start_service "$engine"; then
        answer='目标引擎启动失败，已恢复原引擎。'
      else
        answer='两个引擎均未能启动，请查看服务日志。'
      fi
    fi
  fi
  # The client may leave mid-switch. Bound opening/writing its acknowledgement.
  timeout 2 bash -c 'printf "%s\n" "$1" > "$2"' bash "$answer" "$reply" 8>&- 9>&- || true
  [[ -n $launcher_pid ]] || exit 1
done
wait "$launcher_pid"
