#!/usr/bin/env python3
"""Interactive KVHarbor demonstration for a locally running service."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import select
import stat
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / ".local" / "demo" / "runs"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audio_bench import asr, tts
from stream_bench import SSEDecoder, get_json


COMMANDS = """直接输入问题，模型会边生成边显示。
用 ↑ 和 ↓ 浏览本次会话的输入历史。
/tts [文本]  合成任意中英文短句。单独输入 /tts 后可输入新文本，直接回车则用最近一条模型回答。
/play        播放最近一次合成的 WAV。
/asr [WAV]   转写指定的 WAV。省略路径会使用最近一次合成的音频。
/mix         同时发送文本、识别和合成请求，显示三项结果。
/engine      显示当前后端。用 /engine llama 或 /engine trt 在本端口切换。
/status      展示三类 worker 的活跃、排队和名额。
/reset       清空文本对话。
/quit        退出交互。
"""
SYSTEM_MESSAGE = {"role": "system", "content": "先给出结论，再按需要简要解释。"}


def stream_reply(url, model, messages, max_tokens, timeout):
    body = json.dumps(
        {"model": model, "messages": messages, "stream": True,
         "temperature": 0, "max_tokens": max_tokens},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    decoder = SSEDecoder()
    chunks = []
    start = time.perf_counter()
    first_ms = None
    finish_reason = None
    print("模型> ", end="", flush=True)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError("文本接口没有返回 SSE。")
        while line := response.readline(65536):
            for event in decoder.feed(line):
                if event == "[DONE]":
                    if first_ms is None:
                        raise ValueError("模型返回了空回答。")
                    total_ms = (time.perf_counter() - start) * 1000
                    ending = ""
                    if finish_reason == "length":
                        ending = f" · 达到生成上限（{max_tokens} tokens）"
                    print(f"\n首段 {first_ms:.0f} ms · 全程 {total_ms:.0f} ms · 流结束 [DONE]{ending}")
                    return "".join(chunks)
                if not event:
                    continue
                payload = json.loads(event)
                if error := payload.get("error"):
                    raise ValueError(f"文本流错误 {error}")
                choice = (payload.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if first_ms is None:
                        first_ms = (time.perf_counter() - start) * 1000
                    chunks.append(content)
                    print(content, end="", flush=True)
    raise ValueError("文本流在 [DONE] 之前结束。")


def synthesize(url, text, *, announce=True):
    if not text.strip():
        raise ValueError("请输入要合成的文本。")
    if len(text.encode("utf-8")) > 600:
        raise ValueError("文本超过当前语音接口的 600 字节上限，请输入较短的句子。")
    RUNS.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="voice.", dir=RUNS))
    wav_path = folder / "speech.wav"
    # These are the Mandarin and English voices used by the repo's audio fixtures.
    voice = 45 if re.search(r"[\u3400-\u9fff]", text) else 2
    if announce:
        print(f"TTS 输入> {text}")
    result = tts(url, text, voice, wav_path)
    if announce:
        report_speech(result, wav_path)
    return wav_path, result


def report_speech(result, wav_path):
    print(f"TTS 输出> {result['sample_rate']} Hz · {result['duration_s']:.2f} 秒")
    print(f"WAV  {wav_path}")


def play_wav(wav_path):
    player = shutil.which("ffplay")
    if player:
        result = subprocess.run(
            [player, "-nodisp", "-autoexit", "-loglevel", "error", str(wav_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and not result.stderr.strip():
            return
    if shutil.which("explorer.exe") and shutil.which("wslpath"):
        windows_path = subprocess.check_output(
            ["wslpath", "-w", str(wav_path)], text=True,
        ).strip()
        subprocess.Popen(
            ["explorer.exe", windows_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"已交给 Windows 播放器  {windows_path}")
        return
    raise ValueError(f"无法启动播放器。音频文件在 {wav_path}。")


def transcribe(url, wav_path):
    with wave.open(str(wav_path), "rb") as audio:
        if audio.getnframes() > 30 * audio.getframerate():
            raise ValueError("这段 WAV 超过 ASR 的 30 秒上限，请合成较短的文本。")
    print(f"ASR 输入> {wav_path}")
    text = asr(url, wav_path)["text"]
    print(f"ASR 输出> {text}")
    return text


def show_status(url):
    state = get_json(url, "/state")
    for task, worker in state["tasks"].items():
        print(f"{task.upper()}  活跃 {worker['active']}  排队 {worker['queued']}  名额 {worker['slots']}")
    print(f"累计执行 {state['started']}  归还 {state['slot_releases']}")


ENGINES = {"llama": "llama.cpp", "trt": "tensorrt-edge-llm"}


def load_service(url, requested):
    health = get_json(url, "/health")
    runtimes = health["runtimes"]
    for task in ("llm", "asr", "tts"):
        print(f"{task.upper()}  {runtimes[task]['model']}  {runtimes[task]['backend']}")
    llm = runtimes["llm"]
    if drafts := llm.get("mtp_draft_tokens", 0):
        print(f"MTP  每轮最多提议 {drafts} 个 token")
    limit = health.get("max_output_tokens", 0)
    budget = requested or limit or 384
    if limit:
        budget = min(budget, limit)
    print(f"上下文 {llm['context']:,} · 当前输出预算 {budget:,} · 服务上限 {limit or '未设置'} tokens")
    return llm, budget, health.get("request_timeout_ms", 30000) / 1000 + 5


def switch_engine(port, target):
    if target not in ENGINES:
        raise ValueError("用法：/engine llama 或 /engine trt")
    control = RUNS / f"port-{port}"
    try:
        writer = os.open(control, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("此端口没有演示启动器。请使用 scripts/demo/serve.sh 启动。") from error
    try:
        if not stat.S_ISFIFO(os.fstat(writer).st_mode):
            raise ValueError("无效的演示控制管道。")
        token = uuid4().hex
        reply = RUNS / f"reply-{token}"
        os.mkfifo(reply, 0o600)
        try:
            reader = os.open(reply, os.O_RDWR | os.O_NONBLOCK)
            try:
                os.write(writer, f"{target} {token}\n".encode("ascii"))
                print(f"正在切换到 {ENGINES[target]}…", flush=True)
                if not select.select([reader], [], [], 300)[0]:
                    raise ValueError("等待演示启动器超时。请用 /engine 核实当前引擎。")
                result = os.read(reader, 4096).decode().strip()
            finally:
                os.close(reader)
        finally:
            reply.unlink(missing_ok=True)
    finally:
        os.close(writer)
    if result != "ok":
        raise ValueError(result or "演示启动器未返回结果。")


def mixed_demo(url, model, max_tokens, timeout):
    wav = REPO_ROOT / "assets" / "readme" / "ferry.wav"
    prompt = [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}]
    with ThreadPoolExecutor(max_workers=3) as pool:
        text = pool.submit(stream_reply, url, model, prompt, min(48, max_tokens), timeout)
        transcript = pool.submit(asr, url, wav)
        speech = pool.submit(synthesize, url, "See you at the pier.", announce=False)
        text.result()
        print(f"ASR 输出> {transcript.result()['text']}")
        wav_path, result = speech.result()
        report_speech(result, wav_path)
    print("混合演示完成：LLM、ASR、TTS 均已返回。")
    return wav_path


def main():
    parser = argparse.ArgumentParser(description="KVHarbor 本机交互演示")
    parser.add_argument("--port", type=int, default=os.getenv("KVHARBOR_DEMO_PORT", "18082"))
    parser.add_argument("--max-tokens", type=int,
                        help="单轮输出预算，最高使用当前后端 /health 公布的上限")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1 至 65535 之间。")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("单轮输出 token 上限必须大于零。")
    if sys.stdin.isatty():
        try:
            import readline
        except ImportError:
            pass
        else:
            readline.set_auto_history(True)
    url = f"http://127.0.0.1:{args.port}"
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    try:
        llm, max_tokens, timeout = load_service(url, args.max_tokens)
    except (KeyError, OSError, ValueError, http.client.HTTPException) as error:
        parser.exit(1, f"{url} 上的演示服务尚未就绪。请先运行 serve.sh。\n{error}\n")
    print(COMMANDS)
    history = []
    last_wav = None
    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if "\x1b" in line:
            print("检测到终端按键序列，请重新输入。")
            continue
        command, _, argument = line.partition(" ")
        try:
            match command:
                case "/quit":
                    return
                case "/reset":
                    history.clear()
                    print("已清空文本对话。")
                case "/status":
                    show_status(url)
                case "/engine":
                    target = argument.strip()
                    if target:
                        previous = llm["backend"]
                        switch_engine(args.port, target)
                        llm, max_tokens, timeout = load_service(url, args.max_tokens)
                        if llm["backend"] != ENGINES[target]:
                            raise ValueError("健康检查返回的引擎与所选引擎不一致。")
                        if llm["backend"] != previous:
                            history.clear()
                            print("已切换引擎，文本对话已清空。")
                    else:
                        print(f"当前引擎 {get_json(url, '/health')['runtimes']['llm']['backend']}")
                case "/mix":
                    last_wav = mixed_demo(url, llm["model"], max_tokens, timeout)
                case "/tts":
                    text = argument.strip()
                    if not text:
                        text = input("要合成的文本（Enter 使用最近一条模型回答）> ").strip()
                    if not text and history:
                        text = history[-1]["content"]
                    last_wav, _ = synthesize(url, text)
                case "/asr":
                    wav = Path(argument.strip()).expanduser() if argument.strip() else last_wav
                    if wav is None:
                        raise ValueError("请提供 WAV 路径，或先用 /tts 生成音频。")
                    transcribe(url, wav)
                case "/play":
                    if last_wav is None:
                        raise ValueError("请先用 /tts 生成一段音频。")
                    play_wav(last_wav)
                case _:
                    if command.startswith("/"):
                        print(COMMANDS)
                        continue
                    question = {"role": "user", "content": line}
                    reply = stream_reply(
                        url, llm["model"], [SYSTEM_MESSAGE, *history, question], max_tokens, timeout,
                    )
                    history.extend((question, {"role": "assistant", "content": reply}))
        except EOFError:
            return
        except KeyboardInterrupt:
            print("\n当前操作已取消。")
        except urllib.error.HTTPError as error:
            with error:
                detail = error.read(4096).decode("utf-8", errors="replace")
            print(f"\n请求失败 HTTP {error.code}  {detail or error.reason}")
        except (OSError, ValueError, http.client.HTTPException, wave.Error) as error:
            print(f"\n请求失败  {error}")


if __name__ == "__main__":
    main()
