#!/usr/bin/env python3
"""Render the README architecture diagram in both languages and themes."""

from argparse import ArgumentParser
from html import escape
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "readme"

COPY = {
    "en": {
        "title": "KVHarbor serving architecture",
        "description": (
            "One kvharbor-server process routes HTTP requests through a shared "
            "scheduler with separate queues for text generation, speech recognition "
            "and speech synthesis. Runtime adapters call the selected text engine or "
            "sherpa-onnx on CPU. The HTTP API writes SSE, JSON or PCM16 responses."
        ),
        "client": "HTTP client",
        "process": "Single C++20 process",
        "request": "HTTP request",
        "response": "HTTP response",
        "http": "HTTP API",
        "http_detail": "Routes requests · Writes SSE, JSON or PCM16",
        "typed_request": "Typed request",
        "runtime_output": "Results and chunks",
        "scheduler": "Admission and scheduling",
        "scheduler_detail": (
            "Bounded queues per task · Deadlines · Cancellation · Backpressure"
        ),
        "text": "Text adapter",
        "asr": "ASR adapter",
        "tts": "TTS adapter",
        "choose": "Choose one at startup",
    },
    "zh-CN": {
        "title": "KVHarbor 服务架构",
        "description": (
            "同一个 kvharbor-server 进程经共享调度将请求分到文本生成、语音识别和语音合成队列。"
            "各运行时适配器调用选定的文本引擎或 CPU 上的 sherpa-onnx。"
            "HTTP 接口负责返回 SSE、JSON 或 PCM16。"
        ),
        "client": "HTTP 客户端",
        "process": "单个 C++20 进程",
        "request": "HTTP 请求",
        "response": "HTTP 响应",
        "http": "HTTP 接口",
        "http_detail": "路由请求 · 返回 SSE、JSON 或 PCM16",
        "typed_request": "类型化请求",
        "runtime_output": "结果和数据块",
        "scheduler": "请求准入与调度",
        "scheduler_detail": "分任务有界队列 · 截止时间 · 取消 · 背压",
        "text": "文本适配器",
        "asr": "语音识别适配器",
        "tts": "语音合成适配器",
        "choose": "启动时二选一",
    },
}

PALETTES = {
    "light": {
        "canvas": "#ffffff", "process": "#f6f8fa", "panel": "#ffffff",
        "scheduler": "#eaf3ff", "border": "#8c959f", "ink": "#1f2328",
        "secondary": "#424a53", "request": "#0969da", "response": "#57606a",
    },
    "dark": {
        "canvas": "#0d1117", "process": "#161b22", "panel": "#0d1117",
        "scheduler": "#13273d", "border": "#687481", "ink": "#e6edf3",
        "secondary": "#aab6c3", "request": "#79c0ff", "response": "#9da7b3",
    },
}


def render(language, colors):
    label = {name: escape(value) for name, value in COPY[language].items()}
    c = colors
    adapters = "\n".join(
        f'  <rect class="panel" x="{x - 104}" y="296" width="208" height="38" rx="5"/>\n'
        f'  <text class="node" x="{x}" y="321" text-anchor="middle">{label[task]}</text>'
        for x, task in ((138, "text"), (360, "asr"), (582, "tts"))
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 448" width="720" height="448"
     xml:lang="{language}" role="img" aria-labelledby="diagram-title diagram-description">
  <title id="diagram-title">{label['title']}</title>
  <desc id="diagram-description">{label['description']}</desc>
  <style>
    text {{
      font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      fill: {c['ink']};
    }}
    .panel {{ fill: {c['panel']}; stroke: {c['border']}; stroke-width: 1.25; }}
    .node {{ font-size: 17px; font-weight: 650; }}
    .detail {{ font-size: 14px; fill: {c['secondary']}; }}
    .mono {{ font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; }}
    .engine {{ font-size: 15px; font-weight: 650; }}
    .device {{ font-size: 12px; fill: {c['secondary']}; text-anchor: end; }}
  </style>
  <defs>
    <marker id="request-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8"
            markerUnits="userSpaceOnUse" orient="auto">
      <path d="M0 0L8 4 0 8Z" fill="{c['request']}"/>
    </marker>
    <marker id="response-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8"
            markerUnits="userSpaceOnUse" orient="auto">
      <path d="M0 0L8 4 0 8Z" fill="{c['response']}"/>
    </marker>
  </defs>

  <rect width="720" height="448" fill="{c['canvas']}"/>
  <rect x="12" y="66" width="696" height="370" rx="8"
        fill="{c['process']}" stroke="{c['border']}" stroke-width="1.25"/>
  <text class="mono" x="28" y="89" font-size="15" font-weight="700">kvharbor-server</text>
  <text class="detail" x="692" y="89" text-anchor="end">{label['process']}</text>

  <rect class="panel" x="282" y="8" width="156" height="32" rx="5"/>
  <rect class="panel" x="150" y="101" width="420" height="57" rx="5"/>
  <rect x="106" y="198" width="508" height="57" rx="5"
        fill="{c['scheduler']}" stroke="{c['request']}" stroke-width="1.25"/>
  <g text-anchor="middle">
    <text class="node" x="360" y="30">{label['client']}</text>
    <text class="node" x="360" y="125">{label['http']}</text>
    <text class="detail" x="360" y="146">{label['http_detail']}</text>
    <text class="node" x="360" y="222">{label['scheduler']}</text>
    <text class="detail" x="360" y="243">{label['scheduler_detail']}</text>
  </g>

{adapters}

  <rect class="panel" x="34" y="355" width="208" height="65" rx="5"/>
  <rect class="panel" x="256" y="355" width="208" height="65" rx="5"/>
  <rect class="panel" x="478" y="355" width="208" height="65" rx="5"/>

  <text class="detail" x="45" y="373">{label['choose']}</text>
  <text class="engine" x="45" y="393">llama.cpp</text>
  <text class="engine" x="45" y="412">TensorRT Edge-LLM</text>
  <text class="engine" x="267" y="382">sherpa-onnx</text>
  <text class="detail" x="267" y="405">Qwen3-ASR</text>
  <text class="engine" x="489" y="382">sherpa-onnx</text>
  <text class="detail" x="489" y="405">Kokoro</text>
  <g class="mono device">
    <text x="231" y="393">CPU / CUDA</text>
    <text x="231" y="412">CUDA</text>
    <text x="453" y="382">CPU</text>
    <text x="675" y="382">CPU</text>
  </g>

  <g fill="none" stroke="{c['request']}" stroke-width="1.6" stroke-linejoin="round">
    <path d="M360 255V274M138 274H582"/>
    <g marker-end="url(#request-head)">
      <path d="M347 40V101"/>
      <path d="M347 158V198"/>
      <path d="M138 274V296"/>
      <path d="M360 274V296"/>
      <path d="M582 274V296"/>
      <path d="M138 334V355"/>
      <path d="M360 334V355"/>
      <path d="M582 334V355"/>
    </g>
  </g>
  <g fill="none" stroke="{c['response']}" stroke-width="1.6" stroke-dasharray="4 3"
     marker-end="url(#response-head)">
    <path d="M373 101V40"/>
    <path d="M373 198V158"/>
  </g>
  <g class="detail">
    <text x="335" y="59" text-anchor="end">{label['request']}</text>
    <text x="385" y="59">{label['response']}</text>
    <text x="335" y="182" text-anchor="end">{label['typed_request']}</text>
    <text x="385" y="182">{label['runtime_output']}</text>
  </g>
</svg>
'''


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in SVGs differ")
    args = parser.parse_args()
    for language, suffix in (("en", ""), ("zh-CN", "-zh")):
        for theme, colors in PALETTES.items():
            path = ASSETS / f"architecture{suffix}-{theme}.svg"
            svg = render(language, colors)
            if args.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != svg:
                    parser.exit(1, f"Regenerate {path.name}\n")
            else:
                path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
