#!/usr/bin/env python3
"""Render the README architecture diagram in both languages and themes."""

from argparse import ArgumentParser
from html import escape
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "readme"

COPY = {
    "en": (
        "KVHarbor serving architecture",
        "The C++20 HTTP server routes each task to its own queue and workers. "
        "Text uses one GPU backend selected at startup and returns SSE. "
        "CPU transcription returns JSON and CPU synthesis streams PCM16.",
        "HTTP API · C++20", "Admission · Deadlines", "Cancellation · Backpressure",
        "Queues & workers", "Inference backend", "HTTP response",
        "Generate text", "Transcribe", "Synthesize", "GPU · choose one at startup",
        "Text stream", "Transcript", "Audio stream",
    ),
    "zh-CN": (
        "KVHarbor 服务架构",
        "C++20 HTTP 服务把请求分到各自的任务队列和工作线程。"
        "文本生成在启动时选择一个 GPU 后端，返回 SSE 文本流。"
        "CPU 语音识别返回 JSON，CPU 语音合成返回 PCM16 音频流。",
        "HTTP 接口 · C++20", "请求准入 · 截止时间", "请求取消 · 输出背压",
        "队列与工作线程", "推理后端", "HTTP 响应",
        "文本生成", "语音识别", "语音合成", "GPU · 启动时二选一",
        "文本流", "转写结果", "音频流",
    ),
}

PALETTES = {
    "light": {
        "canvas": "#eef4f2", "frame": "#bbccc8", "header": "#14363a",
        "head_text": "#f1faf7", "head_detail": "#bde4d9", "head_line": "#658a85",
        "icon_bg": "#285353", "icon_ink": "#a9ead6",
        "text_lane": "#daede5", "text_border": "#accbc1",
        "audio_lane": "#f8fbfa", "audio_border": "#c9d8d6",
        "ink": "#193739", "secondary": "#466361", "wire": "#517a73",
        "accent": "#165648",
    },
    "dark": {
        "canvas": "#141f24", "frame": "#3d555b", "header": "#203a39",
        "head_text": "#f1faf7", "head_detail": "#bbe6dc", "head_line": "#6a918b",
        "icon_bg": "#2e5350", "icon_ink": "#b2f3dd",
        "text_lane": "#233b36", "text_border": "#456f63",
        "audio_lane": "#1c2c33", "audio_border": "#3b5259",
        "ink": "#e8f3f0", "secondary": "#b0c6c4", "wire": "#86b5ac",
        "accent": "#a9ead1",
    },
}


def render(language, colors):
    (title, description, http, admission, cancellation, queues, inference, response,
     generate, transcribe, synthesize, gpu, text_stream, transcript, audio_stream) = (
        escape(label) for label in COPY[language]
    )
    c = colors
    # Each horizontal lane starts at the same HTTP router and ends at its own response.
    lanes = (
        (200, 127, 264, c["text_lane"], c["text_border"]),
        (341, 70, 376, c["audio_lane"], c["audio_border"]),
        (425, 70, 460, c["audio_lane"], c["audio_border"]),
    )
    cards = "\n".join(
        f'  <rect x="67" y="{top}" width="628" height="{height}" rx="15" '
        f'fill="{fill}" stroke="{border}"/>'
        for top, height, _, fill, border in lanes
    )
    arrows = "\n".join(
        f'    <path d="M35 {center - 12}q0 12 12 12h17m-5-5 5 5-5 5 '
        f'M255 {center}h23m-5-5 5 5-5 5 M517 {center}h23m-5-5 5 5-5 5"/>'
        for _, _, center, _, _ in lanes
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 520" width="720" height="520"
     xml:lang="{language}" role="img" aria-labelledby="diagram-title diagram-description" font-family="system-ui, -apple-system, Segoe UI, PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif">
  <title id="diagram-title">{title}</title>
  <desc id="diagram-description">{description}</desc>
  <rect x=".5" y=".5" width="719" height="519" rx="24" fill="{c['canvas']}" stroke="{c['frame']}"/>
  <path d="M24 1H696a23 23 0 0 1 23 23v107H1V24A23 23 0 0 1 24 1Z" fill="{c['header']}"/>
  <rect x="27" y="37" width="57" height="57" rx="15" fill="{c['icon_bg']}"/>
  <path d="M40 65h9m0 0v-10q0-4 4-4h16M49 65h20M49 65v10q0 4 4 4h16
           M65 47l4 4-4 4m0 6 4 4-4 4m0 6 4 4-4 4"
        fill="none" stroke="{c['icon_ink']}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="101" y="63" fill="{c['head_text']}" font-size="37" font-weight="750" letter-spacing="-.9">KVHarbor</text>
  <text x="103" y="91" fill="{c['head_detail']}" font-size="21" font-weight="550">{http}</text>
  <path d="M365 38v56" fill="none" stroke="{c['head_line']}"/>
  <g fill="{c['head_text']}" font-size="21" font-weight="550">
    <text x="393" y="59">{admission}</text>
    <text x="393" y="86">{cancellation}</text>
  </g>

  <g fill="{c['secondary']}" font-size="19" font-weight="650">
    <text x="84" y="176">{queues}</text>
    <text x="296" y="176">{inference}</text>
    <text x="557" y="176">{response}</text>
  </g>
{cards}
  <g fill="none" stroke="{c['wire']}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M35 133v315"/>
{arrows}
  </g>
  <circle cx="35" cy="131" r="4.5" fill="{c['wire']}"/>

  <g fill="{c['ink']}" font-size="23" font-weight="680">
    <text x="84" y="272">{generate}</text>
    <text x="84" y="384">{transcribe}</text>
    <text x="84" y="468">{synthesize}</text>
  </g>
  <text x="296" y="226" fill="{c['accent']}" font-size="18" font-weight="650">{gpu}</text>
  <path d="M288 264h9m0-9h11m-11 0v40h11" fill="none" stroke="{c['wire']}" stroke-width="1.5" stroke-linecap="round"/>
  <g fill="{c['ink']}" font-size="21" font-weight="680">
    <text x="319" y="262">llama.cpp</text>
    <text x="319" y="302">TensorRT Edge-LLM</text>
    <text x="296" y="369">Qwen3-ASR</text>
    <text x="296" y="453">Kokoro</text>
  </g>
  <g fill="{c['secondary']}" font-size="18">
    <text x="296" y="393">sherpa-onnx · CPU</text>
    <text x="296" y="477">sherpa-onnx · CPU</text>
    <text x="557" y="288">{text_stream}</text>
    <text x="557" y="393">{transcript}</text>
    <text x="557" y="477">{audio_stream}</text>
  </g>
  <g fill="{c['accent']}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="22" font-weight="700">
    <text x="557" y="264">SSE</text>
    <text x="557" y="369">JSON</text>
    <text x="557" y="453">PCM16</text>
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
