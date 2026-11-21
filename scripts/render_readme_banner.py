#!/usr/bin/env python3
"""Render the English and Chinese README banners."""

from argparse import ArgumentParser
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "readme"


def render(language, subtitle, description, tasks):
    styles = (
        (72, "SSE", "#21433e", "#47776b", "#a5ebd1"),
        (134, "JSON", "#1d3137", "#355158", "#b1d1ce"),
        (196, "PCM16", "#1d3137", "#355158", "#b1d1ce"),
    )
    rows = "\n".join(
        f'''  <rect x="600" y="{center - 22}" width="206" height="44" rx="9" fill="{fill}" stroke="{stroke}"/>
  <circle cx="619" cy="{center}" r="3.5" fill="{ink}"/>
  <text x="633" y="{center + 7}" fill="#e9f5f3" font-size="22" font-weight="650">{task}</text>
  <text x="791" y="{center + 6}" fill="{ink}" text-anchor="end" font-size="20" font-weight="650" font-family="ui-monospace, SFMono-Regular, Consolas, monospace">{protocol}</text>'''
        for task, (center, protocol, fill, stroke, ink) in zip(tasks, styles)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 268" width="860" height="268"
     xml:lang="{language}" role="img" aria-labelledby="banner-title banner-description"
     font-family="system-ui, -apple-system, Segoe UI, PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif">
  <title id="banner-title">KVHarbor</title>
  <desc id="banner-description">{description}</desc>
  <defs>
    <pattern id="grid" width="22" height="22" patternUnits="userSpaceOnUse" x="435" y="30">
      <path d="M22 0H0v22" fill="none" stroke="#294449"/>
    </pattern>
  </defs>
  <rect x=".5" y=".5" width="859" height="267" rx="21" fill="#121f2b" stroke="#2f4750"/>
  <text x="35" y="103" fill="#f2f9f8" font-size="65" font-weight="750" letter-spacing="-2.4">KVHarbor</text>
  <text x="39" y="143" fill="#c2d7df" font-size="25" font-weight="550">{subtitle}</text>
  <path d="M39 167H391" stroke="#3b5862"/>
  <text x="39" y="207" fill="#9ee5d0" font-size="22" font-weight="650">llama.cpp / TensorRT Edge-LLM</text>

  <rect x="435" y="29" width="394" height="210" rx="14" fill="#162a32"/>
  <rect x="435" y="29" width="394" height="210" rx="14" fill="url(#grid)" stroke="#3a555e"/>
  <path d="M538 134h57m-31 0V83q0-11 11-11h20m-31 62v51q0 11 11 11h20
           M590 67l5 5-5 5m0 52 5 5-5 5m0 52 5 5-5 5"
        fill="none" stroke="#77beae" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="564" cy="134" r="4" fill="#9ee5d0" stroke="#162a32" stroke-width="1.5"/>
  <rect x="456" y="112" width="82" height="44" rx="9" fill="#203f43" stroke="#51837b"/>
  <text x="497" y="142" fill="#bdf1e3" text-anchor="middle" font-size="23" font-weight="700"
        font-family="ui-monospace, SFMono-Regular, Consolas, monospace">HTTP</text>
{rows}
</svg>
'''


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in SVGs differ")
    args = parser.parse_args()
    for filename, language, subtitle, description, tasks in (
        ("banner.svg", "en", "Single-GPU inference service",
         "A C++20 inference service. The HTTP API streams text via SSE, "
         "returns transcripts as JSON, and streams PCM16 speech.", ("Text", "ASR", "TTS")),
        ("banner-zh.svg", "zh-CN", "单卡推理服务",
         "C++20 推理服务。HTTP 接口返回 SSE 文本流、JSON 转写和 PCM16 音频流。",
         ("文本生成", "语音识别", "语音合成")),
    ):
        path = ASSETS / filename
        svg = render(language, subtitle, description, tasks)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != svg:
                parser.exit(1, f"Regenerate {filename}\n")
        else:
            path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
