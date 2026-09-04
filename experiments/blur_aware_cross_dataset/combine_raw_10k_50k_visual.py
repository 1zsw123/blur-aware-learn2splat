#!/usr/bin/env python3
"""Combine matched two-column renders into a RAW/10K/50K comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HEADER_HEIGHT = 52


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ten-k", type=Path, required=True)
    parser.add_argument("--fifty-k", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ten_k = Image.open(args.ten_k).convert("RGB")
    fifty_k = Image.open(args.fifty_k).convert("RGB")
    if ten_k.size != fifty_k.size or ten_k.width % 2:
        raise ValueError(f"incompatible source images: {ten_k.size}, {fifty_k.size}")
    candidate = None
    if args.candidate is not None:
        candidate = Image.open(args.candidate).convert("RGB")
        if candidate.size != ten_k.size:
            raise ValueError(
                f"candidate image has incompatible size: {candidate.size}, {ten_k.size}"
            )

    width = ten_k.width // 2
    body_height = ten_k.height - HEADER_HEIGHT
    labels = ["RAW blurred input", "Ours at 10K", "Ours at 50K"]
    sources = [ten_k, ten_k, fifty_k]
    source_columns = [0, 1, 1]
    if candidate is not None:
        labels.append(args.candidate_label)
        sources.append(candidate)
        source_columns.append(1)
    canvas = Image.new(
        "RGB", (len(labels) * width, HEADER_HEIGHT + body_height), "white"
    )
    for column, (source, source_column) in enumerate(zip(sources, source_columns)):
        left = source_column * width
        canvas.paste(
            source.crop((left, HEADER_HEIGHT, left + width, source.height)),
            (column * width, HEADER_HEIGHT),
        )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for column, label in enumerate(labels):
        draw.text((column * width + 12, 13), label, fill="black", font=font)
    draw.line((0, HEADER_HEIGHT - 1, canvas.width, HEADER_HEIGHT - 1), fill="black")
    for column in range(1, len(labels)):
        x = column * width
        draw.line((x, 0, x, canvas.height), fill="#888888")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
