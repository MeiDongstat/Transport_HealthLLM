#!/usr/bin/env python3
"""Detect the primary language of each prompt in a HealthBench JSONL file."""

import argparse
import json
import re
from pathlib import Path

import fasttext


def prompt_text(prompt):
    """Convert a plain or chat-formatted prompt into text for language detection."""
    if isinstance(prompt, list):
        # Chat-formatted prompts store their text in the content field of each message.
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)


def output_path(src):
    """Derive the language-annotated output path from the input filename."""
    name = src.name
    # Preserve known evaluation variants; otherwise remove a date suffix from the stem.
    tag = (
        "consensus"
        if "consensus" in name
        else "hard"
        if "hard" in name
        else "oss"
        if "oss" in name
        else re.sub(r"[_-]?\d{4}.*", "", src.stem)
    )
    return src.with_name(f"healthbench_{tag}_lan.jsonl")


def main():
    """Add the predicted language and confidence to every input JSONL record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="2025-05-07-06-14-12_oss_eval.jsonl")
    parser.add_argument("-m", "--model", default=Path(__file__).with_name("lid.176.bin"))
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    src = Path(args.input)
    # Resolve a bare default filename beside this script when it is not in the working directory.
    if not src.exists():
        src = Path(__file__).with_name(args.input)
    dst = Path(args.output) if args.output else output_path(src)
    model = fasttext.load_model(str(args.model))

    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            # fastText language identification expects one prompt per physical line.
            labels, scores = model.predict(prompt_text(row.get("prompt", "")).replace("\n", " "), k=1)
            # Store the normalized language code and fastText's prediction probability.
            row["language"] = labels[0].removeprefix("__label__")
            row["confidence"] = float(scores[0])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dst)


if __name__ == "__main__":
    main()


# python3 HealthLLM_transfer/code/representation/healthbench_language_id.py \
#  HealthLLM_transfer/data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl \
#  -o HealthLLM_transfer/data/healthbench/healthbench_oss_lan.jsonl
