#!/usr/bin/env python3
"""Run JudgeBench on a 10-pair subset of the GPT-4o dataset using Gemini Flash Lite.

Reads the user's Gemini API key from the ``GEMINI_API_KEY`` environment variable,
builds a deterministic 10-pair subset of the JudgeBench GPT-4o response pairs if
one does not already exist, and then invokes ``run_judge.py`` to evaluate the
subset with Gemini Flash Lite as the judge model.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGEBENCH_DIR = REPO_ROOT / "third_party" / "judgebench"
RUN_JUDGE_SCRIPT = JUDGEBENCH_DIR / "run_judge.py"
DATA_DIR = JUDGEBENCH_DIR / "data"
FULL_DATASET = DATA_DIR / "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
SUBSET_DATASET = REPO_ROOT / "data" / "dataset=judgebench-pilot10,response_model=gpt-4o-2024-05-13.jsonl"

JUDGE_NAME = os.environ.get("JUDGE_NAME", "arena_hard")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash-lite")
NUM_PAIRS = 10
SEED = 42
CONCURRENCY_LIMIT = os.environ.get("CONCURRENCY_LIMIT", "5")

def build_subset() -> None:
    SUBSET_DATASET.parent.mkdir(parents=True, exist_ok=True)

    if SUBSET_DATASET.exists():
        print(f"[pilot] Subset already exists: {SUBSET_DATASET}")
        return

    if not FULL_DATASET.exists():
        sys.exit(f"[pilot] Missing source dataset: {FULL_DATASET}")

    with FULL_DATASET.open("r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    subset = pairs[:NUM_PAIRS]

    with SUBSET_DATASET.open("w", encoding="utf-8") as f:
        for pair in subset:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[pilot] Wrote {len(subset)} pairs to {SUBSET_DATASET}")

def run_judgebench() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(
            "[pilot] GEMINI_API_KEY is not set. Export it before running, e.g.:\n"
            "    export GEMINI_API_KEY=your-key"
        )

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUN_JUDGE_SCRIPT),
        "--judge_name", JUDGE_NAME,
        "--judge_model", JUDGE_MODEL,
        "--pairs", str(SUBSET_DATASET),
        "--concurrency_limit", str(CONCURRENCY_LIMIT),
    ]

    print(f"[pilot] Running: {' '.join(cmd)} (cwd={REPO_ROOT})")
    print(f"[pilot] Outputs will be written to: {output_dir}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

if __name__ == "__main__":
    build_subset()
    run_judgebench()