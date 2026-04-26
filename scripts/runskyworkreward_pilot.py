"""Run a JudgeBench pilot using Skywork-Reward-Llama-3.1-8B on a 10-pair subset.

The reward model runs locally via HuggingFace transformers and requires a CUDA-capable
GPU. No API key is needed, but the extra dependencies must be installed first:

    pip install -r third_party/judgebench/requirements-cuda.txt

The script:
  1. Ensures a 10-pair subset of the GPT-4o JudgeBench dataset exists (creating it if not).
  2. Invokes the JudgeBench `run_judge.py` command against that subset using
     judge_name=reward_model and judge_model=Skywork/Skywork-Reward-Llama-3.1-8B.
  3. Passes --single_game because reward models score each response independently
     and are not sensitive to ordering.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGEBENCH_DIR = REPO_ROOT / "third_party" / "judgebench"
RUN_JUDGE_SCRIPT = JUDGEBENCH_DIR / "run_judge.py"
FULL_DATASET = JUDGEBENCH_DIR / "data" / "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
SUBSET_PATH = REPO_ROOT / "data" / "dataset=judgebench-pilot10,response_model=gpt-4o-2024-05-13.jsonl"
SUBSET_SIZE = 10
SEED = 42

JUDGE_NAME = "reward_model"
JUDGE_MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B"


def ensure_subset() -> Path:
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SUBSET_PATH.exists():
        print(f"Subset already exists at {SUBSET_PATH}, reusing it.")
        return SUBSET_PATH

    if not FULL_DATASET.exists():
        raise FileNotFoundError(f"Expected GPT-4o JudgeBench dataset at {FULL_DATASET}")

    with FULL_DATASET.open("r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    if len(pairs) < SUBSET_SIZE:
        raise ValueError(f"Dataset only has {len(pairs)} pairs, cannot sample {SUBSET_SIZE}.")

    rng = random.Random(SEED)
    subset = rng.sample(pairs, SUBSET_SIZE)

    with SUBSET_PATH.open("w", encoding="utf-8") as f:
        for pair in subset:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote {SUBSET_SIZE}-pair subset to {SUBSET_PATH}.")
    return SUBSET_PATH


def run_judgebench(subset_path: Path) -> int:
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("NOTE: This reward model runs locally and requires a CUDA-capable GPU.")
    print("      Ensure you have installed: pip install -r third_party/judgebench/requirements-cuda.txt")
    print(f"      Model will be downloaded from HuggingFace on first run: {JUDGE_MODEL}")
    print()

    cmd = [
        sys.executable,
        str(RUN_JUDGE_SCRIPT),
        "--judge_name", JUDGE_NAME,
        "--judge_model", JUDGE_MODEL,
        "--single_game",
        "--pairs", str(subset_path),
    ]
    print(f"Running: {' '.join(cmd)} (cwd={REPO_ROOT})")
    print(f"Outputs will be written to: {output_dir}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    subset_path = ensure_subset()
    return run_judgebench(subset_path)


if __name__ == "__main__":
    sys.exit(main())
