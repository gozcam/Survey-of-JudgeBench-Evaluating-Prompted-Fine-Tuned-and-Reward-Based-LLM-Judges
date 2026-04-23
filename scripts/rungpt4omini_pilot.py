"""Run a JudgeBench pilot using GPT-4o-mini on a 10-pair subset of the GPT-4o response pairs.

The script:
  1. Ensures a 10-pair subset of the GPT-4o JudgeBench dataset exists (creating it if not).
  2. Invokes the JudgeBench `run_judge.py` command against that subset using
     the Arena-Hard judge powered by `gpt-4o-mini`, using the OpenAI API key
     exported as `OPENAI_API_KEY` in the environment.
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
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUN_JUDGE_SCRIPT),
        "--judge_name", "arena_hard",
        "--judge_model", "gpt-4o-mini",
        "--pairs", str(subset_path),
        "--concurrency_limit", "5",
    ]
    print(f"Running: {' '.join(cmd)} (cwd={REPO_ROOT})")
    print(f"Outputs will be written to: {output_dir}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode

def main() -> int:
    subset_path = ensure_subset()
    return run_judgebench(subset_path)

if __name__ == "__main__":
    sys.exit(main())