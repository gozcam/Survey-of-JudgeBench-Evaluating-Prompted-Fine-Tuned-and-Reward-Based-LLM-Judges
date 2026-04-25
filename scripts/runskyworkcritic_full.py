"""Run JudgeBench on the full GPT-4o response-pair dataset using Skywork-Critic-Llama-3.1-8B.

The model is served locally via a vLLM OpenAI-compatible server on localhost:8000.
No API key is required; ensure the vLLM server is running before executing this script.

Outputs will be written to REPO_ROOT / "outputs".
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGEBENCH_DIR = REPO_ROOT / "third_party" / "judgebench"
RUN_JUDGE_SCRIPT = JUDGEBENCH_DIR / "run_judge.py"
FULL_DATASET = JUDGEBENCH_DIR / "data" / "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"

JUDGE_NAME = "skywork_critic"
JUDGE_MODEL = "Skywork/Skywork-Critic-Llama-3.1-8B"


def run_judgebench() -> int:
    if not RUN_JUDGE_SCRIPT.exists():
        print(f"ERROR: run_judge.py not found at {RUN_JUDGE_SCRIPT}", file=sys.stderr)
        return 1

    if not FULL_DATASET.exists():
        print(f"ERROR: Full dataset not found at {FULL_DATASET}", file=sys.stderr)
        return 1

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("NOTE: Ensure the vLLM server is running at http://localhost:8000 serving")
    print(f"      {JUDGE_MODEL} before proceeding.")

    cmd = [
        sys.executable,
        str(RUN_JUDGE_SCRIPT),
        "--judge_name", JUDGE_NAME,
        "--judge_model", JUDGE_MODEL,
        "--pairs", str(FULL_DATASET),
        "--concurrency_limit", "1",
    ]

    print(f"Running: {' '.join(cmd)} (cwd={REPO_ROOT})")
    print(f"Outputs will be written to: {output_dir}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    return run_judgebench()


if __name__ == "__main__":
    sys.exit(main())
