"""Analyze JudgeBench full-run output files.

Loads all full-run JSONL files from outputs/, computes per-category accuracy,
cross-model comparison, and extracts failures for qualitative inspection.

Saves to outputs/analysis/:
  summary.txt          - human-readable accuracy tables
  comparison.csv       - cross-model table (spreadsheet-friendly)
  failures_<model>.jsonl - failures per model (prompts stripped)
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"

CATEGORY_MAP = {
    "livebench-math": "livebench-math",
    "livebench-reasoning": "livebench-reasoning",
    "livecodebench": "livecodebench",
}

CATEGORIES = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", "Overall"]


def top_level_category(source: str) -> str:
    if source in CATEGORY_MAP:
        return CATEGORY_MAP[source]
    if source.startswith("mmlu-pro"):
        return "mmlu-pro"
    return "other"


def flip(decision: str) -> str:
    if decision == "A>B":
        return "B>A"
    if decision == "B>A":
        return "A>B"
    return decision


def score_pair(pair: dict) -> str:
    """Return 'correct', 'incorrect', or 'inconsistent' using reverse-order logic."""
    judgments = pair["judgments"]
    j1 = judgments[0] if len(judgments) > 0 else None
    j2 = judgments[1] if len(judgments) > 1 else None

    if j1 is None:
        return "incorrect"
    if j2 is None:
        d = j1["decision"]
        return "correct" if d == pair["label"] else "incorrect"

    decision1 = j1["decision"]
    decision2 = flip(j2["decision"])
    label = pair["label"]

    counter = 0
    for d in [decision1, decision2]:
        if d == label:
            counter += 1
        elif d == flip(label):
            counter -= 1

    if counter > 0:
        return "correct"
    if counter < 0:
        return "incorrect"
    return "inconsistent"


def load_output_file(path: Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(pairs: list[dict]) -> dict:
    """Return per-category counts and accuracy."""
    stats = {cat: {"correct": 0, "incorrect": 0, "inconsistent": 0} for cat in CATEGORIES}

    for pair in pairs:
        cat = top_level_category(pair["source"])
        outcome = score_pair(pair)
        if cat in stats:
            stats[cat][outcome] += 1
        stats["Overall"][outcome] += 1

    results = {}
    for cat, counts in stats.items():
        total = counts["correct"] + counts["incorrect"] + counts["inconsistent"]
        if total == 0:
            results[cat] = {"accuracy": None, **counts, "total": 0}
        else:
            acc = 100.0 * counts["correct"] / total
            results[cat] = {"accuracy": acc, **counts, "total": total}

    return results


def extract_failures(pairs: list[dict]) -> list[dict]:
    """Return failure records with prompts stripped."""
    failures = []
    for pair in pairs:
        outcome = score_pair(pair)
        if outcome == "correct":
            continue

        judgments = [j for j in pair["judgments"] if j is not None]
        decisions = [j["decision"] for j in judgments]
        reasoning_snippets = []
        for j in judgments:
            inner = j.get("judgment", {})
            resp = inner.get("response", "")
            reasoning_snippets.append(resp[:600] if resp else "")

        failures.append({
            "pair_id": pair["pair_id"],
            "source": pair["source"],
            "category": top_level_category(pair["source"]),
            "outcome": outcome,
            "label": pair["label"],
            "decisions": decisions,
            "question": pair["question"][:400],
            "judge_model": judgments[0].get("judgment", {}).get("judge_model", ""),
            "reasoning_1": reasoning_snippets[0] if reasoning_snippets else "",
            "reasoning_2": reasoning_snippets[1] if len(reasoning_snippets) > 1 else "",
        })

    return failures


def model_short_name(path: Path) -> str:
    stem = path.stem
    if "judge_model=" in stem:
        return stem.split("judge_model=")[1]
    return stem


def print_model_summary(model_name: str, metrics: dict) -> str:
    lines = [f"\n{'='*60}", f"  {model_name}", f"{'='*60}"]
    header = f"  {'Category':<24} {'Acc':>6}  {'Correct':>8} {'Wrong':>6} {'Incons':>7} {'Total':>6}"
    lines.append(header)
    lines.append("  " + "-" * 57)
    for cat in CATEGORIES:
        m = metrics[cat]
        if m["total"] == 0:
            lines.append(f"  {cat:<24} {'n/a':>6}  {'':>8} {'':>6} {'':>7} {0:>6}")
        else:
            acc = f"{m['accuracy']:.1f}%" if m["accuracy"] is not None else "n/a"
            lines.append(
                f"  {cat:<24} {acc:>6}  {m['correct']:>8} {m['incorrect']:>6} {m['inconsistent']:>7} {m['total']:>6}"
            )
    return "\n".join(lines)


def _rate(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total > 0 else "n/a"


def print_comparison_table(model_metrics: dict[str, dict]) -> str:
    models = list(model_metrics.keys())
    lines = [f"\n{'='*60}", "  Cross-model accuracy comparison", f"{'='*60}"]

    col_w = 14
    header = f"  {'Category':<24}" + "".join(f"  {m[:col_w]:>{col_w}}" for m in models)
    lines.append(header)
    lines.append("  " + "-" * (24 + (col_w + 2) * len(models)))

    for cat in CATEGORIES:
        row = f"  {cat:<24}"
        for model in models:
            m = model_metrics[model].get(cat, {})
            row += f"  {_rate(m.get('correct', 0), m.get('total', 0)):>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def print_outcome_comparison_tables(model_metrics: dict[str, dict]) -> str:
    models = list(model_metrics.keys())
    col_w = 14
    sep = "  " + "-" * (24 + (col_w + 2) * len(models))
    header = f"  {'Category':<24}" + "".join(f"  {m[:col_w]:>{col_w}}" for m in models)

    def make_table(title: str, key: str) -> list[str]:
        lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}", header, sep]
        for cat in CATEGORIES:
            row = f"  {cat:<24}"
            for model in models:
                m = model_metrics[model].get(cat, {})
                row += f"  {_rate(m.get(key, 0), m.get('total', 0)):>{col_w}}"
            lines.append(row)
        return lines

    out = make_table("Incorrect rate by category (% of all pairs)", "incorrect")
    out += make_table("Inconsistency rate by category (% of all pairs)", "inconsistent")
    return "\n".join(out)


def print_failure_mode_table(model_name: str, metrics: dict) -> str:
    lines = [
        f"\n{'='*60}",
        f"  {model_name} — failure modes by category",
        f"{'='*60}",
        f"  {'Category':<24} {'Incorrect':>10} {'Incons':>7} {'Failures':>9} {'Total':>6}"
        f"  {'Incorr%':>8} {'Incons%':>8} {'Fail%':>6}",
        "  " + "-" * 83,
    ]
    for cat in CATEGORIES:
        m = metrics[cat]
        total = m["total"]
        inc = m["incorrect"]
        ins = m["inconsistent"]
        fail = inc + ins
        if total == 0:
            lines.append(f"  {cat:<24} {'':>10} {'':>7} {'':>9} {0:>6}  {'n/a':>8} {'n/a':>8} {'n/a':>6}")
        else:
            lines.append(
                f"  {cat:<24} {inc:>10} {ins:>7} {fail:>9} {total:>6}"
                f"  {_rate(inc, total):>8} {_rate(ins, total):>8} {_rate(fail, total):>6}"
            )
    return "\n".join(lines)


def sample_failures(failures: list[dict], n: int = 2) -> str:
    if not failures:
        return ""
    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for f in failures:
        by_cat[f["category"]].append(f)

    lines = []
    for cat in sorted(by_cat):
        sample = by_cat[cat][:n]
        lines.append(f"\n  --- {cat} ({len(by_cat[cat])} failures, showing {len(sample)}) ---")
        for f in sample:
            q = f["question"].replace("\n", " ")[:200]
            lines.append(f"    pair_id : {f['pair_id']}")
            lines.append(f"    label   : {f['label']}  |  decisions: {f['decisions']}  |  outcome: {f['outcome']}")
            lines.append(f"    question: {q}")
            snippet = f["reasoning_1"][:300].replace("\n", " ")
            lines.append(f"    judge(1): {snippet}")
            lines.append("")
    return "\n".join(lines)


def save_comparison_csv(model_metrics: dict[str, dict], path: Path) -> None:
    """Save one CSV with accuracy, incorrect rate, and inconsistency rate per model/category."""
    models = list(model_metrics.keys())
    col_headers = []
    for model in models:
        col_headers += [f"{model}_accuracy", f"{model}_incorrect_rate", f"{model}_inconsistency_rate"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category"] + col_headers)
        for cat in CATEGORIES:
            row = [cat]
            for model in models:
                m = model_metrics[model].get(cat, {})
                total = m.get("total", 0)
                if total == 0:
                    row += ["", "", ""]
                else:
                    row += [
                        f"{m['accuracy']:.2f}",
                        f"{100.0 * m['incorrect'] / total:.2f}",
                        f"{100.0 * m['inconsistent'] / total:.2f}",
                    ]
            writer.writerow(row)


def main() -> int:
    # collect full-run output files (exclude pilot subsets)
    output_files = sorted(
        p for p in OUTPUTS_DIR.glob("dataset=judgebench,*.jsonl")
        if "pilot" not in p.name
    )

    if not output_files:
        print("No full-run output files found in outputs/.", file=sys.stderr)
        return 1

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    model_metrics: dict[str, dict] = {}
    model_failures: dict[str, list] = {}

    print(f"Found {len(output_files)} full-run output file(s):")
    for p in output_files:
        print(f"  {p.name}")

    for path in output_files:
        model = model_short_name(path)
        pairs = load_output_file(path)
        metrics = compute_metrics(pairs)
        failures = extract_failures(pairs)
        model_metrics[model] = metrics
        model_failures[model] = failures

    # build text summary
    summary_parts = ["JudgeBench Output Analysis", "=" * 60]

    for model in model_metrics:
        summary_parts.append(print_model_summary(model, model_metrics[model]))

    summary_parts.append(print_comparison_table(model_metrics))
    summary_parts.append(print_outcome_comparison_tables(model_metrics))

    summary_parts.append(f"\n{'='*60}\n  Failure mode breakdown by category\n{'='*60}")
    for model in model_metrics:
        summary_parts.append(print_failure_mode_table(model, model_metrics[model]))

    summary_parts.append(f"\n{'='*60}\n  Representative failures (2 per category)\n{'='*60}")
    for model, failures in model_failures.items():
        summary_parts.append(f"\n[[ {model} ]]")
        summary_parts.append(sample_failures(failures, n=2))

    summary_text = "\n".join(summary_parts)
    print(summary_text)

    summary_path = ANALYSIS_DIR / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")

    # save comparison CSV
    csv_path = ANALYSIS_DIR / "comparison.csv"
    save_comparison_csv(model_metrics, csv_path)
    print(f"Saved comparison CSV to {csv_path}")

    # save failures per model
    for model, failures in model_failures.items():
        safe_model = model.replace("/", "_").replace(":", "_")
        failures_path = ANALYSIS_DIR / f"failures_{safe_model}.jsonl"
        with failures_path.open("w", encoding="utf-8") as f:
            for rec in failures:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(failures)} failures to {failures_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
