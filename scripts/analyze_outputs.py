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


def _skywork_identify(model_pairs: dict[str, list]) -> tuple[str | None, str | None]:
    """Return (reward_key, critic_key) for the two Skywork models, or None if absent."""
    reward_key = critic_key = None
    for model, pairs in model_pairs.items():
        if not pairs:
            continue
        jn = pairs[0].get("judge_name", "")
        if jn == "reward_model" and "Skywork" in model:
            reward_key = model
        elif jn == "skywork_critic" and "Skywork" in model:
            critic_key = model
    return reward_key, critic_key


def _reward_scores(pair: dict) -> tuple[float, float] | None:
    """Extract (score_A, score_B) from a reward-model pair, or None."""
    try:
        scores = pair["judgments"][0]["judgment"]["scores"]
        if scores and len(scores) == 2:
            return float(scores[0]), float(scores[1])
    except (IndexError, KeyError, TypeError):
        pass
    return None


def skywork_pairwise_agreement(reward_pairs: list[dict], critic_pairs: list[dict]) -> dict:
    """Compute how often reward and critic agree/disagree and who was right."""
    reward_by_id = {p["pair_id"]: p for p in reward_pairs}
    critic_by_id = {p["pair_id"]: p for p in critic_pairs}
    shared_ids = set(reward_by_id) & set(critic_by_id)

    counts = {
        "both_correct": 0,
        "both_incorrect": 0,
        "reward_only": 0,
        "critic_only": 0,
        "other": 0,
        "total": len(shared_ids),
    }

    for pid in shared_ids:
        r_outcome = score_pair(reward_by_id[pid])
        c_outcome = score_pair(critic_by_id[pid])
        r_correct = r_outcome == "correct"
        c_correct = c_outcome == "correct"
        if r_correct and c_correct:
            counts["both_correct"] += 1
        elif not r_correct and not c_correct:
            counts["both_incorrect"] += 1
        elif r_correct:
            counts["reward_only"] += 1
        elif c_correct:
            counts["critic_only"] += 1
        else:
            counts["other"] += 1

    return counts


def skywork_margin_stats(reward_pairs: list[dict]) -> dict:
    """Return score margin stats split by outcome and per category."""
    by_outcome: dict[str, list[float]] = {"correct": [], "incorrect": [], "inconsistent": []}
    by_cat: dict[str, dict[str, list[float]]] = {
        cat: {"correct": [], "incorrect": []} for cat in CATEGORIES
    }

    for pair in reward_pairs:
        scores = _reward_scores(pair)
        if scores is None:
            continue
        margin = abs(scores[0] - scores[1])
        outcome = score_pair(pair)
        if outcome in by_outcome:
            by_outcome[outcome].append(margin)
        cat = top_level_category(pair["source"])
        if cat in by_cat:
            bucket = "correct" if outcome == "correct" else "incorrect"
            by_cat[cat][bucket].append(margin)

    def _agg(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        import statistics
        return f"{statistics.mean(vals):.2f} (n={len(vals)})"

    stats = {
        "by_outcome": {k: _agg(v) for k, v in by_outcome.items()},
        "by_cat": {cat: {k: _agg(v) for k, v in buckets.items()} for cat, buckets in by_cat.items()},
    }
    return stats


def print_skywork_comparison(
    reward_key: str,
    critic_key: str,
    model_metrics: dict[str, dict],
    reward_pairs: list[dict],
    critic_pairs: list[dict],
) -> str:
    reward_m = model_metrics[reward_key]
    critic_m = model_metrics[critic_key]

    lines = [
        f"\n{'='*60}",
        "  Skywork Reward vs Critic — Direct Comparison",
        f"{'='*60}",
        f"  Reward : {reward_key}",
        f"  Critic : {critic_key}",
    ]

    # per-category accuracy table
    lines += [
        "",
        f"  {'Category':<24} {'Reward':>8} {'Critic':>8} {'Delta':>7}",
        "  " + "-" * 50,
    ]
    for cat in CATEGORIES:
        rm = reward_m[cat]
        cm = critic_m[cat]
        r_acc = rm["accuracy"]
        c_acc = cm["accuracy"]
        if r_acc is None or c_acc is None:
            lines.append(f"  {cat:<24} {'n/a':>8} {'n/a':>8} {'n/a':>7}")
        else:
            delta = c_acc - r_acc
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"  {cat:<24} {r_acc:>7.1f}% {c_acc:>7.1f}% {sign}{delta:>5.1f}"
            )

    # pairwise agreement
    agreement = skywork_pairwise_agreement(reward_pairs, critic_pairs)
    n = agreement["total"]
    lines += [
        "",
        f"  --- Pairwise agreement  (N={n} shared pairs) ---",
        f"  {'Both correct':<24} {agreement['both_correct']:>5}  ({_rate(agreement['both_correct'], n):>6})",
        f"  {'Both incorrect':<24} {agreement['both_incorrect']:>5}  ({_rate(agreement['both_incorrect'], n):>6})",
        f"  {'Reward correct only':<24} {agreement['reward_only']:>5}  ({_rate(agreement['reward_only'], n):>6})",
        f"  {'Critic correct only':<24} {agreement['critic_only']:>5}  ({_rate(agreement['critic_only'], n):>6})",
        f"  {'Other (incons/tie)':<24} {agreement['other']:>5}  ({_rate(agreement['other'], n):>6})",
        f"  {'Agreement rate':<24} {agreement['both_correct'] + agreement['both_incorrect']:>5}  "
        f"({_rate(agreement['both_correct'] + agreement['both_incorrect'], n):>6})",
    ]

    # reward score margin analysis
    margin_stats = skywork_margin_stats(reward_pairs)
    lines += [
        "",
        "  --- Reward model score margin analysis ---",
        "  (margin = |score_A - score_B|; higher = more confident)",
        f"  Correct   pairs — mean margin: {margin_stats['by_outcome']['correct']}",
        f"  Incorrect pairs — mean margin: {margin_stats['by_outcome']['incorrect']}",
        f"  Incons.   pairs — mean margin: {margin_stats['by_outcome']['inconsistent']}",
        "",
        f"  {'Category':<24} {'Correct margin':>20} {'Incorrect margin':>20}",
        "  " + "-" * 66,
    ]
    for cat in CATEGORIES:
        bc = margin_stats["by_cat"].get(cat, {})
        lines.append(
            f"  {cat:<24} {bc.get('correct', 'n/a'):>20} {bc.get('incorrect', 'n/a'):>20}"
        )

    return "\n".join(lines)


def _classify_paradigm(model_key: str, pairs: list[dict]) -> str:
    if not pairs:
        return "Other"
    jn = pairs[0].get("judge_name", "")
    if jn == "reward_model":
        return "Reward Model"
    if jn == "skywork_critic":
        return "Fine-tuned (Critic)"
    if jn == "arena_hard":
        return "Prompted"
    return "Other"


def _display_name(model_key: str) -> str:
    k = model_key.lower()
    if "gemini" in k:
        return "gemini-2.5-fl"
    if "4.1-mini" in model_key:
        return "gpt-4.1-mini"
    if "4o-mini" in k:
        return "gpt-4o-mini"
    if "llama" in k and "skywork" not in k:
        return "llama-3.1-8B"
    if "reward" in k:
        return "Skywork-Reward"
    if "critic" in k:
        return "Skywork-Critic"
    return model_key[:14]


def _final_verdict(pair: dict) -> str:
    """Collapse the (possibly two-pass) judgments into a single verdict string."""
    judgments = pair["judgments"]
    j1 = judgments[0] if judgments else None
    j2 = judgments[1] if len(judgments) > 1 else None
    if j1 is None:
        return "unknown"
    if j2 is None:
        return j1["decision"]
    d1 = j1["decision"]
    d2 = flip(j2["decision"])
    return d1 if d1 == d2 else "tie"


def _pairwise_agreement_rate(k1: str, k2: str, model_pairs: dict) -> tuple[int, int]:
    by_id1 = {p["pair_id"]: p for p in model_pairs[k1]}
    by_id2 = {p["pair_id"]: p for p in model_pairs[k2]}
    shared = set(by_id1) & set(by_id2)
    if not shared:
        return 0, 0
    agree = sum(1 for pid in shared if _final_verdict(by_id1[pid]) == _final_verdict(by_id2[pid]))
    return agree, len(shared)


def print_prompted_indepth(prompted_keys: list[str], model_metrics: dict, model_pairs: dict) -> str:
    labels = [_display_name(k) for k in prompted_keys]
    col_w = 13

    lines = [
        f"\n{'='*60}",
        "  Prompted judges — in-depth comparison",
        f"{'='*60}",
        "  Models: " + " | ".join(labels),
    ]

    # Inconsistency rates by category
    lines += [
        "",
        "  --- Inconsistency rates by category ---",
        "  (2-pass reverse-order scoring: inconsistency = flipped verdict on swap)",
        f"  {'Category':<24}" + "".join(f"  {lb[:col_w]:>{col_w}}" for lb in labels),
        "  " + "-" * (24 + (col_w + 2) * len(prompted_keys)),
    ]
    for cat in CATEGORIES:
        row = f"  {cat:<24}"
        for k in prompted_keys:
            m = model_metrics[k][cat]
            row += f"  {_rate(m['inconsistent'], m['total']):>{col_w}}"
        lines.append(row)

    # Pairwise agreement matrix
    lines += [
        "",
        "  --- Pairwise agreement matrix ---",
        "  (% of pairs where both models give the same final verdict)",
        f"  {'':24}" + "".join(f"  {lb[:col_w]:>{col_w}}" for lb in labels),
        "  " + "-" * (24 + (col_w + 2) * len(prompted_keys)),
    ]
    for i, k1 in enumerate(prompted_keys):
        row = f"  {labels[i]:<24}"
        for k2 in prompted_keys:
            if k1 == k2:
                row += f"  {'100.0%':>{col_w}}"
            else:
                agree, total = _pairwise_agreement_rate(k1, k2, model_pairs)
                row += f"  {_rate(agree, total):>{col_w}}"
        lines.append(row)

    # Consensus analysis
    by_id = {k: {p["pair_id"]: p for p in model_pairs[k]} for k in prompted_keys}
    shared_ids = set.intersection(*(set(d.keys()) for d in by_id.values()))
    n = len(shared_ids)
    nm = len(prompted_keys)
    buckets = {
        "unanimous_correct": 0,
        "majority_correct": 0,
        "even_split": 0,
        "majority_wrong": 0,
        "unanimous_wrong": 0,
    }
    for pid in shared_ids:
        correct_count = sum(1 for k in prompted_keys if score_pair(by_id[k][pid]) == "correct")
        if correct_count == nm:
            buckets["unanimous_correct"] += 1
        elif correct_count > nm / 2:
            buckets["majority_correct"] += 1
        elif correct_count == nm / 2:
            buckets["even_split"] += 1
        elif correct_count > 0:
            buckets["majority_wrong"] += 1
        else:
            buckets["unanimous_wrong"] += 1

    maj = nm // 2 + 1
    lines += [
        "",
        f"  --- Consensus across all {nm} prompted models  (N={n} pairs) ---",
        f"  {'All correct (unanimous)':<34} {buckets['unanimous_correct']:>5}  ({_rate(buckets['unanimous_correct'], n):>6})",
        f"  {f'Majority correct ({maj}/{nm}+)':<34} {buckets['majority_correct']:>5}  ({_rate(buckets['majority_correct'], n):>6})",
        f"  {f'Even split ({nm//2}/{nm})':<34} {buckets['even_split']:>5}  ({_rate(buckets['even_split'], n):>6})",
        f"  {f'Majority wrong (only 1/{nm} right)':<34} {buckets['majority_wrong']:>5}  ({_rate(buckets['majority_wrong'], n):>6})",
        f"  {'All wrong (unanimous)':<34} {buckets['unanimous_wrong']:>5}  ({_rate(buckets['unanimous_wrong'], n):>6})",
    ]

    # Per-category ranking
    lines += [
        "",
        "  --- Per-category ranking (best -> worst accuracy) ---",
    ]
    for cat in CATEGORIES:
        ranked = sorted(
            prompted_keys,
            key=lambda k: model_metrics[k][cat]["accuracy"] or -1,
            reverse=True,
        )
        parts = "  ".join(
            f"{_display_name(k)} {model_metrics[k][cat]['accuracy']:.1f}%"
            if model_metrics[k][cat]["accuracy"] is not None
            else f"{_display_name(k)} n/a"
            for k in ranked
        )
        lines.append(f"  {cat:<24}  {parts}")

    return "\n".join(lines)


def print_paradigm_comparison(model_metrics: dict, model_pairs: dict) -> str:
    groups: dict[str, list[str]] = {}
    for model, pairs in model_pairs.items():
        p = _classify_paradigm(model, pairs)
        groups.setdefault(p, []).append(model)

    paradigm_order = ["Prompted", "Fine-tuned (Critic)", "Reward Model"]
    ordered = {p: groups[p] for p in paradigm_order if p in groups}
    for p in groups:
        if p not in ordered:
            ordered[p] = groups[p]

    paradigms = list(ordered.keys())
    col_w = 17

    lines = [
        f"\n{'='*60}",
        "  Judge paradigm comparison",
        f"{'='*60}",
        "  Paradigm groupings:",
    ]
    for p, keys in ordered.items():
        names = ", ".join(_display_name(k) for k in keys)
        lines.append(f"    {p:<20}: {names}")

    # average accuracy by paradigm
    lines += [
        "",
        "  --- Average accuracy by paradigm ---",
        f"  {'Category':<24}" + "".join(f"  {p[:col_w]:>{col_w}}" for p in paradigms),
        "  " + "-" * (24 + (col_w + 2) * len(paradigms)),
    ]
    for cat in CATEGORIES:
        row = f"  {cat:<24}"
        for p in paradigms:
            accs = [
                model_metrics[k][cat]["accuracy"]
                for k in ordered[p]
                if model_metrics[k][cat]["accuracy"] is not None
            ]
            row += f"  {sum(accs)/len(accs):>{col_w-1}.1f}%" if accs else f"  {'n/a':>{col_w}}"
        lines.append(row)

    # average inconsistency by paradigm
    lines += [
        "",
        "  --- Average inconsistency rate by paradigm ---",
        "  (Reward Model uses single-pass scoring — inconsistency is 0% by design)",
        f"  {'Category':<24}" + "".join(f"  {p[:col_w]:>{col_w}}" for p in paradigms),
        "  " + "-" * (24 + (col_w + 2) * len(paradigms)),
    ]
    for cat in CATEGORIES:
        row = f"  {cat:<24}"
        for p in paradigms:
            rates = []
            for k in ordered[p]:
                m = model_metrics[k][cat]
                if m["total"] > 0:
                    rates.append(100.0 * m["inconsistent"] / m["total"])
            row += f"  {sum(rates)/len(rates):>{col_w-1}.1f}%" if rates else f"  {'0.0%':>{col_w}}"
        lines.append(row)

    # best paradigm per category
    lines += [
        "",
        "  --- Best-performing paradigm per category ---",
    ]
    for cat in CATEGORIES:
        best_p = best_acc = None
        for p in paradigms:
            accs = [
                model_metrics[k][cat]["accuracy"]
                for k in ordered[p]
                if model_metrics[k][cat]["accuracy"] is not None
            ]
            if not accs:
                continue
            avg = sum(accs) / len(accs)
            if best_acc is None or avg > best_acc:
                best_acc = avg
                best_p = p
        if best_p is not None:
            lines.append(f"  {cat:<24}  -> {best_p:<22} ({best_acc:.1f}% avg)")
        else:
            lines.append(f"  {cat:<24}  -> n/a")

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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    model_pairs: dict[str, list] = {}

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
        model_pairs[model] = pairs

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

    # Skywork-specific comparison section
    reward_key, critic_key = _skywork_identify(model_pairs)
    if reward_key and critic_key:
        summary_parts.append(
            print_skywork_comparison(
                reward_key,
                critic_key,
                model_metrics,
                model_pairs[reward_key],
                model_pairs[critic_key],
            )
        )
    else:
        missing = []
        if not reward_key:
            missing.append("Skywork Reward")
        if not critic_key:
            missing.append("Skywork Critic")
        summary_parts.append(
            f"\n[Skywork comparison skipped — missing output(s): {', '.join(missing)}]"
        )

    # prompted judges in-depth
    prompted_keys = [
        k for k, pairs in model_pairs.items()
        if pairs and pairs[0].get("judge_name") == "arena_hard"
    ]
    if len(prompted_keys) >= 2:
        summary_parts.append(print_prompted_indepth(prompted_keys, model_metrics, model_pairs))

    # paradigm-level comparison
    summary_parts.append(print_paradigm_comparison(model_metrics, model_pairs))

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
