#!/usr/bin/env python3
"""Evaluate NIH TrialBench target-trial Recall@K.

NIH TrialBench has one target/reference trial per vignette. This script checks
where that target trial appears in each TrialGPT ranked output and reports
Recall@K, with Recall@10 as the default figure metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def result_path(results_dir: Path, patient_id: str) -> Path:
    return results_dir / f"full_email_{patient_id}.json"


def resolve_results_dir(args: argparse.Namespace, target_study_type: str) -> Path:
    study_type = target_study_type.upper()
    if study_type == "INTERVENTIONAL" and args.interventional_results_dir:
        return args.interventional_results_dir
    if study_type == "OBSERVATIONAL" and args.observational_results_dir:
        return args.observational_results_dir
    if args.results_dir:
        return args.results_dir
    raise ValueError(f"No results directory configured for target_study_type={target_study_type!r}")


def target_row(
    ranked_rows: List[Dict[str, Any]],
    target_trial_id: str,
) -> tuple[Optional[int], Optional[Dict[str, Any]]]:
    for rank, row in enumerate(ranked_rows, start=1):
        trial_id = clean_text(row.get("trial_id") or row.get("nctid") or row.get("NCTID"))
        if trial_id == target_trial_id:
            return rank, row
    return None, None


def target_title(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    return clean_text(row.get("trial_title") or row.get("title") or row.get("brief_title"))


def candidate_trial_count(case: Dict[str, Any]) -> int:
    count = case.get("candidate_trial_count")
    if count not in (None, ""):
        return int(count)
    return len(case.get("candidate_trial_ids") or case.get("candidate_trials") or [])


def summarize_group(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    status_counts = Counter(row["status"] for row in rows)
    ranks = [int(row["target_rank"]) for row in rows if row.get("target_rank")]
    hit_key = f"hit_at_{k}"
    hit_count = sum(int(row.get(hit_key) or 0) for row in rows)
    mrr_sum = sum(float(row.get("mrr") or 0.0) for row in rows)

    summary: Dict[str, Any] = {
        "n_cases": len(rows),
        "n_ranked_targets": status_counts.get("ranked", 0),
        "n_missing_output": status_counts.get("missing_output", 0),
        "n_target_not_ranked": status_counts.get("target_not_ranked", 0),
        f"hit_count_at_{k}": hit_count,
        f"hit_rate_at_{k}": round(hit_count / len(rows), 6) if rows else 0.0,
        f"target_trial_recall_at_{k}_pct": round(100.0 * hit_count / len(rows), 2) if rows else 0.0,
        "mrr": round(mrr_sum / len(rows), 6) if rows else 0.0,
    }

    if ranks:
        summary["rank_mean"] = round(statistics.fmean(ranks), 4)
        summary["rank_median"] = round(statistics.median(ranks), 4)
        summary["rank_min"] = min(ranks)
        summary["rank_max"] = max(ranks)
    return summary


def grouped_summaries(rows: List[Dict[str, Any]], group_field: str, k: int) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[clean_text(row.get(group_field))].append(row)
    return {group: summarize_group(groups[group], k) for group in sorted(groups)}


def summarize(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    overall = summarize_group(rows, k)
    return {
        "metric": f"target_trial_recall_at_{k}",
        "definition": (
            "Fraction of patient queries for which the vignette target/reference trial appears "
            f"in the top {k} ranked trials. Missing output files and unranked target trials count as misses."
        ),
        "k": k,
        "overall": overall,
        "by_ic": grouped_summaries(rows, "ic", k),
        "by_study_type": grouped_summaries(rows, "target_study_type", k),
        "total_cases": overall["n_cases"],
        "ranked_target_count": overall["n_ranked_targets"],
        "missing_output_count": overall["n_missing_output"],
        "target_not_ranked_count": overall["n_target_not_ranked"],
        f"hit_rate_at_{k}": overall[f"hit_rate_at_{k}"],
        f"target_trial_recall_at_{k}_pct": overall[f"target_trial_recall_at_{k}_pct"],
        "mrr": overall["mrr"],
    }


def print_summary(summary: Dict[str, Any]) -> None:
    k = int(summary["k"])
    overall = summary["overall"]
    print(f"NIH TrialBench target-trial Recall@{k}")
    print(f"Cases: {overall['n_cases']}")
    print(f"Ranked target trials: {overall['n_ranked_targets']}")
    if overall["n_missing_output"]:
        print(f"Missing output files: {overall['n_missing_output']}")
    if overall["n_target_not_ranked"]:
        print(f"Target trials not ranked: {overall['n_target_not_ranked']}")
    print(
        f"Recall@{k}: {overall[f'hit_count_at_{k}']}/{overall['n_cases']} "
        f"= {overall[f'hit_rate_at_{k}']} ({overall[f'target_trial_recall_at_{k}_pct']}%)"
    )
    print(f"MRR: {overall['mrr']}")
    if "rank_mean" in overall:
        print(
            "Target rank: "
            f"mean={overall['rank_mean']}, median={overall['rank_median']}, "
            f"min={overall['rank_min']}, max={overall['rank_max']}"
        )


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any], k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"target_recall_at{k}_summary.json"
    rows_path = out_dir / f"target_recall_at{k}_per_case.csv"
    group_path = out_dir / f"target_recall_at{k}_summary_by_group.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hit_key = f"hit_at_{k}"
    fieldnames = [
        "patient_id",
        "ic",
        "target_trial_id",
        "target_study_type",
        "search_space_rule",
        "candidate_trial_count",
        "status",
        "target_rank",
        hit_key,
        "mrr",
        "target_trial_title",
        "result_path",
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    group_fieldnames = [
        "group_type",
        "group",
        "n_cases",
        "n_ranked_targets",
        "n_missing_output",
        "n_target_not_ranked",
        f"hit_count_at_{k}",
        f"hit_rate_at_{k}",
        f"target_trial_recall_at_{k}_pct",
        "mrr",
        "rank_mean",
        "rank_median",
        "rank_min",
        "rank_max",
    ]
    with group_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fieldnames)
        writer.writeheader()
        for group_type, key in [("ic", "by_ic"), ("study_type", "by_study_type")]:
            for group, group_summary in summary[key].items():
                writer.writerow({"group_type": group_type, "group": group, **group_summary})
        writer.writerow({"group_type": "all", "group": "ALL", **summary["overall"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NIH TrialBench target-trial Recall@K.")
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=Path("data/nih_trialbench/benchmark_cases.jsonl"),
        help="NIH TrialBench benchmark_cases.jsonl used for matching.",
    )
    parser.add_argument("--results-dir", type=Path, help="Directory containing full_email_<patient_id>.json files.")
    parser.add_argument(
        "--interventional-results-dir",
        type=Path,
        help="Directory containing full_email_<patient_id>.json files for interventional target trials.",
    )
    parser.add_argument(
        "--observational-results-dir",
        type=Path,
        help="Directory containing full_email_<patient_id>.json files for observational target trials.",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall cutoff. Defaults to 10.")
    parser.add_argument("--out-dir", type=Path, help="Optional directory for summary JSON and per-case CSV.")
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be a positive integer.")
    if not (args.results_dir or args.interventional_results_dir or args.observational_results_dir):
        parser.error("Provide --results-dir or study-type-specific results directories.")

    rows: List[Dict[str, Any]] = []
    hit_key = f"hit_at_{args.k}"
    for case in read_jsonl(args.cases_path):
        patient_id = clean_text(case.get("patient_id"))
        target_trial_id = clean_text(case.get("target_trial_id"))
        target_study_type = clean_text(case.get("target_study_type"))
        output_path = result_path(resolve_results_dir(args, target_study_type), patient_id)
        base_row: Dict[str, Any] = {
            "patient_id": patient_id,
            "ic": clean_text(case.get("ic")),
            "target_trial_id": target_trial_id,
            "target_study_type": target_study_type,
            "search_space_rule": clean_text(case.get("search_space_rule")),
            "candidate_trial_count": candidate_trial_count(case),
            "status": "",
            "target_rank": "",
            hit_key: 0,
            "mrr": 0.0,
            "target_trial_title": "",
            "result_path": str(output_path),
        }

        if not output_path.exists():
            rows.append({**base_row, "status": "missing_output"})
            continue

        data = json.loads(output_path.read_text(encoding="utf-8"))
        ranked_rows = list(data.get("cover_page_list") or [])
        rank, target = target_row(ranked_rows, target_trial_id)
        if target is None:
            rows.append({**base_row, "status": "target_not_ranked"})
            continue

        rows.append(
            {
                **base_row,
                "status": "ranked",
                "target_rank": rank,
                hit_key: int(rank <= args.k),
                "mrr": 1.0 / rank,
                "target_trial_title": target_title(target),
            }
        )

    summary = summarize(rows, args.k)
    print_summary(summary)
    if args.out_dir:
        write_outputs(args.out_dir, rows, summary, args.k)


if __name__ == "__main__":
    main()
