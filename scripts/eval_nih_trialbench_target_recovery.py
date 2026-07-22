#!/usr/bin/env python3
"""Evaluate NIH TrialBench by target-trial category recovery.

This benchmark has one target/reference trial per vignette. The report answers:
what percentage of target trials did TrialGPT assign to highly recommended,
possible match, or low fit?
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CATEGORY_ORDER = ["highly recommended", "possible match", "low fit"]
CATEGORY_DISPLAY = {
    "highly recommended": "Highly recommended",
    "possible match": "Possible match",
    "low fit": "Low fit",
}
HIT_KS = [1, 3, 5, 10]


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


def score_to_category(score: Optional[float]) -> str:
    if score is None:
        return "missing_score"
    if score > 90:
        return "highly recommended"
    if score >= 80:
        return "possible match"
    return "low fit"


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def blank_if_none(value: Optional[float]) -> Any:
    return "" if value is None else value


def trialgpt1_fit_score(row: Dict[str, Any]) -> Optional[float]:
    """Map TrialGPT 1.0 aggregation scores onto the 0-100 category scale."""
    relevance = parse_float(row.get("relevance_score_R"))
    eligibility = parse_float(row.get("eligibility_score_E"))
    if relevance is not None and eligibility is not None:
        return (relevance + eligibility) / 2.0

    aggregation_score = parse_float(row.get("aggregation_score"))
    if aggregation_score is not None:
        return aggregation_score * 50.0
    return None


def get_score(row: Dict[str, Any], score_field: str) -> Optional[float]:
    if score_field == "trialgpt1_fit_score":
        return trialgpt1_fit_score(row)

    fields = ["final_score", "score"] if score_field == "auto" else [score_field]
    for field in fields:
        score = parse_float(row.get(field))
        if score is not None:
            return score
    return None


def score_definition(score_field: str) -> str:
    if score_field == "trialgpt1_fit_score":
        return "TrialGPT 1.0 fit score on 0-100 scale: (relevance_score_R + eligibility_score_E) / 2"
    if score_field == "auto":
        return "First numeric value from final_score, then score"
    return score_field


def category_thresholds(score_field: str) -> Dict[str, str]:
    score_name = score_definition(score_field)
    return {
        "highly recommended": f"{score_name} > 90",
        "possible match": f"80 <= {score_name} <= 90",
        "low fit": f"{score_name} < 80",
    }


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
        trial_id = clean_text(row.get("trial_id") or row.get("nctid"))
        if trial_id == target_trial_id:
            return rank, row
    return None, None


def pct(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * count / denominator, 2)


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = Counter(row["status"] for row in rows)
    category_counts = Counter(
        row["target_category"]
        for row in rows
        if row["status"] == "ranked" and row["target_category"] in CATEGORY_ORDER
    )
    ranked_target_count = sum(category_counts.values())
    ranks = [int(row["target_rank"]) for row in rows if row.get("target_rank")]

    summary: Dict[str, Any] = {
        "n_cases": len(rows),
        "n_ranked_targets": ranked_target_count,
        "n_missing_output": status_counts.get("missing_output", 0),
        "n_target_not_ranked": status_counts.get("target_not_ranked", 0),
        "n_missing_score": status_counts.get("missing_score", 0),
    }
    for category in CATEGORY_ORDER:
        key = category.replace(" ", "_")
        count = category_counts.get(category, 0)
        summary[f"n_{key}"] = count
        summary[f"pct_{key}"] = pct(count, ranked_target_count)

    for k in HIT_KS:
        summary[f"hit_rate_at_{k}"] = round(sum(1 for rank in ranks if rank <= k) / len(rows), 6) if rows else 0.0
    summary["mrr"] = round(sum(1.0 / rank for rank in ranks) / len(rows), 6) if rows else 0.0

    if ranks:
        summary["rank_mean"] = round(statistics.fmean(ranks), 4)
        summary["rank_median"] = round(statistics.median(ranks), 4)
        summary["rank_min"] = min(ranks)
        summary["rank_max"] = max(ranks)
    return summary


def grouped_summaries(rows: List[Dict[str, Any]], group_field: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[clean_text(row.get(group_field))].append(row)
    return {group: summarize_group(groups[group]) for group in sorted(groups)}


def summarize(rows: List[Dict[str, Any]], score_field: str) -> Dict[str, Any]:
    overall = summarize_group(rows)
    summary: Dict[str, Any] = {
        "score_field": score_field,
        "score_definition": score_definition(score_field),
        "category_thresholds": category_thresholds(score_field),
        "overall": overall,
        "by_ic": grouped_summaries(rows, "ic"),
        "by_study_type": grouped_summaries(rows, "target_study_type"),
        # Backward-compatible keys used by the earlier version of this script.
        "total_cases": overall["n_cases"],
        "ranked_target_count": overall["n_ranked_targets"],
        "missing_output_count": overall["n_missing_output"],
        "target_not_ranked_count": overall["n_target_not_ranked"],
        "missing_score_count": overall["n_missing_score"],
        "category_counts": {
            category: overall[f"n_{category.replace(' ', '_')}"] for category in CATEGORY_ORDER
        },
        "category_percent_of_ranked_targets": {
            category: overall[f"pct_{category.replace(' ', '_')}"] for category in CATEGORY_ORDER
        },
        "category_percent_of_all_cases": {
            category: pct(overall[f"n_{category.replace(' ', '_')}"], overall["n_cases"])
            for category in CATEGORY_ORDER
        },
    }
    if "rank_mean" in overall:
        summary["target_rank"] = {
            "mean": overall["rank_mean"],
            "median": overall["rank_median"],
            "min": overall["rank_min"],
            "max": overall["rank_max"],
        }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    overall = summary["overall"]
    total = overall["n_cases"]
    ranked = overall["n_ranked_targets"]
    print("NIH TrialBench target-trial recovery")
    print(f"Cases: {total}")
    print(f"Ranked target trials: {ranked}")
    print(f"Score definition: {summary['score_definition']}")
    if overall["n_missing_output"]:
        print(f"Missing output files: {overall['n_missing_output']}")
    if overall["n_target_not_ranked"]:
        print(f"Target trials not ranked: {overall['n_target_not_ranked']}")
    if overall["n_missing_score"]:
        print(f"Ranked target trials with missing score: {overall['n_missing_score']}")
    print("")
    print("Target-trial assigned category:")
    for category in CATEGORY_ORDER:
        key = category.replace(" ", "_")
        count = overall[f"n_{key}"]
        pct_ranked = overall[f"pct_{key}"]
        pct_all = pct(count, total)
        print(f"- {CATEGORY_DISPLAY[category]}: {count}/{ranked} ({pct_ranked}%) ranked targets; {pct_all}% of all cases")
    if "rank_mean" in overall:
        print("")
        print(
            "Target rank: "
            f"mean={overall['rank_mean']}, median={overall['rank_median']}, "
            f"min={overall['rank_min']}, max={overall['rank_max']}"
        )
        print(
            "Target retrieval: "
            f"hit@1={overall['hit_rate_at_1']}, hit@3={overall['hit_rate_at_3']}, "
            f"hit@5={overall['hit_rate_at_5']}, hit@10={overall['hit_rate_at_10']}, "
            f"MRR={overall['mrr']}"
        )


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "target_recovery_summary.json"
    rows_path = out_dir / "target_recovery_per_case.csv"
    group_path = out_dir / "target_recovery_summary_by_group.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "patient_id",
        "ic",
        "target_trial_id",
        "target_study_type",
        "search_space_rule",
        "candidate_trial_count",
        "status",
        "target_rank",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "target_score",
        "target_final_score",
        "target_category_score",
        "target_matching_score",
        "target_aggregation_score",
        "target_relevance_score_R",
        "target_eligibility_score_E",
        "target_fit_score_0_100",
        "target_category",
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
        "n_missing_score",
        "n_highly_recommended",
        "pct_highly_recommended",
        "n_possible_match",
        "pct_possible_match",
        "n_low_fit",
        "pct_low_fit",
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "hit_rate_at_10",
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
    parser = argparse.ArgumentParser(description="Evaluate NIH TrialBench target-trial category recovery.")
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
    parser.add_argument(
        "--score-field",
        choices=["final_score", "score", "auto", "trialgpt1_fit_score"],
        default="final_score",
        help=(
            "Score used for category thresholds. Use final_score for TrialGPT 2.0 outputs; "
            "use trialgpt1_fit_score for TrialGPT 1.0 ranked outputs."
        ),
    )
    parser.add_argument("--out-dir", type=Path, help="Optional directory for summary JSON and per-case CSV.")
    args = parser.parse_args()
    if not (args.results_dir or args.interventional_results_dir or args.observational_results_dir):
        parser.error("Provide --results-dir or study-type-specific results directories.")

    rows: List[Dict[str, Any]] = []
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
            "candidate_trial_count": int(case.get("candidate_trial_count") or len(case.get("candidate_trials") or [])),
            "status": "",
            "target_rank": "",
            "hit_at_1": "",
            "hit_at_3": "",
            "hit_at_5": "",
            "hit_at_10": "",
            "mrr": "",
            "target_score": "",
            "target_final_score": "",
            "target_category_score": "",
            "target_matching_score": "",
            "target_aggregation_score": "",
            "target_relevance_score_R": "",
            "target_eligibility_score_E": "",
            "target_fit_score_0_100": "",
            "target_category": "",
            "target_trial_title": "",
            "result_path": str(output_path),
        }

        if not output_path.exists():
            rows.append({**base_row, "status": "missing_output", "target_category": "missing_output"})
            continue

        data = json.loads(output_path.read_text(encoding="utf-8"))
        ranked_rows = list(data.get("cover_page_list") or [])
        rank, target = target_row(ranked_rows, target_trial_id)
        if target is None:
            rows.append({**base_row, "status": "target_not_ranked", "target_category": "target_not_ranked"})
            continue

        target_score = get_score(target, "score")
        target_final_score = get_score(target, "final_score")
        target_fit_score = trialgpt1_fit_score(target)
        category_score = get_score(target, args.score_field)
        category = score_to_category(category_score)
        status = "missing_score" if category == "missing_score" else "ranked"
        rows.append(
            {
                **base_row,
                "status": status,
                "target_rank": rank,
                "hit_at_1": int(rank <= 1),
                "hit_at_3": int(rank <= 3),
                "hit_at_5": int(rank <= 5),
                "hit_at_10": int(rank <= 10),
                "mrr": 1.0 / rank,
                "target_score": blank_if_none(target_score),
                "target_final_score": blank_if_none(target_final_score),
                "target_category_score": blank_if_none(category_score),
                "target_matching_score": blank_if_none(parse_float(target.get("matching_score"))),
                "target_aggregation_score": blank_if_none(parse_float(target.get("aggregation_score"))),
                "target_relevance_score_R": blank_if_none(parse_float(target.get("relevance_score_R"))),
                "target_eligibility_score_E": blank_if_none(parse_float(target.get("eligibility_score_E"))),
                "target_fit_score_0_100": blank_if_none(target_fit_score),
                "target_category": category,
                "target_trial_title": clean_text(target.get("trial_title") or target.get("title")),
            }
        )

    summary = summarize(rows, args.score_field)
    print_summary(summary)
    if args.out_dir:
        write_outputs(args.out_dir, rows, summary)


if __name__ == "__main__":
    main()
