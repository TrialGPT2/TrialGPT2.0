#!/usr/bin/env python3
"""Build TrialGPT 2.0 inputs for the NIH TrialBench/NIH-Syn release.

NIH TrialBench evaluation is target-trial recovery: each vignette has one
reference target trial, and that target trial is the only gold label used by
the matcher/evaluator.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=False) + "\n")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def trial_id_of(trial: Dict[str, Any]) -> str:
    return clean_text(trial.get("nctid") or trial.get("trial_id"))


def study_type_of(trial: Dict[str, Any]) -> str:
    return clean_text(trial.get("study_type")).upper()


def is_nhgri_patient(patient: Dict[str, Any]) -> bool:
    patient_id = clean_text(patient.get("patient_id")).lower()
    ic = clean_text(patient.get("ic")).upper()
    return ic == "NHGRI" or patient_id.startswith("nhgri-")


def build_search_space(
    patient: Dict[str, Any],
    target_study_type: str,
    trials_by_type: Dict[str, List[Dict[str, Any]]],
    all_trials: List[Dict[str, Any]],
    nhgri_all_study_types: bool,
) -> tuple[str, List[Dict[str, Any]]]:
    if nhgri_all_study_types and is_nhgri_patient(patient):
        return "all_study_types_for_nhgri", all_trials
    return "same_study_type_as_target", trials_by_type.get(target_study_type, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NIH TrialBench data for TrialGPT 2.0.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing patients.jsonl and trial_search_space.jsonl.")
    parser.add_argument("--trial-dir", type=Path, required=True, help="Directory containing canonical NIH trial JSON files.")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Output root inside TrialGPT_2.0.")
    parser.add_argument("--benchmark-name", default="nih_trialbench", help="Benchmark dataset name.")
    parser.add_argument(
        "--no-nhgri-all-study-types",
        dest="nhgri_all_study_types",
        action="store_false",
        help="Disable the NHGRI exception and use target study type for every patient.",
    )
    parser.set_defaults(nhgri_all_study_types=True)
    args = parser.parse_args()

    source_dir = args.source_dir
    patients_path = source_dir / "patients.jsonl"
    trial_search_space_path = source_dir / "trial_search_space.jsonl"

    patients = read_jsonl(patients_path)
    trial_sources = read_jsonl(trial_search_space_path)
    trial_dir = args.trial_dir

    output_root = args.output_root
    dataset_dir = output_root / args.benchmark_name
    trials_dir = dataset_dir / "trials"

    trial_sources_by_id = {clean_text(row.get("trial_id")): row for row in trial_sources}
    if len(trial_sources_by_id) != len(trial_sources):
        raise RuntimeError("Duplicate trial_id values found in trial_search_space.jsonl.")

    missing_trial_files = []
    canonical_trials_by_id: Dict[str, Dict[str, Any]] = {}
    canonical_trial_paths_by_id: Dict[str, Path] = {}
    for trial_id in sorted(trial_sources_by_id):
        trial_path = trial_dir / f"{trial_id}.json"
        if not trial_path.exists():
            missing_trial_files.append(trial_id)
            continue
        trial_payload = json.loads(trial_path.read_text(encoding="utf-8"))
        payload_trial_id = trial_id_of(trial_payload)
        if payload_trial_id != trial_id:
            raise RuntimeError(f"Trial ID mismatch for {trial_path}: expected {trial_id}, found {payload_trial_id}")
        canonical_trials_by_id[trial_id] = trial_payload
        canonical_trial_paths_by_id[trial_id] = trial_path

    if missing_trial_files:
        raise RuntimeError(f"{len(missing_trial_files)} trial IDs are missing from --trial-dir: {missing_trial_files[:20]}")

    all_trials = [canonical_trials_by_id[trial_id] for trial_id in sorted(canonical_trials_by_id)]
    trials_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trial in all_trials:
        trials_by_type[study_type_of(trial)].append(trial)

    benchmark_cases: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    case_search_spaces: List[Dict[str, Any]] = []
    search_rule_counts: Counter[str] = Counter()
    candidate_count_by_rule: Counter[str] = Counter()
    target_type_counts: Counter[str] = Counter()

    for patient in sorted(patients, key=lambda row: clean_text(row.get("patient_id"))):
        patient_id = clean_text(patient.get("patient_id"))
        target_trial_id = clean_text(patient.get("target_trial_id"))
        if target_trial_id not in canonical_trials_by_id:
            raise RuntimeError(f"target_trial_id {target_trial_id!r} for {patient_id} is missing from trial_search_space.jsonl")

        target_study_type = study_type_of(canonical_trials_by_id[target_trial_id])
        target_type_counts[target_study_type] += 1
        search_rule, candidate_trials = build_search_space(
            patient,
            target_study_type,
            trials_by_type,
            all_trials,
            args.nhgri_all_study_types,
        )
        candidate_trial_ids = [trial_id_of(trial) for trial in candidate_trials]
        candidate_ids = set(candidate_trial_ids)
        if target_trial_id not in candidate_ids:
            raise RuntimeError(
                f"target_trial_id {target_trial_id!r} for {patient_id} is outside its candidate search space"
            )
        search_rule_counts[search_rule] += 1
        candidate_count_by_rule[search_rule] += len(candidate_trials)

        patient_summary = clean_text(patient.get("patient_summary"))
        case_search_spaces.append({
            "patient_id": patient_id,
            "ic": clean_text(patient.get("ic")),
            "target_trial_id": target_trial_id,
            "target_study_type": target_study_type,
            "search_space_rule": search_rule,
            "candidate_trial_count": len(candidate_trial_ids),
            "candidate_trial_ids": candidate_trial_ids,
        })
        queries.append({
            "patient_id": patient_id,
            "ic": clean_text(patient.get("ic")),
            "patient_summary": patient_summary,
            "target_trial_id": target_trial_id,
            "target_study_type": target_study_type,
            "search_space_rule": search_rule,
            "candidate_trial_count": len(candidate_trials),
        })
        benchmark_cases.append({
            "patient_id": patient_id,
            "ic": clean_text(patient.get("ic")),
            "target_trial_id": target_trial_id,
            "target_study_type": target_study_type,
            "search_space_rule": search_rule,
            "patient_summary": patient_summary,
            "candidate_trial_count": len(candidate_trial_ids),
            "candidate_trial_ids": candidate_trial_ids,
        })

    dataset_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    stale_benchmark_dir = output_root / f"{args.benchmark_name}_benchmark"
    if stale_benchmark_dir.exists():
        shutil.rmtree(stale_benchmark_dir)
    stale_qrels_dir = dataset_dir / "qrels"
    if stale_qrels_dir.exists():
        shutil.rmtree(stale_qrels_dir)
    stale_skipped_path = dataset_dir / "skipped_annotations_outside_candidate_pool.jsonl"
    if stale_skipped_path.exists():
        stale_skipped_path.unlink()
    stale_annotations_path = dataset_dir / "clinician_annotations.jsonl"
    if stale_annotations_path.exists():
        stale_annotations_path.unlink()
    for stale_name in ["build_summary.json", "patients.jsonl", "trial_search_space.jsonl"]:
        stale_path = dataset_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    write_jsonl(dataset_dir / "queries.jsonl", queries)
    write_jsonl(dataset_dir / "case_search_space.jsonl", case_search_spaces)
    write_jsonl(dataset_dir / "benchmark_cases.jsonl", benchmark_cases)

    for trial_id, trial_path in sorted(canonical_trial_paths_by_id.items()):
        (trials_dir / f"{trial_id}.json").write_bytes(trial_path.read_bytes())

    summary = {
        "benchmark_name": args.benchmark_name,
        "source": "NIH-Syn Hugging Face export",
        "source_dir_name": source_dir.name,
        "trial_source": "NIH_update_trials",
        "trial_dir_name": trial_dir.name,
        "source_files": [
            "patients.jsonl",
            "trial_search_space.jsonl",
        ],
        "patients": len(patients),
        "trial_search_space": len(trial_sources),
        "trial_json_files_used": len(canonical_trials_by_id),
        "trial_study_type_counts": dict(sorted(Counter(study_type_of(trial) for trial in all_trials).items())),
        "target_study_type_counts": dict(sorted(target_type_counts.items())),
        "search_rule_counts": dict(sorted(search_rule_counts.items())),
        "candidate_pairs": sum(case["candidate_trial_count"] for case in benchmark_cases),
        "candidate_pairs_by_rule": dict(sorted(candidate_count_by_rule.items())),
        "benchmark_cases_file": "benchmark_cases.jsonl",
        "case_search_space_file": "case_search_space.jsonl",
        "target_reference_field": "target_trial_id",
        "nhgri_all_study_types": bool(args.nhgri_all_study_types),
        "qrels_written": False,
    }

    readme = (
        "# NIH TrialBench\n\n"
        "This directory was generated from the NIH-Syn Hugging Face release for TrialGPT 2.0.\n\n"
        "Trial objects were copied from the NIH update trial JSON directory with source fields and field order preserved.\n\n"
        "Search-space rule:\n"
        "- NHGRI patients use all trial study types.\n"
        "- All other patients use trials with the same study type as the patient's target/reference trial.\n\n"
        "Files:\n"
        "- `benchmark_cases.jsonl`: compact matcher input with candidate trial IDs.\n"
        "- `queries.jsonl`: patient IDs and summaries used by the evaluator.\n"
        "- `case_search_space.jsonl`: per-vignette trial search-space IDs used for matching.\n"
        "- `trials/`: exact trial JSON files copied from the NIH update trial directory.\n"
        "\n"
        "Evaluation is target-trial recovery. Each `benchmark_cases.jsonl` row includes the "
        "vignette's `target_trial_id`; no qrels or clinician annotation labels are packaged.\n"
    )
    (dataset_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
