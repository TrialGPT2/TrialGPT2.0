from __future__ import annotations

import argparse
from pathlib import Path

from src.trial_matching.matcher_with_usage_logging import (
    DEFAULT_CACHED_INPUT_COST_PER_1M,
    DEFAULT_COST_UNIT,
    DEFAULT_INPUT_COST_PER_1M,
    DEFAULT_OUTPUT_COST_PER_1M,
    NOTE_FIELD_TO_CASE_KEY,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a copied trial matcher with per patient-trial timing/token logs."
    )
    parser.add_argument("--cases_path", type=Path, required=True, help="Path to benchmark_cases.jsonl")
    parser.add_argument(
        "--trial_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing trial JSON files when cases use candidate_trial_ids. "
            "Defaults to <cases_path parent>/trials."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/trial_matching_logged"),
        help="Directory for per-patient ranked outputs. Defaults to outputs/trial_matching_logged",
    )
    parser.add_argument(
        "--note_field",
        type=str,
        choices=sorted(NOTE_FIELD_TO_CASE_KEY.keys()),
        default="brief_note",
        help="Patient note alias to score against trials. Packaged benchmark cases use patient_summary.",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional override for LLM deployment/model name")
    parser.add_argument("--nprocs", type=int, default=4, help="Parallel workers per patient over candidate trials")
    parser.add_argument("--resume", action="store_true", help="Skip patients whose output file already exists")
    parser.add_argument("--show_progress", action="store_true", help="Show per-patient and per-trial progress")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep going when individual trial calls fail; failed trials are recorded in output metadata and logs",
    )
    parser.add_argument(
        "--pair_log_dir",
        type=Path,
        default=None,
        help="Directory for per-patient JSONL pair logs. Default: <out_dir>/pair_logs",
    )
    parser.add_argument(
        "--omit_temperature",
        action="store_true",
        help="Do not send temperature. Useful for deployments that only accept the default temperature.",
    )
    parser.add_argument(
        "--input_cost_per_1m",
        type=float,
        default=DEFAULT_INPUT_COST_PER_1M,
        help=(
            "Estimated input cost per 1M uncached prompt tokens. "
            "Default is the GPT-5.4 Codex credit rate."
        ),
    )
    parser.add_argument(
        "--cached_input_cost_per_1m",
        type=float,
        default=DEFAULT_CACHED_INPUT_COST_PER_1M,
        help=(
            "Estimated input cost per 1M cached prompt tokens. "
            "Default is the GPT-5.4 Codex credit rate."
        ),
    )
    parser.add_argument(
        "--output_cost_per_1m",
        type=float,
        default=DEFAULT_OUTPUT_COST_PER_1M,
        help=(
            "Estimated output cost per 1M completion tokens. "
            "Default is the GPT-5.4 Codex credit rate."
        ),
    )
    parser.add_argument(
        "--cost_unit",
        type=str,
        default=DEFAULT_COST_UNIT,
        help="Unit label for estimated cost fields, e.g. credits or USD.",
    )
    args = parser.parse_args()

    written = run(
        cases_path=args.cases_path,
        trial_dir=args.trial_dir,
        out_dir=args.out_dir,
        note_field=args.note_field,
        model=args.model,
        nprocs=args.nprocs,
        resume=args.resume,
        show_progress=args.show_progress,
        continue_on_error=args.continue_on_error,
        pair_log_dir=args.pair_log_dir,
        omit_temperature=args.omit_temperature,
        input_cost_per_1m=args.input_cost_per_1m,
        cached_input_cost_per_1m=args.cached_input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
        cost_unit=args.cost_unit,
    )
    print(f"Wrote {len(written)} patient outputs to {args.out_dir}")
    print(f"Pair logs: {args.pair_log_dir or (args.out_dir / 'pair_logs')}")


if __name__ == "__main__":
    main()
