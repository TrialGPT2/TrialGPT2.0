#!/usr/bin/env bash
set -euo pipefail

CASES_PATH="${CASES_PATH:?Set CASES_PATH to a benchmark_cases.jsonl file.}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for TrialGPT 2.0 full_email_*.json outputs.}"
MODEL="${MODEL:-gpt-5.4}"
NOTE_FIELD="${NOTE_FIELD:-full_note}"
NPROCS="${NPROCS:-32}"
TRIAL_DIR="${TRIAL_DIR:-}"

TRIAL_DIR_ARGS=()
if [[ -n "$TRIAL_DIR" ]]; then
  TRIAL_DIR_ARGS=(--trial_dir "$TRIAL_DIR")
fi

python -m src.trial_matching.run_matcher_with_usage_logging \
  --cases_path "$CASES_PATH" \
  "${TRIAL_DIR_ARGS[@]}" \
  --out_dir "$OUT_DIR" \
  --note_field "$NOTE_FIELD" \
  --model "$MODEL" \
  --nprocs "$NPROCS" \
  --resume \
  --show_progress \
  --continue_on_error
