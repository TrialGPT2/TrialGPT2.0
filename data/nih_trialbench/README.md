# NIH TrialBench

This directory was generated from the NIH-Syn Hugging Face release for TrialGPT 2.0.

Trial objects were copied from the NIH update trial JSON directory with source fields and field order preserved.

Search-space rule:
- NHGRI patients use all trial study types.
- All other patients use trials with the same study type as the patient's target/reference trial.

Files:
- `benchmark_cases.jsonl`: compact matcher input with candidate trial IDs.
- `queries.jsonl`: patient IDs and summaries used by the evaluator.
- `case_search_space.jsonl`: per-vignette trial search-space IDs used for matching.
- `trials/`: exact trial JSON files copied from the NIH update trial directory.

Evaluation is target-trial recovery. Each `benchmark_cases.jsonl` row includes the vignette's `target_trial_id`; no qrels or clinician annotation labels are packaged.
