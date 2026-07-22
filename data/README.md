# Benchmark Data

This directory contains the benchmark datasets used by the matching and
metric scripts:

- `nih_trialbench/`: NIH TrialBench matcher cases, per-vignette trial search spaces, and trial JSON files.
- `sigir/`: SIGIR matcher cases, queries, qrels, and trial JSON files.
- `trec_2021/`: TREC 2021 matcher cases, queries, qrels, and trial JSON files.
- `trec_2022/`: TREC 2022 matcher cases, queries, qrels, and trial JSON files.

Each dataset's `benchmark_cases.jsonl` file stores compact matcher cases with
`patient_id`, `patient_summary`, `candidate_trial_count`, and
`candidate_trial_ids`. The matcher loads full trial details from the same dataset
directory's `trials/` folder.

Note: SIGIR has 59 query/case rows, but only 58 qrels-evaluable patients.
`sigir-201428` has no judged candidate trials and its matching output contains an
empty ranking.

NIH TrialBench was built from the NIH-Syn Hugging Face export. Trial objects are
copied from NIH update trial JSON files with source fields and field order
preserved. For NHGRI patients, the matching search space includes all study
types. For all other patients, the search space includes trials with the same
study type as the patient's target/reference trial. The compact search-space
mapping is `nih_trialbench/case_search_space.jsonl`, with one row per vignette
and a `candidate_trial_ids` list.
NIH TrialBench evaluation is target-trial recovery: each vignette has one
`target_trial_id`, and the evaluator checks where that target trial appears in
the model output.
