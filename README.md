# TrialGPT 2.0

This folder contains the TrialGPT 2.0 code needed for:

1. Matching patients to candidate clinical trials.
2. Computing evaluation summaries: qrels ranking metrics for SIGIR/TREC and target-trial recovery/Recall@10 for NIH TrialBench.

GPT-5.4, GPT-5.2, and GPT-4.1 use the same matching entrypoint; choose the model with the `--model` argument.

## Layout

- `src/trial_matching/run_matcher_with_usage_logging.py`: matching CLI.
- `src/trial_matching/matcher_with_usage_logging.py`: core matching implementation.
- `src/trial_matching/prompts/trial_matcher.py`: TrialGPT 2.0 matching prompt.
- `src/llm/client.py`: Azure OpenAI/OpenAI chat client.
- `llm_pair_metrics.py`: token/timing capture helper.
- `scripts/eval_trialgpt_rankings_qrels.py`: qrels-based metric computation for SIGIR/TREC `full_email_*.json` rankings.
- `scripts/eval_nih_trialbench_target_recovery.py`: NIH TrialBench target-trial category recovery.
- `scripts/eval_nih_trialbench_target_recall_at10.py`: NIH TrialBench target-trial Recall@10 evaluation.
- `scripts/build_nih_trialbench.py`: build NIH TrialBench matcher/evaluation files from the NIH-Syn Hugging Face export.
- `examples/env.example.sh`: environment variable template.
- `examples/run_matching.sh`: generic matching runner using `MODEL` as a hyperparameter.
- `data/`: NIH TrialBench, SIGIR, TREC 2021, and TREC 2022 benchmark data plus prepared matcher cases.

## Benchmarks

The repository includes four benchmark datasets:

- `data/nih_trialbench`
- `data/sigir`
- `data/trec_2021`
- `data/trec_2022`

Each dataset directory contains a compact `benchmark_cases.jsonl` file for the
matching CLI. Case rows store `patient_id`, `patient_summary`,
`candidate_trial_count`, and `candidate_trial_ids`; the matcher loads full trial
details from the same directory's `trials/` folder.

SIGIR has 59 query/case rows, but 58 qrels-evaluable patients. The extra case,
`sigir-201428`, has no judged candidate trials and produces an empty ranking.

NIH TrialBench was built from the NIH-Syn Hugging Face export with 126 patients
and 1,373 trials. Trial objects are copied from the NIH update trial JSON files
with source fields and field order preserved. NHGRI patients use all study types as their
matching search space; all other patients use trials with the same study type as
the target/reference trial. NIH TrialBench evaluation is target-trial recovery:
each case marks only its `target_trial_id` as the gold target, and the report is
the percentage of target trials assigned to highly recommended, possible match,
or low fit. The compact per-vignette trial search spaces are in
`data/nih_trialbench/case_search_space.jsonl`.

To rebuild NIH TrialBench files from a local NIH-Syn export:

```bash
python scripts/build_nih_trialbench.py \
  --source-dir /path/to/huggingface \
  --trial-dir /path/to/NIH_update_trials
```

## Environment

Use Python 3.10 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

For Azure OpenAI:

```bash
export LLM_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<your-api-key>"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

For OpenAI:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="<your-api-key>"
```

## Run Matching

Run directly:

```bash
python -m src.trial_matching.run_matcher_with_usage_logging \
  --cases_path data/trec_2022/benchmark_cases.jsonl \
  --out_dir outputs/trec_2022_trial_matcher_full \
  --note_field full_note \
  --model gpt-5.4 \
  --nprocs 32 \
  --resume \
  --show_progress \
  --continue_on_error
```

Use `--model gpt-5.2` or `--model gpt-4.1` for the other model runs.

The input `benchmark_cases.jsonl` is expected to contain `patient_id`,
`patient_summary`, `candidate_trial_count`, and `candidate_trial_ids`. The output
directory will contain one `full_email_<patient_id>.json` file per patient plus
`pair_logs/`.

You can also use the generic example runner:

```bash
MODEL=gpt-5.4 \
CASES_PATH=data/trec_2022/benchmark_cases.jsonl \
OUT_DIR=outputs/trec_2022_trial_matcher_full \
bash examples/run_matching.sh
```

## Compute Ranking Metrics

```bash
python scripts/eval_trialgpt_rankings_qrels.py \
  --queries-jsonl data/trec_2022/queries.jsonl \
  --qrels data/trec_2022/qrels/test.tsv \
  --trial-dir data/trec_2022/trials \
  --results-dir outputs/trec_2022_trial_matcher_full \
  --k 10 \
  --max-rel 2 \
  --mrr-relevance-threshold 2 \
  --subtrial-agg first \
  --pad-to-pool
```

The evaluator reports only MRR, GradedAP, GradedPrecision@10, and GradednDCG@10.

## Compute NIH TrialBench Target Recovery

```bash
python scripts/eval_nih_trialbench_target_recovery.py \
  --cases-path data/nih_trialbench/benchmark_cases.jsonl \
  --results-dir outputs/nih_trialbench_trial_matcher_full_gpt41 \
  --out-dir outputs/nih_trialbench_trial_matcher_full_gpt41_target_recovery
```

The NIH evaluator uses the same category thresholds as the NCI target-trial
distribution analysis: `final_score > 90` is highly recommended, `80 <=
final_score <= 90` is possible match, and `final_score < 80` is low fit.

For TrialGPT 1.0 ranked outputs, use the aggregation-derived 0-100 fit score
instead of the ranked file's `final_score`:

```bash
python scripts/eval_nih_trialbench_target_recovery.py \
  --cases-path data/nih_trialbench/benchmark_cases.jsonl \
  --interventional-results-dir /path/to/interventional/ranked_full_email \
  --observational-results-dir /path/to/observational/ranked_full_email \
  --score-field trialgpt1_fit_score \
  --out-dir outputs/nih_trialbench_trialgpt1_target_recovery
```

In this mode the category score is `(relevance_score_R + eligibility_score_E) /
2`; the category thresholds are otherwise unchanged.

## Compute NIH TrialBench Target-Trial Recall@10

```bash
python scripts/eval_nih_trialbench_target_recall_at10.py \
  --cases-path data/nih_trialbench/benchmark_cases.jsonl \
  --results-dir outputs/nih_trialbench_trial_matcher_full_gpt41 \
  --k 10 \
  --out-dir outputs/nih_trialbench_trial_matcher_full_gpt41_target_recall_at10
```

This evaluator reports the fraction of patient queries where the
`target_trial_id` appears in the top 10 ranked trials. Missing output files and
targets absent from the ranked list count as misses. It writes per-case target
ranks plus overall, IC-level, and study-type summaries.

For TrialGPT 1.0 runs split by study type:

```bash
python scripts/eval_nih_trialbench_target_recall_at10.py \
  --cases-path data/nih_trialbench/benchmark_cases.jsonl \
  --interventional-results-dir /path/to/interventional/ranked_full_email \
  --observational-results-dir /path/to/observational/ranked_full_email \
  --k 10 \
  --out-dir outputs/nih_trialbench_trialgpt1_target_recall_at10
```
