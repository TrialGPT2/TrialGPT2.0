<h1 align="center">TrialGPT 2.0</h1>

<p align="center">
  an AI-assisted patient-to-trial matching system
</p>

<p align="center">
  <a href="https://github.com/NLM-DIR/TrialGPT2"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://www.ncbi.nlm.nih.gov/research/trialgpt"><img src="https://img.shields.io/badge/Interface-TrialGPT%202.0-2EA44F?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Interface"></a>
  <a href="https://huggingface.co/datasets/ncbi/TrialGPT2-NIH-Syn"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a>
</p>



## 📑 Contents

- [📖 Overview](#1)
- [⚙️ Installation](#2)
- [📦 Data](#3)
- [⚡ Quick Start](#4)
- [🚀 Full Matching and Evaluation Workflow](#5)
  - [🔐 Step 1: Configure the LLM backend](#5-1)
  - [🧠 Step 2: Run TrialGPT 2.0 matching](#5-2)
  - [📄 Step 3: Inspect ranked outputs and pair logs](#5-3)
  - [📊 Step 4: Evaluate SIGIR/TREC rankings](#5-4)
  - [🎯 Step 5: Evaluate NIH TrialBench target recovery](#5-5)
- [🧱 Rebuild NIH TrialBench Files](#6)
- [📁 Repository Layout](#7)
- [🗺️ Project Info](#8)
  - [📜 Manuscript](#8-1)

<h2 id="1">📖 Overview</h2>

TrialGPT 2.0 extends the [original TrialGPT framework](https://github.com/ncbi-nlp/TrialGPT) beyond eligibility-focused matching to support clinical trial recommendation in real-world review workflows.
A publicly available web interface is available at the [TrialGPT 2.0 Interface](https://www.ncbi.nlm.nih.gov/research/trialgpt).


This repository contains:

- The TrialGPT 2.0 matching prompt and matching runner.
- Azure OpenAI/OpenAI client code with timing, token, and estimated cost logging.
- Prepared benchmark files for NIH TrialBench, SIGIR, TREC 2021, and TREC 2022.
- Evaluation scripts for qrels-based ranking metrics and NIH target-trial recovery.


<h2 id="2">⚙️ Installation</h2>

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The required Python packages are intentionally minimal:

```text
openai>=1.0.0
tqdm>=4.0.0
```

<h2 id="3">📦 Data</h2>

The repository includes four packaged benchmark directories under `data/`:

| Benchmark | Cases | Trial files | Patient-trial pairs | Evaluation type |
| --- | ---: | ---: | ---: | --- |
| `nih_trialbench` | 126 | 1,373 | 101,682 | Target-trial recovery |
| `sigir` | 59 | 3,593 | 3,835 | Qrels ranking metrics |
| `trec_2021` | 75 | 26,162 | 35,832 | Qrels ranking metrics |
| `trec_2022` | 50 | 26,585 | 35,394 | Qrels ranking metrics |

Each benchmark directory follows the same compact layout:

```text
data/<benchmark>/
  benchmark_cases.jsonl
  case_search_space.jsonl
  queries.jsonl
  qrels/test.tsv          # SIGIR/TREC only
  trials/
    NCT*.json
```

`benchmark_cases.jsonl` is the main matcher input. Each row contains a patient
identifier, a `patient_summary`, and the list of `candidate_trial_ids` defining
that patient's search space. The matcher loads full trial records from the
same benchmark directory's `trials/` folder by default.

For NIH TrialBench, each vignette has one target/reference trial. The per-vignette
search spaces are stored in `data/nih_trialbench/case_search_space.jsonl`.

Note: SIGIR has 59 matcher cases, but 58 qrels-evaluable patients. The case
`sigir-201428` has no judged candidate trials and produces an empty ranking.

<h2 id="4">⚡ Quick Start</h2>

Set your LLM credentials, then run matching on one packaged benchmark.

```bash
export LLM_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<your-api-key>"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export LLM_MODEL="gpt-4.1"
```

```bash
python -m src.trial_matching.run_matcher_with_usage_logging \
  --cases_path data/sigir/benchmark_cases.jsonl \
  --out_dir outputs/sigir_trial_matcher_full_gpt41 \
  --note_field full_note \
  --model gpt-4.1 \
  --nprocs 16 \
  --resume \
  --show_progress \
  --continue_on_error
```

The command writes one ranked file per patient:

```text
outputs/sigir_trial_matcher_full_gpt41/
  full_email_<patient_id>.json
  pair_logs/
```

<h2 id="5">🚀 Full Matching and Evaluation Workflow</h2>

<h3 id="5-1">🔐 Step 1: Configure the LLM backend</h3>

For Azure OpenAI, set:

```bash
export LLM_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<your-api-key>"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export LLM_MODEL="gpt-5.4"
```

`LLM_MODEL` should match your Azure deployment name. You can also pass the model
directly to the matching CLI with `--model`. GPT-5.4, GPT-5.2, and GPT-4.1 use
the same code path and prompt.

For the OpenAI API, set:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="<your-api-key>"
export LLM_MODEL="gpt-4.1"
```

Additional optional controls:

```bash
export LLM_TEMPERATURE="0.0"
export LLM_TIMEOUT_S="60"
export LLM_MAX_RETRIES="3"
export LLM_MAX_TOKENS="0"
```

<h3 id="5-2">🧠 Step 2: Run TrialGPT 2.0 matching</h3>

Run the matcher directly:

```bash
python -m src.trial_matching.run_matcher_with_usage_logging \
  --cases_path data/trec_2022/benchmark_cases.jsonl \
  --out_dir outputs/trec_2022_trial_matcher_full_gpt54 \
  --note_field full_note \
  --model gpt-5.4 \
  --nprocs 32 \
  --resume \
  --show_progress \
  --continue_on_error
```

The matcher workflow is:

1. Load one patient row from `benchmark_cases.jsonl`.
2. Resolve the case search space from `candidate_trial_ids`.
3. Load trial JSON records from `--trial_dir` or `<cases_path parent>/trials`.
4. Apply the TrialGPT 2.0 prompt in `src/trial_matching/prompts/trial_matcher.py`.
5. Score patient-trial pairs in parallel using `--nprocs`.
6. Sort scored trials into a ranked `cover_page_list`.
7. Write patient-level ranked outputs and pair-level logs.

You can also use the generic shell runner:

```bash
MODEL=gpt-5.4 \
CASES_PATH=data/trec_2022/benchmark_cases.jsonl \
OUT_DIR=outputs/trec_2022_trial_matcher_full_gpt54 \
NPROCS=32 \
bash examples/run_matching.sh
```

To switch datasets, change `CASES_PATH` and `OUT_DIR`:

```bash
CASES_PATH=data/sigir/benchmark_cases.jsonl
CASES_PATH=data/trec_2021/benchmark_cases.jsonl
CASES_PATH=data/trec_2022/benchmark_cases.jsonl
CASES_PATH=data/nih_trialbench/benchmark_cases.jsonl
```

<h3 id="5-3">📄 Step 3: Inspect ranked outputs and pair logs</h3>

Each patient output is named:

```text
full_email_<patient_id>.json
```

The main ranked list is stored in `cover_page_list`. Each row contains the
trial identifier, model-derived scores, matching rationale fields, and trial
metadata used by downstream evaluators.

The `pair_logs/` directory stores one JSONL log per patient with per-pair timing,
token usage, estimated cost, parse status, and error information. These logs are
useful for audit and debugging but are not required for metric computation.

<h3 id="5-4">📊 Step 4: Evaluate SIGIR/TREC rankings</h3>

SIGIR, TREC 2021, and TREC 2022 use qrels-based ranking metrics. The evaluator
reports only:

- MRR
- GradedAP
- GradedPrecision@10
- GradednDCG@10

Use the matching benchmark paths for SIGIR or TREC 2021:

```bash
python scripts/eval_trialgpt_rankings_qrels.py \
  --queries-jsonl data/sigir/queries.jsonl \
  --qrels data/sigir/qrels/test.tsv \
  --trial-dir data/sigir/trials \
  --results-dir outputs/sigir_trial_matcher_full_gpt54 \
  --k 10 \
  --max-rel 2 \
  --mrr-relevance-threshold 2 \
  --subtrial-agg first \
  --pad-to-pool
```


<h3 id="5-5">🎯 Step 5: Evaluate NIH TrialBench target recovery</h3>

NIH TrialBench is evaluated by target-trial recovery. Each vignette has one
`target_trial_id`, and the evaluator checks what category TrialGPT assigned to
that target trial.

```bash
python scripts/eval_nih_trialbench_target_recovery.py \
  --cases-path data/nih_trialbench/benchmark_cases.jsonl \
  --results-dir outputs/nih_trialbench_trial_matcher_full_gpt54 \
  --out-dir outputs/nih_trialbench_trial_matcher_full_gpt54_target_recovery
```

The default TrialGPT 2.0 category thresholds are:

- Highly recommended: `final_score > 90`
- Possible match: `80 <= final_score <= 90`
- Low fit: `final_score < 80`

The evaluator writes:

```text
target_recovery_summary.json
target_recovery_per_case.csv
target_recovery_summary_by_group.csv
```

The target-trial Recall@10 evaluator is also provided:

```bash
python scripts/eval_nih_trialbench_target_recall_at10.py \
  --cases-path data/nih_trialbench/benchmark_cases.jsonl \
  --results-dir outputs/nih_trialbench_trial_matcher_full_gpt54 \
  --k 10 \
  --out-dir outputs/nih_trialbench_trial_matcher_full_gpt54_target_recall_at10
```

<h2 id="6">🧱 Rebuild NIH TrialBench Files</h2>

The packaged NIH TrialBench files were built from the NIH-Syn Hugging Face
export and NIH updated trial JSON files. To rebuild them from local sources:

```bash
python scripts/build_nih_trialbench.py \
  --source-dir /path/to/huggingface \
  --trial-dir data/nih_trialbench/trials \
  --output-root data \
  --benchmark-name nih_trialbench
```


<h2 id="7">📁 Repository Layout</h2>

```text
TrialGPT_2.0/
  README.md
  requirements.txt
  llm_pair_metrics.py
  examples/
    env.example.sh
    run_matching.sh
  scripts/
    build_nih_trialbench.py
    eval_nih_trialbench_target_recall_at10.py
    eval_nih_trialbench_target_recovery.py
    eval_trialgpt_rankings_qrels.py
  src/
    llm/
      client.py
    trial_matching/
      matcher_with_usage_logging.py
      run_matcher_with_usage_logging.py
      prompts/
        trial_matcher.py
  data/
    nih_trialbench/
    sigir/
    trec_2021/
    trec_2022/
```

Core files:

- `src/trial_matching/prompts/trial_matcher.py`: TrialGPT 2.0 matching prompt.
- `src/trial_matching/run_matcher_with_usage_logging.py`: matching CLI.
- `src/trial_matching/matcher_with_usage_logging.py`: matching, ranking, parsing, and logging logic.
- `src/llm/client.py`: Azure OpenAI/OpenAI chat client.
- `llm_pair_metrics.py`: timing, token, and estimated cost helper.
- `scripts/eval_trialgpt_rankings_qrels.py`: SIGIR/TREC metric computation.
- `scripts/eval_nih_trialbench_target_recovery.py`: NIH target-trial category recovery.
- `scripts/eval_nih_trialbench_target_recall_at10.py`: optional NIH target-trial Recall@10.

<h2 id="8">🗺️ Project Info</h2>

<h3 id="8-1"> 📜 Disclaimer</h3>

This tool shows the results of research conducted in the Computational Biology Branch, DIR/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NLM's disclaimer policy is available.


<h3 id="8-2"> 📚 References</h3>
If you use our repository, please cite the following related paper:

```

```

<h3 id="8-3"> 🫱🏻‍🫲 Acknowledgements</h3>

We appreciate [TrialGPT](https://github.com/ncbi-nlp/TrialGPT) for their open-source contributions.
This research was supported by the Division of Intramural Research (DIR) of the National Library of Medicine (NLM), National Institutes of Health.
