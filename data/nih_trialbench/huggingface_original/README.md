

# TrialGPT 2.0 NIH-TrialBench

## 📖 Overview

NIH-TrialBench is a synthetic patient-trial matching benchmark for evaluating
methods that match patient vignettes to clinical trials. Each synthetic patient
has one target/reference trial ID, and a reviewed subset of patient-trial pairs
contains clinician annotations.

The release contains:

- Synthetic patient summaries.
- One target/reference trial ID per synthetic patient.
- Clinician annotations for reviewed candidate trials.
- A trial search space containing NCT IDs, trial titles, raw eligibility criteria, and study type.


## 📦 Dataset Structure

The dataset is organized as three JSONL files:

| Config | File | Rows | Unit of observation | Description |
| --- | --- | ---: | --- | --- |
| `patients` | `patients.jsonl` | 126 | Synthetic patient | Synthetic patient summaries with target/reference trial IDs. |
| `clinician_annotations` | `clinician_annotations.jsonl` | 126 | Patient-level annotation record | Nested clinician-labeled candidate trials, containing 990 clinician-labeled patient-trial pairs from the evaluated subset. |
| `trial_search_space` | `trial_search_space.jsonl` | 1,373 | Trial | Trial search space with NCT IDs, titles, raw eligibility criteria, and study type. |

<details>
  <summary><b>patients.jsonl</b></summary>
  
| Field | Description |
| --- | --- |
| `patient_id` | Synthetic patient identifier. |
| `ic` | NIH institute or center associated with the synthetic case. |
| `patient_summary` | Synthetic clinical summary. |
| `target_trial_id` | NCT ID of the target/reference trial for the synthetic patient. |

</details>


<details>
  <summary><b>clinician_annotations.jsonl</b></summary>
  
| Field | Description |
| --- | --- |
| `patient_id` | Release-stable synthetic patient identifier matching `patients.jsonl`. |
| `ic` | NIH institute or center associated with the synthetic case. |
| `annotations` | Clinician-labeled trial list for the patient. |
| `annotations[].trial_id` | NCT ID matching `trial_search_space.jsonl`. |
| `annotations[].clinician_label` | Clinician label. |
| `annotations[].clinician_comment` | Optional clinician free-text comment; empty string if no comment was provided. |

Clinician labels:

- `recommended`: clinician judged the trial as recommended for the patient.
- `eligible_but_not_recommended`: clinician judged the patient eligible but did not recommend the trial.
- `ineligible`: clinician judged the patient ineligible.

</details>


<details>
  <summary><b>trial_search_space.jsonl</b></summary>
  
| Field | Description |
| --- | --- |
| `trial_id` | NCT ID. |
| `trial_title` | Trial title. |
| `raw_criteria` | Raw eligibility criteria text. |
| `study_type` | Trial study type. |

`trial_search_space.jsonl` is a trial-level search-space file. It is not a
per-patient candidate list. The TrialGPT 2.0 repository derives per-vignette
candidate search spaces from these trials and the target trial study type.

</details>



## 📜 Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, DIR/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NLM's disclaimer policy is available.

## 🫱🏻‍🫲 Acknowledgements

We appreciate [TrialGPT](https://github.com/ncbi-nlp/TrialGPT) for their open-source contributions.
This research was supported by the Division of Intramural Research (DIR) of the National Library of Medicine (NLM), National Institutes of Health.
