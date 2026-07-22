from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.llm.client import LLMClient, _retry_sleep_s, _safe_json_loads
from src.trial_matching.prompts.trial_matcher import build_trial_matcher_messages

from llm_pair_metrics import call_attempt, calls_subset, capture_chat_completion_metrics, metric_subset
from tqdm import tqdm


NOTE_FIELD_TO_CASE_KEY = {
    "brief_note": "patient_summary",
    "full_note": "patient_summary",
    "partial_note": "patient_summary",
    "stage1_trial_matching_summary": "patient_summary",
}

_THREAD_LOCAL = threading.local()
_LOG_LOCK = threading.Lock()

DEFAULT_INPUT_COST_PER_1M = 62.5
DEFAULT_CACHED_INPUT_COST_PER_1M = 6.25
DEFAULT_OUTPUT_COST_PER_1M = 375.0
DEFAULT_COST_UNIT = "credits"


def trial_id_of(trial: Dict[str, Any]) -> str:
    return str(trial.get("trial_id") or trial.get("nctid") or "").strip()


def trial_title_of(trial: Dict[str, Any]) -> str:
    return str(
        trial.get("brief_title")
        or trial.get("official_title")
        or trial.get("trial_title")
        or trial.get("title")
        or ""
    ).strip()


def trial_criteria_of(trial: Dict[str, Any]) -> str:
    return str(trial.get("eligibility_text") or trial.get("raw_criteria") or "").strip()


class LLMCallFailure(RuntimeError):
    def __init__(self, message: str, runtime: Dict[str, Any]):
        super().__init__(message)
        self.runtime = runtime


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def candidate_trial_ids_of(case: Dict[str, Any]) -> List[str]:
    raw_ids = case.get("candidate_trial_ids")
    if raw_ids is None:
        raw_ids = case.get("trial_ids")
    if raw_ids is None:
        return []
    return [str(trial_id).strip() for trial_id in raw_ids if str(trial_id).strip()]


def load_trial_by_id(trial_id: str, trial_dir: Path, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if trial_id in cache:
        return cache[trial_id]
    trial_path = trial_dir / f"{trial_id}.json"
    if not trial_path.exists():
        raise FileNotFoundError(f"Candidate trial {trial_id!r} not found at {trial_path}")
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    loaded_trial_id = trial_id_of(trial)
    if loaded_trial_id and loaded_trial_id != trial_id:
        raise ValueError(f"Trial ID mismatch in {trial_path}: expected {trial_id}, found {loaded_trial_id}")
    cache[trial_id] = trial
    return trial


def materialize_candidate_trials(
    case: Dict[str, Any],
    *,
    trial_dir: Optional[Path],
    trial_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if "candidate_trials" in case and case.get("candidate_trials") is not None:
        return case

    candidate_trial_ids = candidate_trial_ids_of(case)
    if not candidate_trial_ids:
        return {**case, "candidate_trials": []}
    if trial_dir is None:
        raise ValueError(
            "Case uses candidate_trial_ids but no trial_dir was provided. "
            "Pass --trial_dir or place cases next to a trials/ directory."
        )

    return {
        **case,
        "candidate_trials": [
            load_trial_by_id(trial_id, trial_dir, trial_cache)
            for trial_id in candidate_trial_ids
        ],
    }


def patient_note_for_case(case: Dict[str, Any], note_field: str) -> str:
    case_key = NOTE_FIELD_TO_CASE_KEY[note_field]
    value = case.get(case_key)
    if not value:
        value = case.get("patient_summary") or case.get("full_context") or case.get("brief_note") or case.get("partial_note")
    return str(value or "").strip()


def response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trial_match_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "eligible_reasons": {"type": "string"},
                    "missing_information": {"type": "string"},
                    "ineligible_reasons": {"type": "string"},
                    "rationale": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": [
                    "eligible_reasons",
                    "missing_information",
                    "ineligible_reasons",
                    "rationale",
                    "score",
                    "confidence",
                ],
            },
        },
    }


def get_llm(model: Optional[str] = None) -> LLMClient:
    cached = getattr(_THREAD_LOCAL, "llm", None)
    cache_key = getattr(_THREAD_LOCAL, "cache_key", None)
    desired_key = model or "__env__"
    if cached is not None and cache_key == desired_key:
        return cached

    llm = LLMClient.from_env(model_default=model or "gpt-4o-mini")
    if model and llm.cfg.model != model:
        llm = LLMClient(replace(llm.cfg, model=model))

    _THREAD_LOCAL.llm = llm
    _THREAD_LOCAL.cache_key = desired_key
    return llm


def normalize_match_response(obj: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "eligible_reasons",
        "missing_information",
        "ineligible_reasons",
        "rationale",
        "score",
        "confidence",
    ]
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    score = int(obj["score"])
    confidence = float(obj["confidence"])
    return {
        "eligible_reasons": str(obj["eligible_reasons"]).strip(),
        "missing_information": str(obj["missing_information"]).strip(),
        "ineligible_reasons": str(obj["ineligible_reasons"]).strip(),
        "rationale": str(obj["rationale"]).strip(),
        "score": max(0, min(100, score)),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
    }


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    out: Dict[str, Any] = {}
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    ]:
        value = getattr(usage, key, None)
        if value is not None:
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            out[key] = value
    return out


def _token_count(usage: Dict[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except Exception:
        return 0


def _cached_prompt_tokens(usage: Dict[str, Any]) -> int:
    details = usage.get("prompt_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("cached_tokens") or details.get("cached_prompt_tokens") or 0)
    except Exception:
        return 0


def estimate_cost(
    usage: Dict[str, Any],
    *,
    input_cost_per_1m: float,
    cached_input_cost_per_1m: float,
    output_cost_per_1m: float,
    cost_unit: str,
) -> Dict[str, Any]:
    prompt_tokens = _token_count(usage, "prompt_tokens")
    completion_tokens = _token_count(usage, "completion_tokens")
    cached_tokens = min(_cached_prompt_tokens(usage), prompt_tokens)
    uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)

    input_cost = (
        (uncached_prompt_tokens * input_cost_per_1m)
        + (cached_tokens * cached_input_cost_per_1m)
    ) / 1_000_000.0
    output_cost = (completion_tokens * output_cost_per_1m) / 1_000_000.0
    total_cost = input_cost + output_cost
    return {
        "unit": cost_unit,
        "input_cost_per_1m": input_cost_per_1m,
        "cached_input_cost_per_1m": cached_input_cost_per_1m,
        "output_cost_per_1m": output_cost_per_1m,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "input_cost": round(input_cost, 10),
        "output_cost": round(output_cost, 10),
        "total_cost": round(total_cost, 10),
    }


def _estimate_cost_from_metrics(
    metrics: Dict[str, Any],
    *,
    cached_prompt_tokens: int,
    input_cost_per_1m: float,
    cached_input_cost_per_1m: float,
    output_cost_per_1m: float,
    cost_unit: str,
) -> Dict[str, Any]:
    input_tokens = int(metrics.get("input_tokens") or 0)
    output_tokens = int(metrics.get("output_tokens") or 0)
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": int(metrics.get("total_tokens") or (input_tokens + output_tokens)),
        "prompt_tokens_details": {"cached_tokens": min(cached_prompt_tokens, input_tokens)},
    }
    return estimate_cost(
        usage,
        input_cost_per_1m=input_cost_per_1m,
        cached_input_cost_per_1m=cached_input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        cost_unit=cost_unit,
    )


def _runtime_from_trialgpt_metrics(
    metrics: Dict[str, Any],
    *,
    started: float,
    attempts: int,
    usage: Dict[str, Any],
    input_cost_per_1m: float,
    cached_input_cost_per_1m: float,
    output_cost_per_1m: float,
    cost_unit: str,
) -> Dict[str, Any]:
    subset = metric_subset(metrics)
    cached_prompt_tokens = _cached_prompt_tokens(usage)
    cost = _estimate_cost_from_metrics(
        subset,
        cached_prompt_tokens=cached_prompt_tokens if int(subset.get("num_llm_calls") or 0) == 1 else 0,
        input_cost_per_1m=input_cost_per_1m,
        cached_input_cost_per_1m=cached_input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        cost_unit=cost_unit,
    )
    input_tokens = int(subset.get("input_tokens") or 0)
    output_tokens = int(subset.get("output_tokens") or 0)
    total_tokens = int(subset.get("total_tokens") or 0)
    return {
        "elapsed_s": round(time.perf_counter() - started, 6),
        "latency_sec": float(subset.get("latency_sec") or 0.0),
        "num_llm_calls": int(subset.get("num_llm_calls") or 0),
        "attempts": attempts,
        "attempt_count": attempts,
        "retry_count": max(0, attempts - 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "estimated_cost": cost,
        "usage": usage,
        "llm_calls": calls_subset(metrics),
    }


def chat_with_usage(
    llm: LLMClient,
    messages: List[Dict[str, str]],
    *,
    response_format_obj: Dict[str, Any],
    omit_temperature: bool = False,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    cached_input_cost_per_1m: float = DEFAULT_CACHED_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
    cost_unit: str = DEFAULT_COST_UNIT,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = llm.cfg
    started = time.perf_counter()
    attempts = 0
    last_err: Optional[Exception] = None
    parsed: Optional[Dict[str, Any]] = None
    usage: Dict[str, Any] = {}

    with capture_chat_completion_metrics(llm._client.chat.completions, stage="trialagents_matcher") as metrics:
        for attempt in range(cfg.max_retries + 1):
            attempts = attempt + 1
            try:
                kwargs: Dict[str, Any] = {
                    "model": cfg.model,
                    "messages": messages,
                    "response_format": response_format_obj,
                    "timeout": cfg.timeout_s,
                }
                if not omit_temperature:
                    kwargs["temperature"] = cfg.temperature
                if cfg.max_tokens is not None:
                    kwargs["max_tokens"] = cfg.max_tokens

                with call_attempt(metrics, attempts):
                    resp = llm._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                maybe_parsed = _safe_json_loads(content)
                if not isinstance(maybe_parsed, dict):
                    raise ValueError(f"Expected JSON object from matcher, got raw content: {content[:200]!r}")

                usage = _usage_to_dict(getattr(resp, "usage", None))
                parsed = maybe_parsed
                break
            except Exception as e:
                last_err = e
                if attempt >= cfg.max_retries:
                    break
                time.sleep(_retry_sleep_s(cfg, e, attempt))

    runtime = _runtime_from_trialgpt_metrics(
        metrics,
        started=started,
        attempts=attempts,
        usage=usage,
        input_cost_per_1m=input_cost_per_1m,
        cached_input_cost_per_1m=cached_input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        cost_unit=cost_unit,
    )
    if parsed is not None:
        return parsed, usage, runtime

    raise LLMCallFailure(
        f"LLM call failed after retries (elapsed_s={runtime['elapsed_s']}, attempts={attempts}): {last_err}",
        runtime,
    ) from last_err


def score_trial(
    *,
    patient_note: str,
    trial: Dict[str, Any],
    model: Optional[str] = None,
    omit_temperature: bool = False,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    cached_input_cost_per_1m: float = DEFAULT_CACHED_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
    cost_unit: str = DEFAULT_COST_UNIT,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    messages = build_trial_matcher_messages(
        title=trial_title_of(trial),
        brief_summary=str(trial.get("brief_summary") or "").strip(),
        raw_criteria=trial_criteria_of(trial),
        patient_note=patient_note,
    )
    response, _, runtime = chat_with_usage(
        get_llm(model),
        messages,
        response_format_obj=response_format(),
        omit_temperature=omit_temperature,
        input_cost_per_1m=input_cost_per_1m,
        cached_input_cost_per_1m=cached_input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        cost_unit=cost_unit,
    )
    normalized = normalize_match_response(response)
    normalized["final_score"] = normalized["score"] + normalized["confidence"]
    return normalized, runtime


def patient_output_path(out_dir: Path, patient_id: str) -> Path:
    return out_dir / f"full_email_{patient_id}.json"


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _runtime_log_fields(runtime: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "elapsed_s": runtime.get("elapsed_s"),
        "latency_sec": runtime.get("latency_sec"),
        "num_llm_calls": runtime.get("num_llm_calls"),
        "attempts": runtime.get("attempts"),
        "attempt_count": runtime.get("attempt_count"),
        "retry_count": runtime.get("retry_count"),
        "input_tokens": runtime.get("input_tokens"),
        "output_tokens": runtime.get("output_tokens"),
        "prompt_tokens": runtime.get("prompt_tokens"),
        "completion_tokens": runtime.get("completion_tokens"),
        "total_tokens": runtime.get("total_tokens"),
        "cached_prompt_tokens": runtime.get("cached_prompt_tokens"),
        "estimated_cost": runtime.get("estimated_cost", {}),
        "usage": runtime.get("usage", {}),
        "llm_calls": runtime.get("llm_calls", []),
    }


def _empty_token_totals() -> Dict[str, float]:
    return {
        "sum_pair_elapsed_s": 0.0,
        "latency_sec": 0.0,
        "num_llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
        "estimated_input_cost": 0.0,
        "estimated_output_cost": 0.0,
        "estimated_total_cost": 0.0,
    }


def _add_runtime(totals: Dict[str, float], runtime: Dict[str, Any]) -> None:
    totals["sum_pair_elapsed_s"] += float(runtime.get("elapsed_s") or 0.0)
    totals["latency_sec"] += float(runtime.get("latency_sec") or 0.0)
    totals["num_llm_calls"] += int(runtime.get("num_llm_calls") or 0)
    totals["input_tokens"] += int(runtime.get("input_tokens") or 0)
    totals["output_tokens"] += int(runtime.get("output_tokens") or 0)
    totals["prompt_tokens"] += int(runtime.get("prompt_tokens") or 0)
    totals["completion_tokens"] += int(runtime.get("completion_tokens") or 0)
    totals["total_tokens"] += int(runtime.get("total_tokens") or 0)
    totals["cached_prompt_tokens"] += int(runtime.get("cached_prompt_tokens") or 0)
    cost = runtime.get("estimated_cost") or {}
    if isinstance(cost, dict):
        totals["estimated_input_cost"] += float(cost.get("input_cost") or 0.0)
        totals["estimated_output_cost"] += float(cost.get("output_cost") or 0.0)
        totals["estimated_total_cost"] += float(cost.get("total_cost") or 0.0)


def evaluate_case(
    case: Dict[str, Any],
    *,
    out_dir: Path,
    note_field: str,
    model: Optional[str],
    nprocs: int,
    show_progress: bool = False,
    continue_on_error: bool = False,
    pair_log_dir: Optional[Path] = None,
    omit_temperature: bool = False,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    cached_input_cost_per_1m: float = DEFAULT_CACHED_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
    cost_unit: str = DEFAULT_COST_UNIT,
) -> Path:
    patient_id = str(case.get("patient_id", "")).strip()
    patient_note = patient_note_for_case(case, note_field)
    candidate_trials = list(case.get("candidate_trials") or [])
    gold_labels = {str(k): int(v) for k, v in (case.get("gold_labels") or {}).items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = patient_output_path(out_dir, patient_id)
    log_dir = pair_log_dir or (out_dir / "pair_logs")
    pair_log_path = log_dir / f"{patient_id}.jsonl"

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    token_totals = _empty_token_totals()
    patient_started = time.perf_counter()

    if show_progress:
        print(
            f"Evaluating patient {patient_id} "
            f"(note_field={note_field}, trials={len(candidate_trials)}, nprocs={max(1, nprocs)})"
        )

    def task(trial: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any], Optional[Exception]]:
        trial_id = trial_id_of(trial)
        trial_started = time.perf_counter()
        base_log: Dict[str, Any] = {
            "patient_id": patient_id,
            "trial_id": trial_id,
            "note_field": note_field,
            "model": model or get_llm().cfg.model,
        }
        try:
            result, runtime = score_trial(
                patient_note=patient_note,
                trial=trial,
                model=model,
                omit_temperature=omit_temperature,
                input_cost_per_1m=input_cost_per_1m,
                cached_input_cost_per_1m=cached_input_cost_per_1m,
                output_cost_per_1m=output_cost_per_1m,
                cost_unit=cost_unit,
            )
            result.update(
                {
                    "trial_id": trial_id,
                    "trial_title": trial_title_of(trial),
                    "study_type": str(trial.get("study_type") or "").strip(),
                    "latency_sec": runtime.get("latency_sec"),
                    "num_llm_calls": runtime.get("num_llm_calls"),
                    "input_tokens": runtime.get("input_tokens"),
                    "output_tokens": runtime.get("output_tokens"),
                    "total_tokens": runtime.get("total_tokens"),
                    "llm_calls": runtime.get("llm_calls", []),
                    "attempt_count": runtime.get("attempt_count"),
                    "retry_count": runtime.get("retry_count"),
                    "runtime": runtime,
                }
            )
            if gold_labels:
                result["gold_label"] = gold_labels.get(trial_id)
            pair_log = {
                **base_log,
                "status": "ok",
                "trialgpt_status": "success",
                **_runtime_log_fields(runtime),
                "score": result.get("score"),
                "confidence": result.get("confidence"),
                "final_score": result.get("final_score"),
            }
            if gold_labels:
                pair_log["gold_label"] = result.get("gold_label")
            return trial_id, result, pair_log, None
        except Exception as e:
            runtime = getattr(e, "runtime", {}) or {}
            pair_log = {
                **base_log,
                "status": "error",
                "trialgpt_status": "failure",
                **_runtime_log_fields(runtime),
                "elapsed_s": runtime.get("elapsed_s") or round(time.perf_counter() - trial_started, 6),
                "error": str(e),
            }
            return trial_id, None, pair_log, e

    with ThreadPoolExecutor(max_workers=max(1, nprocs)) as ex:
        futures = [ex.submit(task, trial) for trial in candidate_trials]
        iterator = as_completed(futures)
        progress = None
        if show_progress:
            progress = tqdm(iterator, total=len(futures), desc=f"Trials for {patient_id}")
            iterator = progress

        try:
            for future in iterator:
                trial_id, result, pair_log, error = future.result()
                _append_jsonl(pair_log_path, pair_log)
                if error is not None:
                    if not continue_on_error:
                        raise error
                    failures.append({"trial_id": trial_id, "error": str(error)})
                    if show_progress:
                        print(f"Failed trial for {patient_id}/{trial_id}: {error}")
                    continue

                if result is not None:
                    results.append(result)
                    _add_runtime(token_totals, result.get("runtime", {}))
        finally:
            if progress is not None:
                progress.close()

    results.sort(key=lambda row: (-row["final_score"], row["trial_id"]))
    token_totals["sum_pair_elapsed_s"] = round(float(token_totals["sum_pair_elapsed_s"]), 6)
    token_totals["latency_sec"] = round(float(token_totals["latency_sec"]), 3)
    for key in ["estimated_input_cost", "estimated_output_cost", "estimated_total_cost"]:
        token_totals[key] = round(float(token_totals[key]), 10)
    payload = {
        "patient_ID": patient_id,
        "note_field": note_field,
        "patient_summary": patient_note,
        "cover_page_list": results,
        "metadata": {
            "model": model or get_llm().cfg.model,
            "num_trials": len(candidate_trials),
            "num_scored_trials": len(results),
            "num_failed_trials": len(failures),
            "pair_log_path": str(pair_log_path),
            "cost_rates": {
                "unit": cost_unit,
                "input_cost_per_1m": input_cost_per_1m,
                "cached_input_cost_per_1m": cached_input_cost_per_1m,
                "output_cost_per_1m": output_cost_per_1m,
            },
            "runtime": {
                "patient_wall_time_s": round(time.perf_counter() - patient_started, 6),
                **token_totals,
            },
        },
    }
    if failures:
        payload["failures"] = failures
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if show_progress:
        print(
            f"Saved {out_path} "
            f"(scored={len(results)}, failed={len(failures)}, log={pair_log_path})"
        )
    return out_path


def run(
    *,
    cases_path: Path,
    out_dir: Path,
    note_field: str,
    model: Optional[str],
    nprocs: int,
    resume: bool,
    show_progress: bool = False,
    continue_on_error: bool = False,
    trial_dir: Optional[Path] = None,
    pair_log_dir: Optional[Path] = None,
    omit_temperature: bool = False,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    cached_input_cost_per_1m: float = DEFAULT_CACHED_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
    cost_unit: str = DEFAULT_COST_UNIT,
) -> List[Path]:
    if note_field not in NOTE_FIELD_TO_CASE_KEY:
        raise ValueError(f"Unsupported note field: {note_field}")

    cases = list(read_jsonl(cases_path))
    if trial_dir is None and any(
        "candidate_trials" not in case and candidate_trial_ids_of(case)
        for case in cases
    ):
        trial_dir = cases_path.parent / "trials"
    written: List[Path] = []
    trial_cache: Dict[str, Dict[str, Any]] = {}
    if show_progress:
        trial_dir_msg = f", trial_dir={trial_dir}" if trial_dir is not None else ""
        print(
            f"Loaded {len(cases)} cases from {cases_path} "
            f"(note_field={note_field}, resume={resume}{trial_dir_msg})"
        )
    for case in cases:
        patient_id = str(case.get("patient_id", "")).strip()
        out_path = patient_output_path(out_dir, patient_id)
        if resume and out_path.exists():
            if show_progress:
                print(f"Skipping {patient_id}, already complete")
            continue
        case = materialize_candidate_trials(
            case,
            trial_dir=trial_dir,
            trial_cache=trial_cache,
        )
        written.append(
            evaluate_case(
                case,
                out_dir=out_dir,
                note_field=note_field,
                model=model,
                nprocs=nprocs,
                show_progress=show_progress,
                continue_on_error=continue_on_error,
                pair_log_dir=pair_log_dir,
                omit_temperature=omit_temperature,
                input_cost_per_1m=input_cost_per_1m,
                cached_input_cost_per_1m=cached_input_cost_per_1m,
                output_cost_per_1m=output_cost_per_1m,
                cost_unit=cost_unit,
            )
        )
    return written
