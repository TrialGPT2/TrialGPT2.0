from __future__ import annotations

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import patch


METRIC_FIELDS = (
    "latency_sec",
    "num_llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def empty_metrics() -> dict:
    return {
        "latency_sec": 0.0,
        "num_llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": [],
    }


def _usage_int(usage, *field_names: str) -> int:
    if usage is None:
        return 0
    for field_name in field_names:
        if isinstance(usage, dict):
            value = usage.get(field_name)
        else:
            value = getattr(usage, field_name, None)
        if value is not None:
            return int(value)
    return 0


def record_response_usage(metrics: dict, response, call_latency_sec: float | None = None, stage: str | None = None) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens

    metrics["num_llm_calls"] += 1
    metrics["input_tokens"] += input_tokens
    metrics["output_tokens"] += output_tokens
    metrics["total_tokens"] += total_tokens

    attempt_index = int(metrics.get("_current_attempt_index", 1) or 1)
    call_record = {
        "call_index": metrics["num_llm_calls"],
        "attempt_index": attempt_index,
        "retry_index": max(0, attempt_index - 1),
        "is_retry": attempt_index > 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if call_latency_sec is not None:
        call_record["latency_sec"] = round(call_latency_sec, 3)
    if stage:
        call_record["stage"] = stage
    metrics["llm_calls"].append(call_record)


@contextmanager
def capture_chat_completion_metrics(chat_completions, stage: str | None = None, attempt_index: int = 1) -> dict:
    metrics = empty_metrics()
    metrics["_current_attempt_index"] = attempt_index
    started_at = time.perf_counter()
    original_create = chat_completions.create

    def wrapped_create(*args, **kwargs):
        call_started_at = time.perf_counter()
        response = original_create(*args, **kwargs)
        record_response_usage(
            metrics=metrics,
            response=response,
            call_latency_sec=time.perf_counter() - call_started_at,
            stage=stage,
        )
        return response

    try:
        with patch.object(chat_completions, "create", new=wrapped_create):
            yield metrics
    finally:
        metrics["latency_sec"] = round(time.perf_counter() - started_at, 3)


@contextmanager
def call_attempt(metrics: dict, attempt_index: int):
    previous_attempt = metrics.get("_current_attempt_index", 1)
    metrics["_current_attempt_index"] = attempt_index
    try:
        yield
    finally:
        metrics["_current_attempt_index"] = previous_attempt


def metric_subset(metrics: dict | None) -> dict:
    source = metrics or empty_metrics()
    return {field: source.get(field, 0.0 if field == "latency_sec" else 0) for field in METRIC_FIELDS}


def calls_subset(metrics: dict | None) -> list:
    source = metrics or empty_metrics()
    return [dict(call) for call in source.get("llm_calls", [])]


def combine_metrics(*metric_dicts: dict | None) -> dict:
    combined = empty_metrics()
    for metrics in metric_dicts:
        if not metrics:
            continue
        combined["latency_sec"] += float(metrics.get("latency_sec", 0.0) or 0.0)
        combined["num_llm_calls"] += int(metrics.get("num_llm_calls", 0) or 0)
        combined["input_tokens"] += int(metrics.get("input_tokens", 0) or 0)
        combined["output_tokens"] += int(metrics.get("output_tokens", 0) or 0)
        combined["total_tokens"] += int(metrics.get("total_tokens", 0) or 0)
        for call in metrics.get("llm_calls", []):
            combined["llm_calls"].append({
                **dict(call),
                "call_index": len(combined["llm_calls"]) + 1,
            })
    combined["latency_sec"] = round(combined["latency_sec"], 3)
    return combined
