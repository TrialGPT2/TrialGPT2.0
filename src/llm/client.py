# src/llm/client.py
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, AzureOpenAI


Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


@dataclass
class LLMConfig:
    provider: str = "azure"  # "azure" or "openai"
    model: str = "gpt-4.1"  # OpenAI model name OR Azure deployment name
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout_s: float = 60.0
    max_retries: int = 3
    retry_backoff_s: float = 1.5

    # Azure-specific
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: str = "2024-12-01-preview"  # adjust if needed

    # OpenAI-specific
    openai_api_key: Optional[str] = None


class LLMClient:
    """
    Minimal chat client supporting Azure OpenAI or OpenAI.

    - provider="azure": uses AzureOpenAI(endpoint, api_key, api_version)
      and `model` is your *deployment name*.
    - provider="openai": uses OpenAI(api_key) and `model` is model id.

    JSON mode:
      pass response_format={"type": "json_object"} to ask for JSON output.
      or response_format={"type": "json_schema", "json_schema": {...}}
      We'll attempt to parse the response and return dict when possible.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = self._init_client(cfg)

    @staticmethod
    def from_env(model_default: str = "gpt-4o-mini") -> "LLMClient":
        provider = os.getenv("LLM_PROVIDER", "azure").strip().lower()
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or model_default
        )

        cfg = LLMConfig(
            provider=provider,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_backoff_s=float(os.getenv("LLM_RETRY_BACKOFF_S", "1.5")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "0")) or None,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_api_version=(
                os.getenv("AZURE_OPENAI_API_VERSION")
                or os.getenv("AZURE_API_VERSION")
                or "2024-02-01"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        return LLMClient(cfg)

    def _init_client(self, cfg: LLMConfig):
        if cfg.provider == "azure":
            if not cfg.azure_endpoint or not cfg.azure_api_key:
                raise ValueError(
                    "Azure config missing. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
                )
            return AzureOpenAI(
                azure_endpoint=cfg.azure_endpoint,
                api_key=cfg.azure_api_key,
                api_version=cfg.azure_api_version,
            )

        if cfg.provider == "openai":
            if not cfg.openai_api_key:
                raise ValueError("OpenAI config missing. Set OPENAI_API_KEY.")
            return OpenAI(api_key=cfg.openai_api_key)

        raise ValueError(f"Unknown provider: {cfg.provider!r} (use 'azure' or 'openai')")

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Returns:
          - str if not JSON mode / parsing fails
          - dict if JSON mode and parsing succeeds
        """
        cfg = self.cfg
        temp = cfg.temperature if temperature is None else temperature
        mtok = cfg.max_tokens if max_tokens is None else max_tokens

        last_err: Optional[Exception] = None
        for attempt in range(cfg.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": cfg.model,
                    "messages": messages,
                    "temperature": temp,
                }
                if mtok is not None:
                    kwargs["max_tokens"] = mtok
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if seed is not None:
                    kwargs["seed"] = seed

                # NOTE: openai SDK handles timeout differently depending on version.
                # We'll pass `timeout` via client options when possible by using a simple
                # per-request timeout pattern: the SDK supports `timeout` in .create in many versions.
                kwargs["timeout"] = cfg.timeout_s

                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""

                # If JSON mode requested, attempt to parse
                if response_format is not None and response_format.get("type") in {"json_object", "json_schema"}:
                    parsed = _safe_json_loads(content)
                    return parsed if parsed is not None else content

                return content

            except Exception as e:
                last_err = e
                if attempt >= cfg.max_retries:
                    break
                sleep_s = _retry_sleep_s(cfg, e, attempt)
                time.sleep(sleep_s)

        raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parsing. Returns dict if valid JSON object; else None.
    """
    s = s.strip()
    if not s:
        return None

    # Sometimes models wrap JSON in ```json ... ```
    if s.startswith("```"):
        s = s.strip("`")
        # remove optional leading 'json'
        s = s.replace("json", "", 1).strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _is_rate_limit_error(err: Exception) -> bool:
    status_code = getattr(err, "status_code", None)
    if status_code == 429:
        return True

    code = getattr(err, "code", None)
    if str(code).strip().lower() in {"429", "too_many_requests", "rate_limit_exceeded"}:
        return True

    text = f"{type(err).__name__}: {err}".strip().lower()
    signals = (
        " 429",
        "error code: 429",
        "too many requests",
        "too_many_requests",
        "rate limit",
        "ratelimit",
        "throttl",
    )
    return any(signal in text for signal in signals)


def _retry_sleep_s(cfg: LLMConfig, err: Exception, attempt: int) -> float:
    base = max(0.1, float(cfg.retry_backoff_s))
    if _is_rate_limit_error(err):
        # Rate limits usually need materially longer recovery than generic transient failures.
        raw = max(8.0, base * 4.0 * (2 ** attempt))
        return min(90.0, raw + random.uniform(0.0, 1.0))

    raw = base * (2 ** attempt)
    return min(30.0, raw + random.uniform(0.0, 0.5))


# -----------------------
# Env var examples:
# -----------------------
# Azure:
#   export LLM_PROVIDER=azure
#   export LLM_MODEL="gpt-4o"                # <-- your Azure deployment name
#   export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
#   export AZURE_OPENAI_API_KEY="..."
#   export AZURE_OPENAI_API_VERSION="2024-02-01"
#
# OpenAI:
#   export LLM_PROVIDER=openai
#   export LLM_MODEL="gpt-4o-mini"
#   export OPENAI_API_KEY="..."
