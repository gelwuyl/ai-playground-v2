"""OpenRouter API service — replaces the Gemini SDK with OpenAI-compatible HTTP.

Uses httpx to call OpenRouter's /chat/completions endpoint.
OpenRouter is OpenAI-compatible, so this is the same format the student
guide's Prompt 1 describes.
"""
import os
import json
import re
import time
import random
import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def _post_with_retry(payload: dict) -> httpx.Response:
    """POST to OpenRouter with exponential backoff on 429/5xx.

    Honors the Retry-After header when present; otherwise uses exponential
    backoff (BASE_DELAY * 2^attempt) plus a small random jitter to avoid
    thundering herd. Raises for non-retryable errors (401, 400, etc.).
    """
    for attempt in range(MAX_RETRIES):
        resp = httpx.post(
            OPENROUTER_BASE_URL,
            headers=_headers(),
            json=payload,
            timeout=60,
        )

        if resp.status_code == 200:
            return resp

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                sleep_time = float(retry_after)
            else:
                sleep_time = (BASE_DELAY * (2 ** attempt)) + random.uniform(0.1, 0.5)
            print(f"Rate limited (429). Waiting {sleep_time:.2f}s before retry {attempt + 1}/{MAX_RETRIES}.")
            time.sleep(sleep_time)
            continue

        if resp.status_code >= 500:
            sleep_time = BASE_DELAY * (2 ** attempt)
            print(f"Server error {resp.status_code}. Waiting {sleep_time:.2f}s before retry {attempt + 1}/{MAX_RETRIES}.")
            time.sleep(sleep_time)
            continue

        # Non-retryable error (401, 400, 404, etc.)
        resp.raise_for_status()

    raise RuntimeError("Failed after maximum retries due to rate limits.")


def call_openrouter(prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
    """Send a prompt to OpenRouter and return the text response.

    `model` overrides OPENROUTER_MODEL for this call (OpenRouter model id).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = _post_with_retry({"model": model or OPENROUTER_MODEL, "messages": messages})
    return resp.json()["choices"][0]["message"]["content"]


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if the model wrapped its JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def call_openrouter_json(prompt: str, system_prompt: str | None, schema: dict, model: str | None = None) -> dict:
    """Call OpenRouter and request a JSON object matching `schema`.

    Uses response_format json_object where supported, plus fence-stripping
    and retry up to 3x on parse failure.
    Raises ValueError if a valid JSON object cannot be obtained.
    `model` overrides OPENROUTER_MODEL for this call (OpenRouter model id).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for _ in range(3):
        try:
            resp = _post_with_retry(body)
            text = resp.json()["choices"][0]["message"]["content"]
            text = _strip_fences(text)
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"OpenRouter did not return valid JSON: {last_err}")
