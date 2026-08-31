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
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
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
    """Extract the JSON payload from a model response.

    Handles bare JSON, a leading fenced block (```json ... ```), a fenced
    block preceded by prose, and prose around a bare JSON object/array.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

    # Fenced block somewhere after prose
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Bare JSON possibly wrapped in prose: outermost {...} or [...], chosen
    # by whichever opener appears first in the text.
    spans = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            spans.append((start, end + 1))
    if spans:
        start, stop = min(spans)
        return text[start:stop]
    return text


def call_openrouter_json(prompt: str, system_prompt: str | None, schema: dict, model: str | None = None) -> dict:
    """Call OpenRouter and request a JSON object matching `schema`.

    Tries strict JSON mode (response_format json_object) first; some models
    reject `response_format` or return empty content under it (e.g. thinking
    models via OpenRouter), so the last attempts drop it and rely on the
    prompt's explicit JSON instruction plus fence-stripping.
    Raises ValueError if a valid JSON object cannot be obtained.
    `model` overrides OPENROUTER_MODEL for this call (OpenRouter model id).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_err = None
    for use_json_mode in (True, True, False, False):
        body = {"model": model or OPENROUTER_MODEL, "messages": messages}
        if use_json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            resp = _post_with_retry(body)
            text = resp.json()["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError(f"model returned non-string content: {type(text).__name__}")
            text = _strip_fences(text)
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"OpenRouter did not return valid JSON: {last_err}")
