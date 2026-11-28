"""
llm.py - shared LLM helpers for SafeStay.

Everything any task needs to talk to an LLM:

    chat(system, user, ...)          -> reply text (one HTTPS request, temperature 0, cached on disk)
    parse_json(text)                 -> Python object pulled out of a reply
    validate(obj, schema)            -> list of problems ([] means valid)
    unsupported_numbers(text, data)  -> numbers in `text` that do not appear in `data`
    numbers_ok(text, data)           -> True if every number in `text` appears in `data`
    grounded_summary(system, user, data) -> summary text that passed the number check, or None

Providers (all use the same OpenAI-style chat format, so one function handles them):
    "groq"       main LLM, openai/gpt-oss-120b          (key: GROQ_API_KEY)
    "gemini"     second model for Task 3                 (key: GEMINI_API_KEY)
    "ollama"     local alternative second model          (no key)
    "openrouter" backup only if Groq is down             (key: OPENROUTER_API_KEY)

Keys are only ever read from environment variables (put them in a .env file, never in code).
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Configuration

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path=ROOT / ".env"):
    """Tiny .env reader so teammates don't need an extra package.
    Lines look like KEY=value. Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv()

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        "key_env": "GROQ_API_KEY",
    },
    # Google's own Gemini endpoint, plain HTTPS.
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model": os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        "key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "url": os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions"),
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
        "key_env": None,
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
        "key_env": "OPENROUTER_API_KEY",
    },
}

CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ROOT / "cache" / "llm"))
MIN_SECONDS_BETWEEN_CALLS = 1.0   # assignment rule: wait >= 1 s between calls to the same service
TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT", 300))

# Counters other modules can read
STATS = {"api_calls": 0, "cache_hits": 0}
_last_call_time = {}  # provider -> time.time() of last real request


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or returns an error."""


# chat()

def _cache_path(provider, payload):
    key = hashlib.sha256(json.dumps([provider, payload], sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / provider / f"{key}.json"


def _wait_for_rate_limit(provider):
    last = _last_call_time.get(provider)
    if last is not None:
        gap = time.time() - last
        if gap < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - gap)
    _last_call_time[provider] = time.time()


def chat(system, user, provider="groq", json_mode=False, use_cache=True, max_retries=3, max_tokens=None):
    """Send one system message + one user message and return the reply text.

    - temperature is always 0 so the same request gives the same answer
    - the response is saved to cache/llm/<provider>/<hash>.json with the time it was received,
      and reused next time instead of calling the API again
    - json_mode=True asks the provider for a JSON object (we still validate it ourselves)
    - max_tokens raises the reply length limit (needed for long JSON answers, e.g. Task 3)
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    gemini = provider == "gemini"
    if gemini:
        # Gemini's own request format: system text goes in systemInstruction
        payload = {
            "model": cfg["model"],
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0},
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
    else:
        payload = {
            "model": cfg["model"],
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_completion_tokens" if provider == "groq" else "max_tokens"] = max_tokens

    path = _cache_path(provider, payload)
    if use_cache and path.exists():
        STATS["cache_hits"] += 1
        return json.loads(path.read_text())["reply"]

    headers = {"Content-Type": "application/json"}
    if cfg["key_env"]:
        key = os.environ.get(cfg["key_env"])
        if not key:
            raise LLMError(f"Environment variable {cfg['key_env']} is not set (add it to your .env file).")
        if gemini:
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
    url = cfg["url"].format(model=cfg["model"])

    last_error = None
    for attempt in range(max_retries):
        _wait_for_rate_limit(provider)
        body = {k: v for k, v in payload.items() if not (gemini and k == "model")}
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            last_error = f"network error: {e}"
            time.sleep(2 ** attempt)
            continue
        STATS["api_calls"] += 1

        if resp.status_code == 429 or resp.status_code >= 500:
            # rate limited or server trouble: back off and try again
            wait = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(min(wait, 30))
            continue
        if resp.status_code == 400 and json_mode and not gemini and "response_format" in payload:
            payload.pop("response_format")
            last_error = f"HTTP 400 with JSON mode: {resp.text[:200]}"
            continue
        if resp.status_code != 200:
            raise LLMError(f"{provider} returned HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            if gemini:
                parts = data["candidates"][0]["content"]["parts"]
                reply = "".join(p.get("text", "") for p in parts)
            else:
                reply = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape from {provider}: {str(data)[:300]}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "received_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "request": payload,
            "reply": reply,
            "raw_response": data,
        }, indent=2))
        return reply

    raise LLMError(f"{provider} failed after {max_retries} attempts ({last_error})")


# JSON parsing and validation

def parse_json(text):
    # Pull a JSON object or array out of an LLM reply.
    
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first {...} or [...] block in the text
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Reply does not contain valid JSON")


def validate(obj, schema, where="root"):
    """Check a parsed object against a small schema. Returns a list of problems ([] = valid).

    Schema format (a dict, one entry per field):
        {
          "name":   {"type": str, "required": True},
          "rating": {"type": int, "required": True, "min": 1, "max": 5},
          "month":  {"type": int, "required": False, "min": 1, "max": 12, "nullable": True},
          "items":  {"type": list, "required": True, "items": {...schema for each element...}},
        }
    """
    problems = []
    if not isinstance(obj, dict):
        return [f"{where}: expected an object, got {type(obj).__name__}"]
    for field, rule in schema.items():
        if field not in obj:
            if rule.get("required", True):
                problems.append(f"{where}.{field}: missing")
            continue
        value = obj[field]
        if value is None:
            if not rule.get("nullable", False):
                problems.append(f"{where}.{field}: is null")
            continue
        expected = rule["type"]
        # accept ints where floats are expected; never accept bool as a number
        ok_type = isinstance(value, expected) and not (isinstance(value, bool) and expected in (int, float))
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            ok_type = True
        if expected is int and isinstance(value, float) and value.is_integer():
            ok_type = True
        if not ok_type:
            problems.append(f"{where}.{field}: expected {expected.__name__}, got {type(value).__name__}")
            continue
        if "min" in rule and value < rule["min"]:
            problems.append(f"{where}.{field}: {value} is below {rule['min']}")
        if "max" in rule and value > rule["max"]:
            problems.append(f"{where}.{field}: {value} is above {rule['max']}")
        if "choices" in rule and value not in rule["choices"]:
            problems.append(f"{where}.{field}: {value!r} not one of {rule['choices']}")
        if expected is list and "items" in rule:
            for i, item in enumerate(value):
                problems += validate(item, rule["items"], f"{where}.{field}[{i}]")
    return problems


# Number check: every number in the LLM's text must come from the data we sent
# a number: optional minus, digits with optional thousands commas, optional decimals
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def extract_numbers(text):
    # Return the numbers written in `text` as (string_as_written, float_value) pairs.
    
    out = []
    for m in _NUMBER_RE.finditer(text):
        s = m.group(0)
        out.append((s, float(s.replace(",", ""))))
    return out


def _data_values(data):
    # Collect every number that appears in the data we sent to the LLM.
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data = data.to_csv(index=False)
    except ImportError:
        pass
    if not isinstance(data, str):
        data = json.dumps(data, default=str)
    return [v for _, v in extract_numbers(data)]


def _decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def unsupported_numbers(text, data, allowed_extra=()):
    """Numbers in `text` that cannot be found in `data`.

    A number in the text counts as found if some data value, rounded to the same number of
    decimals as written in the text, equals it.
    A percentage in the text may also match a data share written as a fraction.
    `allowed_extra` lets the caller allow a few numbers that are in the question, not the data."""
    values = _data_values(data)
    extra = {float(x) for x in allowed_extra}
    bad = []
    for s, v in extract_numbers(text):
        if v in extra:
            continue
        d = _decimals(s)
        tol = 0.5 * 10 ** (-d) + 1e-9
        found = any(abs(dv - v) <= tol for dv in values)
        if not found:
            # percentage written in text, fraction in the data
            pos = text.find(s)
            if pos != -1 and text[pos + len(s): pos + len(s) + 1] == "%":
                found = any(abs(dv * 100 - v) <= tol for dv in values)
        if not found:
            bad.append(s)
    return bad


def numbers_ok(text, data, allowed_extra=()):
    """True if every number in `text` appears in `data` (see unsupported_numbers)."""
    return not unsupported_numbers(text, data, allowed_extra)


# Grounded summary: ask, check numbers, retry with feedback, give up -> None

def grounded_summary(system, user, data, provider="groq", allowed_extra=(), retries=2):
    """Ask for a summary and only return it if it passes the number check.

    Returns (summary_or_None, log) where log is a list of attempts:
        [{"attempt": 1, "passed": False, "unsupported": ["12.7"]}, ...]
    If it never passes, the caller should show the table alone.

    Because temperature is 0, simply asking again would give the same answer, so each retry
    tells the model which numbers were not in the data."""
    log = []
    prompt = user
    for attempt in range(1, retries + 2):
        reply = chat(system, prompt, provider=provider)
        bad = unsupported_numbers(reply, data, allowed_extra)
        log.append({"attempt": attempt, "passed": not bad, "unsupported": bad})
        if not bad:
            return reply, log
        prompt = (user + "\n\nYour previous answer used these numbers that are NOT in the data: "
                  + ", ".join(bad) + ". Rewrite the summary using only numbers that appear in the data, "
                  "copied exactly as written there, or use no number at all.")
    return None, log
