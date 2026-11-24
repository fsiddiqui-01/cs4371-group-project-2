"""
TEMPORARY LLM helper for Person 4 until Person 3's shared llm.py is ready.

chat()       one HTTPS request to Groq's chat endpoint (system + user messages,
             temperature 0), with replies cached on disk.
numbers_ok() the number check: every number in the LLM's text must appear in the data.

COUNTER keeps how many LLM calls were made and how many came from the cache, so
Task 7 can report API calls per request.
When Person 3's llm.py exists, switch the import in task5/backend.py.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "llm"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

COUNTER = {"llm_calls": 0, "llm_cache_hits": 0}


def load_env(path=ROOT / ".env"):
    """Read KEY=value lines from .env into os.environ (keys never leave the server)."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def chat(messages, json_mode=False, use_cache=True):
    """Send a list of {"role", "content"} messages to Groq and return the reply text."""
    body = {"model": MODEL, "temperature": 0, "max_completion_tokens": 3000, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
    f = CACHE / f"p4_{key}.json"
    if use_cache and f.exists():
        COUNTER["llm_cache_hits"] += 1
        return json.loads(f.read_text())["text"]

    load_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
    for _ in range(5):
        COUNTER["llm_calls"] += 1
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"},
                          json=body, timeout=120)
        if r.status_code != 429:                      # 429 = rate limited: wait and retry
            break
        time.sleep(float(r.headers.get("retry-after", 10)) + 1)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"retrieved": time.time(), "model": MODEL, "text": text}))
    time.sleep(1)                                     # at least 1 s between calls
    return text


def parse_json(text):
    """Pull the first {...} object out of a reply and parse it (None if impossible)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------ number check
NUMBER_WORDS = re.compile(r"\b(hundred|thousand|million|dozen|half|twice|double|triple)\b", re.I)


def extract_numbers(text):
    """All numbers in a text as floats rounded to 2 decimals.
    Handles $, %, commas in thousands (1,234) and dates like 2027-03-12 (-> 2027, 3, 12)."""
    text = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d)", lambda m: m.group(0).replace(",", ""), text)
    return {round(float(x), 2) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def numbers_ok(text, data_text):
    """True if every number in `text` also appears in `data_text`.
    Returns (ok, bad) where bad lists what failed, for retry feedback and Task 7."""
    bad = sorted(extract_numbers(text) - extract_numbers(data_text))
    words = sorted({w.lower() for w in NUMBER_WORDS.findall(text)})
    return (not bad and not words), [*bad, *words]
