"""
Task 7: run every hand-labelled test conversation through the assistant and score it.

Run from the repo root (needs GROQ_API_KEY):   python -m task7.evaluate
Writes:  t7_evaluation.csv   one row per test
         task7/t7_results.md a results table for the report

Measured for each test:
  - field match: extracted fields vs the hand labels (after the last turn)
  - behaviour:   did each turn give the expected kind of reply (answer / ask / decline)?
  - number check: did the answer pass the number check?
  - time and number of LLM calls (and cache hits, API calls if the backend counts them)
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from task5 import assistant as A
from task5 import backend as B

ROOT = Path(__file__).resolve().parent.parent
TESTS = json.loads((Path(__file__).parent / "test_requests.json").read_text())["tests"]


def resolve(label):
    """Turn label tokens like {next_weekend} into today's concrete date."""
    if isinstance(label, str) and label.startswith("{") and label.endswith("}"):
        return A.date_hints()[label[1:-1]]
    return label


def same(field, expected, actual):
    if field == "preferences":                     # every expected word must appear somewhere
        text = " ".join(actual or []).lower()
        return all(w.lower() in text for w in expected)
    if field == "near":
        return expected is None and actual is None or (
            expected is not None and actual is not None and expected.lower() in actual.lower())
    if field == "city":
        return (expected or "").lower() == (actual or "").lower()
    if isinstance(expected, (int, float)) and actual is not None:
        return float(expected) == float(actual)
    return expected == actual


def run_test(t):
    state = A.new_state()
    api = getattr(B, "API_COUNTER", None)
    api0 = api["api_calls"] if api else 0
    hits0 = B.COUNTER["llm_cache_hits"]
    kinds, number_checks, seconds, llm_calls, last = [], [], 0.0, 0, None
    for msg in t["turns"]:
        r = A.handle(msg, state)
        kinds.append(r["kind"])
        seconds += r["seconds"]
        llm_calls += r["llm_calls"]
        if r["kind"] == "answer":
            number_checks.append(bool(r["number_check"]))
        last = r

    got = {"intent": state["intent"], **state["collected"]}
    checked = mismatched = 0
    misses = []
    for f, exp in t["expected_fields"].items():
        exp = resolve(exp)
        checked += 1
        if not same(f, exp, got.get(f)):
            mismatched += 1
            misses.append(f"{f}: expected {exp!r}, got {got.get(f)!r}")

    return {
        "id": t["id"], "source": t["source"], "request": " || ".join(t["turns"]),
        "expected_kinds": "/".join(t["expected_kinds"]), "actual_kinds": "/".join(kinds),
        "behaviour_correct": kinds == t["expected_kinds"],
        "follow_up_or_decline_expected": any(k in ("ask", "decline") for k in t["expected_kinds"]),
        "fields_checked": checked, "fields_matched": checked - mismatched,
        "all_fields_match": mismatched == 0, "field_mismatches": "; ".join(misses),
        "answers": len(number_checks),
        "number_check_passed": all(number_checks) if number_checks else None,
        "seconds": round(seconds, 2), "llm_calls": llm_calls,
        "llm_cache_hits": B.COUNTER["llm_cache_hits"] - hits0,
        "api_calls": (api["api_calls"] - api0) if api else None,
        "final_reply": (last["text"] if last else "")[:300],
    }


def main():
    rows = []
    for t in TESTS:
        print(f"test {t['id']:>2}: {t['turns'][0][:70]}", flush=True)
        try:
            rows.append(run_test(t))
        except Exception as e:                                   # record crashes as failures
            rows.append({"id": t["id"], "source": t["source"], "request": " || ".join(t["turns"]),
                         "expected_kinds": "/".join(t["expected_kinds"]), "actual_kinds": "crash",
                         "behaviour_correct": False, "final_reply": f"{type(e).__name__}: {e}"})
        print("        ->", rows[-1]["actual_kinds"], "| fields ok:", rows[-1].get("all_fields_match"))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "t7_evaluation.csv", index=False)

    fu = df[df.follow_up_or_decline_expected.fillna(False).astype(bool)]
    with_answers = df[df.answers.fillna(0) > 0]
    summary = pd.DataFrame([
        ("Tests", len(df)),
        ("Field match rate (fields matched / fields labelled)",
         f"{df.fields_matched.sum() / max(df.fields_checked.sum(), 1):.0%}"),
        ("Tests with every field correct", f"{df.all_fields_match.fillna(False).astype(bool).mean():.0%}"),
        ("Reply kind correct on every turn", f"{df.behaviour_correct.mean():.0%}"),
        ("Correct follow-up / decline (tests that needed one)", f"{fu.behaviour_correct.mean():.0%}" if len(fu) else "n/a"),
        ("Answers passing the number check",
         f"{with_answers.number_check_passed.fillna(False).astype(bool).mean():.0%}" if len(with_answers) else "n/a"),
        ("Mean time per test (s)", f"{df.seconds.mean():.1f}"),
        ("Mean LLM calls per test", f"{df.llm_calls.mean():.1f}"),
    ], columns=["Measure", "Result"])
    md = ("# Task 7 results\n\n" + summary.to_markdown(index=False) + "\n\n## Failures\n\n" +
          (df[~df.behaviour_correct | ~df.all_fields_match.fillna(False).astype(bool)]
           [["id", "request", "expected_kinds", "actual_kinds", "field_mismatches"]].to_markdown(index=False)))
    (Path(__file__).parent / "t7_results.md").write_text(md + "\n")
    print("\n" + summary.to_string(index=False))
    print("\nWrote t7_evaluation.csv and task7/t7_results.md")


if __name__ == "__main__":
    sys.exit(main())
