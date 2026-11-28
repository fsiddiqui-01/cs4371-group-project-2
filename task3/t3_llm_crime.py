"""
Task 3 - Crime levels from LLMs, checked against the real data.

Asks two LLMs to rate every Seattle neighbourhood from 1 (fewest reported incidents) to 5 (most)
and estimate incidents per month:
    run 1: Groq gpt-oss-120b, no data
    run 2: second model (Gemini Flash or Ollama), no data
    run 3: Groq, with the Task 1 monthly averages as CSV
    run 4: second model, with the Task 1 monthly averages as CSV
then compares each run with the real Task 1 figures.

Outputs (in task3/):
    t3_llm_answers.json   every prompt, raw reply, parsed answer and validation result
    t3_comparison.csv     one row per run x neighbourhood
    t3_metrics.txt        Spearman, top-5 overlap, median % error, skipped / invented, per run
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import llm

OUT = HERE
MONTHS_IN_DATA = 36

# Column names in Person 1's neighbourhood_month.parquet.
NEIGHBOURHOOD_COLS = ["neighbourhood", "neighborhood", "mcpp"]
TOTAL_COLS = ["total", "total_incidents", "incidents", "n_incidents"]

SOCRATA_URL = "https://data.seattle.gov/resource/tazs-3rd5.json"

# Real figures

def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def real_monthly_averages(parquet_path):
    """Average reported incidents per month for each neighbourhood, 2022-2024.
    = total incidents over the 3 years / 36 months (months with no incidents count as 0)."""
    df = pd.read_parquet(parquet_path)
    ncol, tcol = _pick(df, NEIGHBOURHOOD_COLS), _pick(df, TOTAL_COLS)
    real = (df.groupby(ncol)[tcol].sum() / MONTHS_IN_DATA).rename("real_monthly_avg").reset_index()
    real = real.rename(columns={ncol: "neighbourhood"})
    real["real_monthly_avg"] = real["real_monthly_avg"].round(1)
    return real.sort_values("real_monthly_avg", ascending=False).reset_index(drop=True)


def names_from_api():
    """Distinct neighbourhood names straight from the Seattle API (lets Task 3 start before Task 1).
    Cached in cache/socrata/ with the time received."""
    cache = ROOT / "cache" / "socrata" / "t3_neighbourhood_names.json"
    if cache.exists():
        return json.loads(cache.read_text())["names"]
    field = os.environ.get("SEATTLE_NEIGHBOURHOOD_FIELD", "neighborhood")
    params = {"$select": f"distinct {field}", "$limit": 1000}
    headers = {}
    if os.environ.get("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    resp = requests.get(SOCRATA_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    names = sorted({r[field].strip() for r in resp.json()
                    if r.get(field) and r[field].strip() not in ("-", "UNKNOWN", "OOJ")})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"received_at": datetime.now(timezone.utc).isoformat(),
                                 "url": SOCRATA_URL, "params": params, "names": names}, indent=2))
    time.sleep(1)
    return names


# Prompts

SYSTEM = (
    "You estimate reported police incidents for neighbourhoods of Seattle, Washington. "
    "Rules: (1) Answer ONLY with one JSON object, no other text. "
    "(2) Include every neighbourhood from the list exactly once, spelled exactly as given, and no others. "
    "(3) For each give 'rating', an integer from 1 (fewest reported incidents in the city) to 5 (most), "
    "and 'incidents_per_month', your estimate of the average number of reported incidents per month "
    "in 2022-2024 (a number >= 0). "
    'Format: {"neighbourhoods": [{"name": "...", "rating": 3, "incidents_per_month": 120}, ...]}'
)


def prompt_without_data(names):
    return ("Here are the Seattle neighbourhoods (SPD Micro Community Policing Plan areas):\n"
            + "\n".join(names)
            + "\n\nUsing only what you already know, rate each one and estimate its incidents per month.")


def prompt_with_data(real):
    csv = real[["neighbourhood", "real_monthly_avg"]].rename(
        columns={"real_monthly_avg": "avg_reported_incidents_per_month_2022_2024"}).to_csv(index=False)
    return ("Here are the Seattle neighbourhoods with their average reported incidents per month "
            "in 2022-2024, from Seattle Police Department open data:\n\n" + csv +
            "\nUsing this data, rate each neighbourhood and give its incidents per month.")


ANSWER_SCHEMA = {
    "neighbourhoods": {"type": list, "required": True, "items": {
        "name": {"type": str, "required": True},
        "rating": {"type": int, "required": True, "min": 1, "max": 5},
        "incidents_per_month": {"type": float, "required": True, "min": 0},
    }}
}


# Asking and checking

def norm(name):
    """Normalise a neighbourhood name for matching: upper case, letters/digits only."""
    return re.sub(r"[^A-Z0-9]+", " ", str(name).upper()).strip()


def ask(provider, user_prompt, run_id, names):
    """One run: call the model, parse and validate the JSON, and check every neighbourhood is there.
    Up to two retries, each telling the model exactly what was wrong."""
    record = {"run": run_id, "provider": provider, "model": llm.PROVIDERS[provider]["model"],
              "asked_at": datetime.now(timezone.utc).isoformat(),
              "system": SYSTEM, "user": user_prompt, "attempts": []}
    prompt = user_prompt
    for attempt in (1, 2, 3):
        try:
            reply = llm.chat(SYSTEM, prompt, provider=provider, json_mode=True, max_tokens=16000)
        except llm.LLMError as e:
            print(f"  ERROR from {provider}: {e}")
            record["attempts"].append({"attempt": attempt, "error": str(e)})
            break
        try:
            parsed = llm.parse_json(reply)
            problems = llm.validate(parsed, ANSWER_SCHEMA)
        except ValueError as e:
            parsed, problems = None, [str(e)]
        if not problems:
            # every neighbourhood must be answered (the model sometimes stops part-way)
            got = {norm(i.get("name")) for i in parsed["neighbourhoods"] if isinstance(i, dict)}
            missing = [n for n in names if norm(n) not in got]
            if missing:
                problems = [f"{len(missing)} neighbourhoods are missing: " + ", ".join(missing)]
        record["attempts"].append({"attempt": attempt, "reply": reply, "parsed": parsed,
                                   "problems": problems[:20], "valid": not problems})
        if not problems:
            break
        print(f"  attempt {attempt} not accepted: {problems[0][:150]}")
        prompt = (user_prompt + "\n\nYour previous answer was not valid: " + "; ".join(problems[:10])
                  + ". Answer again with only the JSON object in the required format, "
                  "including ALL " + str(len(names)) + " neighbourhoods.")
    final = record["attempts"][-1] if record["attempts"] else {}
    record["valid"] = bool(final.get("valid"))
    # keep the valid entries even if a few rows were bad, so we can still measure something
    record["answer"] = final.get("parsed") if isinstance(final.get("parsed"), dict) else None
    return record


def llm_table(answer):
    rows = []
    for item in (answer or {}).get("neighbourhoods", []):
        if not isinstance(item, dict):
            continue
        rating, est = item.get("rating"), item.get("incidents_per_month")
        rows.append({"llm_name": item.get("name"),
                     "llm_rating": rating if isinstance(rating, (int, float)) else None,
                     "llm_estimate": est if isinstance(est, (int, float)) else None})
    return pd.DataFrame(rows, columns=["llm_name", "llm_rating", "llm_estimate"])


# Metrics (written by hand, no stats library)

def ranks(values):
    """Ranks starting at 1, ties get the average of their positions (needed for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rank correlation = Pearson correlation of the ranks (handles ties)."""
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def compare(real, record):
    """Join one run's answer to the real figures and compute the metrics."""
    got = llm_table(record["answer"])
    got["key"] = got["llm_name"].map(norm)
    got = got.drop_duplicates("key")
    r = real.copy()
    r["key"] = r["neighbourhood"].map(norm)

    merged = r.merge(got, on="key", how="outer", indicator=True)
    merged["status"] = merged["_merge"].map({"both": "matched", "left_only": "skipped",
                                             "right_only": "invented"})
    merged["pct_error"] = ((merged["llm_estimate"] - merged["real_monthly_avg"]).abs()
                           / merged["real_monthly_avg"] * 100).where(merged["real_monthly_avg"] > 0).round(1)
    merged.insert(0, "run", record["run"])
    merged.insert(1, "model", record["model"])

    m = merged[(merged.status == "matched") & merged.llm_rating.notna()]
    rho = spearman(m["llm_rating"], m["real_monthly_avg"])
    rho_est = spearman(m["llm_estimate"].fillna(0), m["real_monthly_avg"])

    real_top5 = set(r.nlargest(5, "real_monthly_avg")["key"])
    # LLM top 5: highest rating first, ties broken by the LLM's own incident estimate
    llm_top5 = set(m.sort_values(["llm_rating", "llm_estimate"], ascending=False).head(5)["key"])

    errs = m["pct_error"].dropna()
    metrics = {
        "run": record["run"], "model": record["model"], "valid_json": record["valid"],
        "neighbourhoods_real": len(r), "neighbourhoods_answered": int(len(got)),
        "matched": int(len(m)),
        "spearman_rating_vs_real": round(rho, 3),
        "spearman_estimate_vs_real": round(rho_est, 3),
        "top5_overlap": f"{len(real_top5 & llm_top5)}/5",
        "median_pct_error": round(statistics.median(errs), 1) if len(errs) else None,
        "skipped": sorted(merged.loc[merged.status == "skipped", "neighbourhood"].tolist()),
        "invented": sorted(merged.loc[merged.status == "invented", "llm_name"].astype(str).tolist()),
    }
    cols = ["run", "model", "status", "neighbourhood", "llm_name", "real_monthly_avg",
            "llm_rating", "llm_estimate", "pct_error"]
    return merged[cols], metrics


# Main

def write_metrics(all_metrics, path):
    lines = ["Task 3 - LLM crime levels vs Seattle Police Department data (2022-2024)",
             f"Generated: {datetime.now(timezone.utc).isoformat()}",
             "Real figure = average reported incidents per month per neighbourhood (Task 1 table / 36 months).",
             "LLM top 5 = highest rating, ties broken by the LLM's incident estimate.",
             "pct_error = |LLM estimate - real| / real x 100, median over matched neighbourhoods.", ""]
    header = f"{'run':<28}{'valid':<7}{'matched':<9}{'spearman(rating)':<18}{'spearman(est)':<15}{'top5':<6}{'median % err':<13}"
    lines += [header, "-" * len(header)]
    for m in all_metrics:
        lines.append(f"{m['run']:<28}{str(m['valid_json']):<7}{m['matched']:>3}/{m['neighbourhoods_real']:<5}"
                     f"{m['spearman_rating_vs_real']:<18}{m['spearman_estimate_vs_real']:<15}"
                     f"{m['top5_overlap']:<6}{str(m['median_pct_error']):<13}")
    lines.append("")
    for m in all_metrics:
        lines.append(f"[{m['run']}] model={m['model']}")
        lines.append(f"  skipped ({len(m['skipped'])}): {', '.join(m['skipped']) or 'none'}")
        lines.append(f"  invented ({len(m['invented'])}): {', '.join(m['invented']) or 'none'}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(ROOT / "neighbourhood_month.parquet"))
    ap.add_argument("--second", default="gemini", choices=["gemini", "ollama", "openrouter"],
                    help="second model from a different provider")
    ap.add_argument("--main", default="groq", choices=["groq", "openrouter"],
                    help="main model (use openrouter only if Groq is down, and say so in the report)")
    ap.add_argument("--names-only", action="store_true",
                    help="Task 1 not ready yet: fetch names from the API and run only the no-data questions")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.names_only:
        names, real = names_from_api(), None
    else:
        real = real_monthly_averages(args.parquet)
        names = sorted(real["neighbourhood"])
    print(f"{len(names)} neighbourhoods")

    runs = [(f"{args.main}_without_data", args.main, prompt_without_data(names)),
            (f"{args.second}_without_data", args.second, prompt_without_data(names))]
    if real is not None:
        runs += [(f"{args.main}_with_data", args.main, prompt_with_data(real)),
                 (f"{args.second}_with_data", args.second, prompt_with_data(real))]

    records = []
    for run_id, provider, prompt in runs:
        print(f"asking {run_id} ...")
        rec = ask(provider, prompt, run_id, names)
        print(f"  valid JSON with all neighbourhoods: {rec['valid']}")
        records.append(rec)

    (OUT / "t3_llm_answers.json").write_text(json.dumps(records, indent=2, default=str))
    print(f"wrote {OUT / 't3_llm_answers.json'}")

    if real is None:
        print("Task 1 table not given: skipping the comparison. Run again without --names-only later.")
        return

    tables, metrics = [], []
    for rec in records:
        t, m = compare(real, rec)
        tables.append(t)
        metrics.append(m)
    pd.concat(tables).to_csv(OUT / "t3_comparison.csv", index=False)
    write_metrics(metrics, OUT / "t3_metrics.txt")
    print(f"wrote {OUT / 't3_comparison.csv'} and {OUT / 't3_metrics.txt'}")
    print((OUT / "t3_metrics.txt").read_text())
    print("LLM calls:", llm.STATS)


if __name__ == "__main__":
    main()
