"""
Task 6 - Best month to visit Seattle.

For each month:
    avg_temp_f         average daily temperature in Seattle, 2020-2024 (city_temperature.csv)
    temp_distance_f    |avg_temp_f - 70|, distance from a comfortable 70 F
    avg_incidents      the city's average monthly incident total, 2022-2024 (Task 1 table)
    avg_incidents_per_day  the same divided by the days in each month (used for scoring, so that
                       short months like February don't look better just because they are short)

Scoring rule (lower score = better month):
    temp_score     = temp_distance_f scaled to 0-1 across the 12 months (0 = closest to 70 F)
    incident_score = avg_incidents_per_day scaled to 0-1 across the 12 months (0 = fewest)
    score          = 0.7 * temp_score + 0.3 * incident_score
The months are ranked by score and the top 3 are recommended.
"""

import argparse
import calendar
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CITY = "Seattle"
TEMP_YEARS = (2020, 2024)
COMFORT_F = 70.0
MISSING = -99
W_TEMP, W_INCIDENTS = 0.7, 0.3

YEAR_COLS = ["year"]
MONTH_COLS = ["month"]
TOTAL_COLS = ["total", "total_incidents", "incidents", "n_incidents"]


def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def monthly_temperature(temps_csv):
    """Average daily temperature per calendar month for Seattle, 2020-2024, missing days removed."""
    t = pd.read_csv(temps_csv)
    t = t[(t["City"] == CITY) & t["Year"].between(*TEMP_YEARS) & (t["AvgTemperature"] != MISSING)]
    if t.empty:
        raise ValueError(f"No {CITY} rows for {TEMP_YEARS} in {temps_csv}")
    out = t.groupby("Month").agg(avg_temp_f=("AvgTemperature", "mean"),
                                 days_used=("AvgTemperature", "size")).reset_index()
    out = out.rename(columns={"Month": "month"})
    out["avg_temp_f"] = out["avg_temp_f"].round(1)
    out["temp_distance_f"] = (out["avg_temp_f"] - COMFORT_F).abs().round(1)
    return out


def monthly_incidents(parquet_path):
    """City-wide incident total for each (year, month), then averaged over 2022-2024 per month."""
    df = pd.read_parquet(parquet_path)
    y, m, tot = _pick(df, YEAR_COLS), _pick(df, MONTH_COLS), _pick(df, TOTAL_COLS)
    city = df.groupby([y, m])[tot].sum().reset_index()
    # incidents per day too, so February (28-29 days) isn't favoured just for being short
    city["days"] = [calendar.monthrange(int(yy), int(mm))[1] for yy, mm in zip(city[y], city[m])]
    city["per_day"] = city[tot] / city["days"]
    out = city.groupby(m).agg(avg_incidents=(tot, "mean"), avg_incidents_per_day=("per_day", "mean")).reset_index()
    out["avg_incidents"] = out["avg_incidents"].round(0)
    out["avg_incidents_per_day"] = out["avg_incidents_per_day"].round(1)
    return out.rename(columns={m: "month"})


def _scale(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0


def score_months(table):
    """Combine temperature and incidents into one score (lower is better) and rank.

    Weights 0.7 temperature / 0.3 incidents, because:
      - Weather changes a visit a lot in Seattle: monthly averages run from the low 40s F in winter
        to the mid 60s F in summer, which decides what a traveller can do outdoors.
      - The city-wide incident total moves much less from month to month, and a city total says
        little about any one visitor's experience (see the Responsible Use section), so it should
        only break near-ties between months with similar weather.
      - Min-max scaling stretches even small differences to the full 0-1 range, so a lower incident
        weight also stops a small monthly change in incidents from outweighing a big change in
        temperature.
    """
    t = table.copy()
    t["temp_score"] = _scale(t["temp_distance_f"]).round(3)
    if "avg_incidents" in t:
        t["incident_score"] = _scale(t["avg_incidents_per_day"]).round(3)
        t["score"] = (W_TEMP * t["temp_score"] + W_INCIDENTS * t["incident_score"]).round(3)
    else:
        t["score"] = t["temp_score"]
    t = t.sort_values(["score", "temp_distance_f"]).reset_index(drop=True)
    t.insert(0, "rank", range(1, len(t) + 1))
    t.insert(2, "month_name", t["month"].map(lambda i: calendar.month_name[int(i)]))
    t["recommended"] = t["rank"] <= 3
    return t


def best_months(temps_csv=HERE / "city_temperature.csv",
                parquet_path=ROOT / "neighbourhood_month.parquet"):
    """Return the ranked 12-month table. Used by t6 itself and by the Streamlit app."""
    table = monthly_temperature(temps_csv)
    if parquet_path and Path(parquet_path).exists():
        table = table.merge(monthly_incidents(parquet_path), on="month", how="left")
    return score_months(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", default=str(HERE / "city_temperature.csv"))
    ap.add_argument("--parquet", default=str(ROOT / "neighbourhood_month.parquet"))
    ap.add_argument("--out", default=str(HERE / "t6_best_months.csv"))
    args = ap.parse_args()

    table = best_months(args.temps, args.parquet)
    if "avg_incidents" not in table:
        print(f"NOTE: {args.parquet} not found - ranked on temperature only. "
              "Run again once Person 1's Task 1 table is ready.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    top = ", ".join(table.loc[table.recommended, "month_name"])
    print(f"\nRecommended months: {top}\nwrote {args.out}")


if __name__ == "__main__":
    main()
