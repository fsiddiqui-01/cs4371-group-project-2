"""
Adapter: Person 2's real Task 4 code  ->  the format the app uses (task5/INTERFACES.md).

Person 2's functions work, but return different shapes and column names:
  search_hotels(...)            -> dict {"cheapest_overall", "cheapest_alternative", "hotels"}
  find_nearby_hotels(key, km)   -> (hotel_name, DataFrame)
  columns latitude / longitude / neighborhood / cheapest_nightly_rate / base_stay_total /
  prices_by_site (a JSON list of {"site", "nightly_rate"}).

This file only renames and reshapes. All searching, caching, fallback, incident filtering and
haversine distances are Person 2's code. Run the app from the repo root: their code uses paths
like outputs/t4_hotels.csv and cache/person2/.
"""
import contextlib
import io
import json
import re
from datetime import datetime, timezone

import pandas as pd

from person2.dynamic_hotel_search import get_top5_neighborhoods
from person2.dynamic_hotel_search import search_hotels as _p2_search
from person2.nearby_hotels import find_nearby_hotels

RENAME = {"latitude": "lat", "longitude": "lon", "neighborhood": "neighbourhood",
          "cheapest_nightly_rate": "cheapest_price", "base_stay_total": "total"}


def _site_col(site):
    return "price_" + re.sub(r"[^a-z0-9]+", "_", str(site).lower()).strip("_")


def _reshape(df, budget, nights=None):
    """Rename columns and spread prices_by_site into one price_<site> column per site."""
    out = df.rename(columns=RENAME).copy()
    site_prices = []
    for raw in out.get("prices_by_site", pd.Series([None] * len(out))):
        try:
            rates = json.loads(raw) if isinstance(raw, str) else []
        except ValueError:
            rates = []
        site_prices.append({_site_col(r.get("site")): r.get("nightly_rate") for r in rates})
    prices = pd.DataFrame(site_prices, index=out.index)
    # keep the most common booking sites first so the table reads well
    prices = prices[prices.notna().sum().sort_values(ascending=False).index]
    out = pd.concat([out, prices], axis=1)
    if "total" not in out and nights:
        out["total"] = out["cheapest_price"] * nights
    out["within_budget"] = out["cheapest_price"].notna() & (out["cheapest_price"] <= float(budget))
    return out


def _pick(row):
    """One hotel (Person 2's dict) -> the small dict the assistant expects."""
    if row is None:
        return None
    r = {RENAME.get(k, k): v for k, v in row.items()}
    return {k: r.get(k) for k in ["hotel_key", "name", "neighbourhood", "cheapest_price", "cheapest_site", "total"]}


def _meta(df, source_col="source"):
    """Price age and live/cached label for the 'Sources' line."""
    times = pd.to_datetime(df["retrieved_at"], errors="coerce", utc=True).dropna()
    oldest = times.min() if len(times) else None
    sources = df[source_col].fillna("unavailable").value_counts().to_dict() if source_col in df else {}
    if sources.get("cached_fallback") or sources.get("unavailable"):
        mode = "cached (the hotel API did not answer for some hotels; older cached prices used)"
    elif sources.get("live") and not sources.get("cached"):
        mode = "live"
    else:
        mode = "cached"
    age = None if oldest is None else round((datetime.now(timezone.utc) - oldest).total_seconds() / 3600, 1)
    return {"retrieved_at": oldest.isoformat() if oldest is not None else "unknown", "mode": mode,
            "age_hours": age, "source_counts": sources,
            "source": "Hotel prices: Xotelo API (Task 4, Person 2).", "is_stub": False}


def search_hotels(check_in, nights, guests, budget, month=None, live=False):
    """Interface version of Person 2's search. live=True -> fresh API calls for 20 hotels (demo)."""
    with contextlib.redirect_stdout(io.StringIO()):          # their code prints a lot; keep the app log clean
        res = _p2_search(check_in=check_in, nights=int(nights), guests=int(guests), budget=float(budget),
                         month=month, max_hotels=20 if live else None, refresh=bool(live))
        top5 = get_top5_neighborhoods(month or int(check_in[5:7]))[0].index.tolist()
    df = _reshape(res["hotels"], budget, nights).sort_values("cheapest_price", na_position="last")
    meta = _meta(res["hotels"])
    meta.update(cheapest=_pick(res["cheapest_overall"]), cheapest_outside_top5=_pick(res["cheapest_alternative"]),
                top5_neighbourhoods=top5)
    unavailable = meta["source_counts"].get("unavailable", 0)
    if unavailable:
        meta["source"] += f" No price available for {unavailable} hotels."
    return df.reset_index(drop=True), meta


def nearby(hotel_key, km=2, check_in=None):
    """Interface version of Person 2's find_nearby_hotels (distances by their haversine)."""
    _, df = find_nearby_hotels(hotel_key, radius_km=km)
    return df.rename(columns=RENAME)[["hotel_key", "name", "neighbourhood", "distance_km",
                                      "cheapest_price", "cheapest_site"]].reset_index(drop=True)


def all_hotels(check_in=None):
    """Every hotel from Person 2's saved Task 4 table (used to find hotels by name)."""
    return _reshape(pd.read_csv("outputs/t4_hotels.csv"), budget=10 ** 9)


def map_hotels(budget):
    """Hotels for the map page, using Person 2's saved prices (no API calls). Returns (df, meta)."""
    raw = pd.read_csv("outputs/t4_hotels.csv")
    df = _reshape(raw, budget)
    meta = _meta(raw.assign(source="cached"))
    meta["dates"] = f"{raw['check_in'].iloc[0]} to {raw['check_out'].iloc[0]}"
    return df, meta
