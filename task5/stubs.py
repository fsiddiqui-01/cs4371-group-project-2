"""
STAND-IN versions of teammates' functions (Person 4 uses these until the real ones exist).

Each function has the SAME name, arguments and output columns agreed in the team plan
(see task5/INTERFACES.md). When a teammate pushes the real function, change ONE import
line in task5/backend.py and nothing else in the app has to change.

Everything returned here carries meta["is_stub"] = True so the app can show a
"placeholder data" warning. Hotels and attractions below are NOT real data.
"""
import calendar
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NM_CSV = ROOT / "neighbourhood_month.csv"          # Person 1's real Task 1 table (CSV copy)
TEMP_CSV = ROOT / "city_temperature.csv"           # Homework 2/4 temperature file


# ---------------------------------------------------------------- Person 1: Task 2
def outlook(month):
    """Task 2 table for one month: 15 rows (top 5 / largest seasonal increase / fewest).

    STUB: same formula as task2/task2.py, but as an importable function so the app can
    call it for any month. Person 1's real outlook(month) should replace this.
    Columns: list, neighborhood, avg_incidents, avg_persons, avg_property,
             share_of_city_pct, seasonal_ratio
    """
    d = pd.read_csv(NM_CSV)
    hoods = sorted(d["neighborhood"].unique())
    full = pd.MultiIndex.from_product([hoods, [2022, 2023, 2024], range(1, 13)],
                                      names=["neighborhood", "year", "month"])
    d = d.set_index(["neighborhood", "year", "month"]).reindex(full).reset_index()
    for c in ["total", "persons", "property"]:
        d[c] = d[c].fillna(0)
    sel = d[d.month == month].groupby("neighborhood")[["total", "persons", "property"]].mean()
    sel.columns = ["avg_incidents", "avg_persons", "avg_property"]
    sel["share_of_city_pct"] = sel.avg_incidents / sel.avg_incidents.sum() * 100
    sel["seasonal_ratio"] = sel.avg_incidents / d.groupby("neighborhood")["total"].mean()
    rows = pd.concat([sel.nlargest(5, "avg_incidents").assign(list="most_incidents"),
                      sel.nlargest(5, "seasonal_ratio").assign(list="largest_seasonal_increase"),
                      sel.nsmallest(5, "avg_incidents").assign(list="fewest_incidents")]).reset_index()
    for c in ["avg_incidents", "avg_persons", "avg_property", "share_of_city_pct"]:
        rows[c] = rows[c].round(1)
    rows["seasonal_ratio"] = rows["seasonal_ratio"].round(2)
    return rows[["list", "neighborhood", "avg_incidents", "avg_persons", "avg_property",
                 "share_of_city_pct", "seasonal_ratio"]]


def month_incidents_by_hood(month):
    """Average incidents per neighbourhood for one month (used by the places() stub)."""
    d = pd.read_csv(NM_CSV)
    return (d[d.month == month].groupby("neighborhood")["total"].sum() / 3).round(1)


# ---------------------------------------------------------------- Person 2: Task 4
# 30 made-up hotels: (name, lat, lon, neighbourhood). NOT real hotels.
_HOTELS = [
    ("Stub Hotel Pike", 47.6101, -122.3405, "DOWNTOWN COMMERCIAL"),
    ("Stub Hotel Union", 47.6089, -122.3350, "DOWNTOWN COMMERCIAL"),
    ("Stub Hotel Sixth Ave", 47.6120, -122.3330, "DOWNTOWN COMMERCIAL"),
    ("Stub Hotel Waterfront", 47.6060, -122.3400, "DOWNTOWN COMMERCIAL"),
    ("Stub Hotel Convention", 47.6115, -122.3315, "DOWNTOWN COMMERCIAL"),
    ("Stub Hotel Belltown", 47.6150, -122.3470, "BELLTOWN"),
    ("Stub Hotel Elliott", 47.6165, -122.3520, "BELLTOWN"),
    ("Stub Hotel Denny", 47.6180, -122.3440, "BELLTOWN"),
    ("Stub Hotel Seattle Center", 47.6230, -122.3530, "QUEEN ANNE"),
    ("Stub Hotel Uptown", 47.6250, -122.3570, "QUEEN ANNE"),
    ("Stub Hotel Mercer", 47.6245, -122.3470, "QUEEN ANNE"),
    ("Stub Hotel Lake Union", 47.6260, -122.3370, "SLU/CASCADE"),
    ("Stub Hotel Westlake", 47.6210, -122.3390, "SLU/CASCADE"),
    ("Stub Hotel Fairview", 47.6230, -122.3300, "SLU/CASCADE"),
    ("Stub Hotel Pioneer", 47.6005, -122.3335, "PIONEER SQUARE"),
    ("Stub Hotel Occidental", 47.5990, -122.3330, "PIONEER SQUARE"),
    ("Stub Hotel Jackson St", 47.5985, -122.3235, "CHINATOWN/INTERNATIONAL DISTRICT"),
    ("Stub Hotel First Hill", 47.6080, -122.3240, "FIRST HILL"),
    ("Stub Hotel Broadway", 47.6190, -122.3210, "CAPITOL HILL"),
    ("Stub Hotel Pike-Pine", 47.6140, -122.3200, "CAPITOL HILL"),
    ("Stub Hotel U District", 47.6610, -122.3130, "UNIVERSITY"),
    ("Stub Hotel Roosevelt", 47.6760, -122.3170, "ROOSEVELT/RAVENNA"),
    ("Stub Hotel Fremont", 47.6510, -122.3500, "FREMONT"),
    ("Stub Hotel Ballard", 47.6680, -122.3840, "BALLARD SOUTH"),
    ("Stub Hotel Wallingford", 47.6610, -122.3340, "WALLINGFORD"),
    ("Stub Hotel Northgate", 47.7060, -122.3260, "NORTHGATE"),
    ("Stub Hotel SoDo", 47.5800, -122.3340, "SODO"),
    ("Stub Hotel Georgetown", 47.5470, -122.3200, "GEORGETOWN"),
    ("Stub Hotel Alki", 47.5790, -122.4080, "ALKI"),
    ("Stub Hotel Columbia City", 47.5600, -122.2860, "COLUMBIA CITY"),
]
_SITES = ["Booking.com", "Agoda", "Trip.com", "Hotel website"]
# pretend the cache was filled 3 hours ago
_RETRIEVED_AT = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(microsecond=0)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Stub copy only. Person 2 must write their own haversine in geo.py."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _prices(i, check_in):
    """Deterministic fake prices per booking site (same input -> same prices)."""
    rnd = random.Random(f"{i}-{check_in}")
    base = rnd.randint(95, 340)
    out = {}
    for s in _SITES:
        out[s] = None if rnd.random() < 0.15 else round(base * rnd.uniform(0.92, 1.12))
    if all(v is None for v in out.values()):
        out["Booking.com"] = base
    return out


def all_hotels(check_in=None):
    """Every hotel with its per-site prices (used by the map page and nearby())."""
    check_in = check_in or (date.today() + timedelta(days=30)).isoformat()
    rows = []
    for i, (name, lat, lon, hood) in enumerate(_HOTELS):
        p = _prices(i, check_in)
        cheapest_site = min((s for s in p if p[s] is not None), key=lambda s: p[s])
        rows.append({"hotel_key": f"stub{i:02d}", "name": name, "lat": lat, "lon": lon,
                     "neighbourhood": hood,
                     **{f"price_{s.lower().replace('.', '_').replace(' ', '_')}": p[s] for s in _SITES},
                     "cheapest_price": p[cheapest_site], "cheapest_site": cheapest_site})
    return pd.DataFrame(rows)


def map_hotels(budget):
    """Hotels for the map page (no API calls). Returns (df, meta)."""
    df = all_hotels()
    df["within_budget"] = df["cheapest_price"] <= float(budget)
    age = (datetime.now(timezone.utc) - _RETRIEVED_AT).total_seconds() / 3600
    return df, {"retrieved_at": _RETRIEVED_AT.isoformat(), "mode": "cached", "age_hours": round(age, 1),
                "dates": "placeholder dates", "source": "Hotel prices: STUB placeholder data", "is_stub": True}


def search_hotels(check_in, nights, guests, budget, month=None, live=False):
    """Task 4 search.

    Returns (df, meta).
      df columns: hotel_key, name, lat, lon, neighbourhood, price_<site>..., cheapest_price,
                  cheapest_site, total, within_budget
      meta: retrieved_at (ISO, UTC), mode ("live" or "cached"), age_hours,
            cheapest (dict or None), cheapest_outside_top5 (dict or None),
            top5_neighbourhoods (list), source, is_stub
    """
    df = all_hotels(check_in)
    df["total"] = df["cheapest_price"] * int(nights)
    df["within_budget"] = df["cheapest_price"] <= float(budget)
    df = df.sort_values("cheapest_price").reset_index(drop=True)

    month = month or int(str(check_in)[5:7])
    top5 = outlook(month).query("list == 'most_incidents'")["neighborhood"].tolist()
    ok = df[df.within_budget]
    calm = ok[~ok.neighbourhood.isin(top5)]
    pick = lambda t: None if t.empty else t.iloc[0][["hotel_key", "name", "neighbourhood",
                                                       "cheapest_price", "cheapest_site", "total"]].to_dict()
    age = (datetime.now(timezone.utc) - _RETRIEVED_AT).total_seconds() / 3600
    meta = {"retrieved_at": _RETRIEVED_AT.isoformat(), "mode": "cached", "age_hours": round(age, 1),
            "cheapest": pick(ok), "cheapest_outside_top5": pick(calm),
            "top5_neighbourhoods": top5,
            "source": "Hotel prices: STUB placeholder data (will be Xotelo)", "is_stub": True}
    return df, meta


def nearby(hotel_key, km=2, check_in=None):
    """Every other hotel within km of the chosen hotel, cheapest first.
    Columns: hotel_key, name, neighbourhood, distance_km, cheapest_price, cheapest_site"""
    df = all_hotels(check_in)
    me = df[df.hotel_key == hotel_key].iloc[0]
    df["distance_km"] = [round(_haversine_km(me.lat, me.lon, a, b), 2) for a, b in zip(df.lat, df.lon)]
    out = df[(df.hotel_key != hotel_key) & (df.distance_km <= km)]
    return out.sort_values("cheapest_price")[["hotel_key", "name", "neighbourhood", "distance_km",
                                              "cheapest_price", "cheapest_site"]].reset_index(drop=True)


# ---------------------------------------------------------------- Person 3: Task 6
def best_months():
    """Task 6 table, one row per month.
    Columns: month, month_name, avg_temp_f, temp_distance_from_70, avg_incidents, score, rank

    STUB: temperatures and incidents are real, but the scoring rule (equal weights) is a
    placeholder. Person 3 defines and justifies the real rule.
    """
    t = pd.read_csv(TEMP_CSV)
    t = t[(t.City == "Seattle") & (t.AvgTemperature > -99)]
    temp = t.groupby("Month")["AvgTemperature"].mean()
    inc = pd.read_csv(NM_CSV).groupby("month")["total"].sum() / 3
    df = pd.DataFrame({"month": range(1, 13)})
    df["month_name"] = [calendar.month_name[m] for m in df.month]
    df["avg_temp_f"] = df.month.map(temp).round(1)
    df["temp_distance_from_70"] = (df.avg_temp_f - 70).abs().round(1)
    df["avg_incidents"] = df.month.map(inc).round(1)
    norm = lambda s: (s - s.min()) / (s.max() - s.min())
    df["score"] = (0.5 * norm(df.temp_distance_from_70) + 0.5 * norm(df.avg_incidents)).round(3)
    df["rank"] = df.score.rank(method="min").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------- Person 2: Task 8
# Approximate coordinates of real Seattle places; the scores below are placeholders.
_ATTRACTIONS = [
    ("Space Needle", 47.6205, -122.3493, "attraction", True, "QUEEN ANNE"),
    ("Pike Place Market", 47.6097, -122.3422, "attraction", True, "DOWNTOWN COMMERCIAL"),
    ("Museum of Pop Culture", 47.6215, -122.3481, "museum", True, "QUEEN ANNE"),
    ("Chihuly Garden and Glass", 47.6206, -122.3505, "museum", True, "QUEEN ANNE"),
    ("Seattle Art Museum", 47.6073, -122.3381, "museum", True, "DOWNTOWN COMMERCIAL"),
    ("Seattle Aquarium", 47.6076, -122.3430, "aquarium", True, "DOWNTOWN COMMERCIAL"),
    ("Olympic Sculpture Park", 47.6166, -122.3553, "park", True, "BELLTOWN"),
    ("Woodland Park Zoo", 47.6685, -122.3504, "zoo", True, "PHINNEY RIDGE"),
    ("Kerry Park", 47.6295, -122.3600, "viewpoint", True, "QUEEN ANNE"),
    ("Discovery Park", 47.6573, -122.4057, "park", True, "MAGNOLIA"),
    ("Gas Works Park", 47.6456, -122.3344, "park", True, "WALLINGFORD"),
    ("Museum of Flight", 47.5180, -122.2964, "museum", True, "GEORGETOWN"),
    ("Volunteer Park", 47.6302, -122.3150, "park", True, "CAPITOL HILL"),
    ("Burke Museum", 47.6606, -122.3105, "museum", True, "UNIVERSITY"),
    ("Alki Beach Park", 47.5790, -122.4100, "park", True, "ALKI"),
    ("Wing Luke Museum", 47.5980, -122.3230, "museum", True, "CHINATOWN/INTERNATIONAL DISTRICT"),
    ("Klondike Gold Rush Museum", 47.6000, -122.3340, "museum", True, "PIONEER SQUARE"),
    ("Museum of History & Industry", 47.6275, -122.3365, "museum", True, "SLU/CASCADE"),
    ("Smith Tower Observatory", 47.6019, -122.3318, "viewpoint", False, "PIONEER SQUARE"),
    ("Fremont Troll", 47.6510, -122.3473, "attraction", False, "FREMONT"),
]


def all_attractions(month):
    inc = month_incidents_by_hood(month)
    hotels = all_hotels()
    rows = []
    for name, lat, lon, kind, wiki, hood in _ATTRACTIONS:
        d = [_haversine_km(lat, lon, a, b) for a, b in zip(hotels.lat, hotels.lon)]
        near = hotels[[x <= 1.0 for x in d]]
        best = near.sort_values("cheapest_price").head(1)
        rows.append({"name": name, "lat": lat, "lon": lon, "kind": kind, "has_wikipedia": wiki,
                     "neighbourhood": hood, "avg_incidents_month": inc.get(hood),
                     "hotels_within_1km": len(near),
                     "cheapest_nearby_hotel": None if best.empty else best.iloc[0]["name"],
                     "cheapest_nearby_price": None if best.empty else int(best.iloc[0]["cheapest_price"])})
    df = pd.DataFrame(rows)
    norm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0
    df["score"] = (0.4 * df.has_wikipedia + 0.3 * norm(df.hotels_within_1km)
                   - 0.3 * norm(df.avg_incidents_month.fillna(0))).round(3)
    df["rank"] = df.score.rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def places(month, budget=None):
    """Task 8. Returns (top10_attractions, top5_neighbourhoods).
    top10 columns: rank, name, lat, lon, kind, has_wikipedia, neighbourhood, avg_incidents_month,
                   hotels_within_1km, cheapest_nearby_hotel, cheapest_nearby_price, score
    top5 columns:  rank, neighbourhood, attractions, avg_incidents_month, hotels_in_budget,
                   median_price, score
    STUB: placeholder scoring rules; Person 2 defines the real ones."""
    att = all_attractions(month)
    hotels = all_hotels()
    if budget:
        hotels = hotels[hotels.cheapest_price <= float(budget)]
    g = att.groupby("neighbourhood").agg(attractions=("name", "count"),
                                         avg_incidents_month=("avg_incidents_month", "first")).reset_index()
    hb = hotels.groupby("neighbourhood")["cheapest_price"].agg(["count", "median"])
    g["hotels_in_budget"] = g.neighbourhood.map(hb["count"]).fillna(0).astype(int)
    g["median_price"] = g.neighbourhood.map(hb["median"]).round(0)
    norm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0
    g["score"] = (0.4 * norm(g.attractions) + 0.3 * norm(g.hotels_in_budget)
                  - 0.3 * norm(g.avg_incidents_month.fillna(0))).round(3)
    g["rank"] = g.score.rank(ascending=False, method="first").astype(int)
    top5 = g.sort_values("rank").head(5)[["rank", "neighbourhood", "attractions", "avg_incidents_month",
                                          "hotels_in_budget", "median_price", "score"]]
    top10 = att.head(10)[["rank", "name", "lat", "lon", "kind", "has_wikipedia", "neighbourhood",
                          "avg_incidents_month", "hotels_within_1km", "cheapest_nearby_hotel",
                          "cheapest_nearby_price", "score"]]
    return top10.reset_index(drop=True), top5.reset_index(drop=True)
