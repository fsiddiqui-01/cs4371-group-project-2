
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# Existing project files
HOTELS_FILE = Path("outputs/t4_hotels.csv")
INCIDENTS_FILE = Path("neighbourhood_month.csv")

# New output and cache
OUTPUT_FILE = Path("outputs/t4_hotels_dynamic.csv")
CACHE_DIR = Path("cache/person2/prices")
API_URL = "https://data.xotelo.com/api/rates"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
last_request_time = None


def calculate_cache_age_minutes(retrieved_at):
    """Return the age of an API response in minutes."""
    if not retrieved_at:
        return None

    retrieved = datetime.fromisoformat(retrieved_at)

    if retrieved.tzinfo is None:
        retrieved = retrieved.replace(tzinfo=timezone.utc)

    age = datetime.now(timezone.utc) - retrieved

    return round(max(0, age.total_seconds() / 60), 2)

def get_prices(hotel_key, check_in, check_out, refresh=False):
    """Fetch prices or reuse the matching cached response."""
    global last_request_time

    cache_file = CACHE_DIR / (
        f"{hotel_key}_{check_in}_{check_out}.json"
    )

    cached = None

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except (ValueError, OSError):
            print(f"Invalid cache: {hotel_key}")

    if cached is not None and not refresh:
        return (
            cached["response"],
            cached["retrieved_at"],
            "cached",
        )

    # At least one second between live requests
    if last_request_time is not None:
        elapsed = time.monotonic() - last_request_time
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

    try:
        last_request_time = time.monotonic()

        response = session.get(
            API_URL,
            params={
                "hotel_key": hotel_key,
                "chk_in": check_in,
                "chk_out": check_out,
            },
            timeout=30,
            headers={
                "User-Agent": "SafeStay-CS4371/1.0"
            },
        )

        response.raise_for_status()
        data = response.json()

        if data.get("error") is not None:
            raise RuntimeError(str(data["error"]))

        if not isinstance(data.get("result"), dict):
            raise RuntimeError("Unexpected API response")

        retrieved_at = datetime.now(
            timezone.utc
        ).isoformat()

        cache_file.write_text(
            json.dumps({
                "retrieved_at": retrieved_at,
                "response": data,
            }, indent=2)
        )

        return data, retrieved_at, "live"

    except (requests.RequestException,
            ValueError, RuntimeError) as error:

        print(f"API failed for {hotel_key}: {error}")

        if cached is not None:
            print("Using older cached prices.")
            return (
                cached["response"],
                cached["retrieved_at"],
                "cached_fallback",
            )

        return None, None, "unavailable"


def parse_rates(data, nights):
    """Find the cheapest booking site and base stay total."""
    rates = (data.get("result") or {}).get("rates") or []
    valid = []

    for item in rates:
        rate = item.get("rate")

        if (
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and rate > 0
        ):
            valid.append({
                "site": item.get("name"),
                "nightly_rate": rate,
                "tax": item.get("tax"),
            })

    if not valid:
        return None, None, None, valid

    cheapest = min(valid, key=lambda x: x["nightly_rate"])

    return (
        cheapest["site"],
        cheapest["nightly_rate"],
        cheapest["nightly_rate"] * nights,
        valid,
    )


def get_top5_neighborhoods(month):
    """Top five by mean monthly incidents, 2022-2024."""
    incidents = pd.read_csv(INCIDENTS_FILE)

    selected = incidents[
        incidents["year"].between(2022, 2024)
        & (incidents["month"] == month)
    ].copy()

    selected["neighborhood"] = (
        selected["neighborhood"].str.upper().str.strip()
    )

    # Detect missing historical years rather than
    # treating missing data as zero.
    counts = selected.groupby("neighborhood")["year"].nunique()
    incomplete = counts[counts < 3]

    if not incomplete.empty:
        print(
            "WARNING: Some neighborhoods have fewer than "
            "three years of data."
        )
        print(incomplete.to_string())

    # Rank neighborhoods with complete 3-year data.
    complete = counts[counts == 3].index
    selected = selected[
        selected["neighborhood"].isin(complete)
    ]

    averages = (
        selected.groupby("neighborhood")["total"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(averages) < 5:
        raise ValueError(
            "Insufficient complete incident data."
        )

    return averages.head(5), len(incomplete)


def search_hotels(
    check_in,
    nights,
    guests,
    budget,
    month=None,
    max_hotels=None,
    refresh=False,
):
    """
    Search hotels for specified dates and budget.

    Guest count is recorded but Xotelo's documented
    rates request does not include a guest parameter.
    """

    # Validate date format
    try:
        arrival = date.fromisoformat(check_in)
    except (ValueError, TypeError):
        raise ValueError(
            "Check-in must use YYYY-MM-DD format."
        )

    today = date.today()

    if arrival <= today:
        raise ValueError(
            "Check-in must be a future date."
        )

    # Xotelo supports dates within approximately
    # 11 months. Use a conservative 330-day limit.
    if (arrival - today).days > 330:
        raise ValueError(
            "Check-in is too far in the future "
            "for the supported hotel search window."
        )

    if type(nights) is not int or nights < 1:
        raise ValueError(
            "Nights must be a positive whole number."
        )

    if type(guests) is not int or guests < 1:
        raise ValueError(
            "Guests must be a positive whole number."
        )

    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not 0 < budget < float("inf")
    ):
        raise ValueError(
            "Budget must be a positive finite number."
        )

    if month is None:
        month = arrival.month

    if type(month) is not int or not 1 <= month <= 12:
        raise ValueError(
            "Month must be a whole number from 1 to 12."
        )

    check_out_date = arrival + timedelta(days=nights)

    if (check_out_date - today).days > 330:
        raise ValueError(
            "Checkout is outside the supported "
            "hotel booking window."
        )

    check_out = check_out_date.isoformat()


    print("\n===== SEATTLE HOTEL SEARCH =====")
    print("Check-in:", check_in)
    print("Check-out:", check_out)
    print("Guests requested:", guests)
    print("Nightly budget:", budget)
    print("Incident comparison month:", month)
    print(
        "Note: Guest-specific prices are not verified "
        "by this Xotelo endpoint."
    )

    hotels = pd.read_csv(HOTELS_FILE)

    if max_hotels is not None:
        hotels = hotels.head(max_hotels)

    results = []

    for index, hotel in hotels.iterrows():
        print(f"\nChecking: {hotel['name']}")

        data, retrieved_at, source = get_prices(
            hotel["hotel_key"],
            check_in,
            check_out,
            refresh=refresh,
        )

        site = nightly = total = None
        rates = []

        if data is not None:
            site, nightly, total, rates = parse_rates(
                data, nights
            )

        results.append({
            "hotel_key": hotel["hotel_key"],
            "name": hotel["name"],
            "latitude": hotel["latitude"],
            "longitude": hotel["longitude"],
            "neighborhood": hotel["neighborhood"],
            "check_in": check_in,
            "check_out": check_out,
            "nights": nights,
            "guests_requested": guests,
            "currency": "USD",
            "cheapest_site": site,
            "cheapest_nightly_rate": nightly,
            "base_stay_total": total,
            "prices_by_site": json.dumps(rates),
            "retrieved_at": retrieved_at,
            "source": source,
            "cache_age_minutes": (
                calculate_cache_age_minutes(retrieved_at)
            ),
        })

    df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    affordable = df[
        df["cheapest_nightly_rate"].notna()
        & (df["cheapest_nightly_rate"] <= budget)
    ].sort_values("cheapest_nightly_rate")

    if affordable.empty:
        print("\nNo hotels with verified rates fit the budget.")
        return {
            "cheapest_overall": None,
            "cheapest_alternative": None,
            "hotels": df,
        }

    cheapest = affordable.iloc[0]

    print("\n===== CHEAPEST OVERALL =====")
    print(cheapest[
        ["name", "neighborhood", "cheapest_site",
         "cheapest_nightly_rate", "base_stay_total",
         "source", "retrieved_at"]
    ].to_string())

    top5, incomplete_count = get_top5_neighborhoods(month)

    print("\n===== TOP FIVE INCIDENT NEIGHBORHOODS =====")
    print(top5.to_string())

    alternatives = affordable[
        ~affordable["neighborhood"].str.upper().str.strip()
        .isin(top5.index)
    ]

    alternative = (
        alternatives.iloc[0]
        if not alternatives.empty
        else None
    )

    if alternative is not None:
        print("\n===== CHEAPEST OUTSIDE TOP FIVE =====")
        print(alternative[
            ["name", "neighborhood", "cheapest_site",
             "cheapest_nightly_rate", "base_stay_total",
             "source", "retrieved_at"]
        ].to_string())


    else:
        print("\n===== CHEAPEST OUTSIDE TOP FIVE =====")
        print(
        "No qualifying hotel found outside the "
        "five highest-incident neighborhoods "
        "within the searched hotels and budget."
        )


    if incomplete_count:
        print(
            "\nWARNING: Incident ranking is provisional "
            "until missing year coverage is reviewed."
        )

    print(f"\nSaved: {OUTPUT_FILE}")

    return {
        "cheapest_overall": cheapest.to_dict(),
        "cheapest_alternative": (
            alternative.to_dict()
            if alternative is not None else None
        ),
        "hotels": df,
    }


if __name__ == "__main__":
    search_hotels(
        check_in="2027-01-15",
        nights=3,
        guests=2,
        budget=150,
        month=1,
        max_hotels=None,
    )
