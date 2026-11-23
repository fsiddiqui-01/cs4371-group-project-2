
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Configuration
HOTELS_CSV = Path("outputs/seattle_hotels_basic.csv")
OUTPUT_CSV = Path("outputs/t4_hotels.csv")
CACHE_DIR = Path("cache/person2/prices")

CHECK_IN = "2026-12-23"
CHECK_OUT = "2026-12-26"
NIGHTS = 3

API_URL = "https://data.xotelo.com/api/rates"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
last_request_time = None


def fetch_hotel_prices(hotel_key):
    global last_request_time

    # Cache is specific to hotel and booking dates.
    cache_file = CACHE_DIR / (
        f"{hotel_key}_{CHECK_IN}_{CHECK_OUT}.json"
    )

    if cache_file.exists():
        print(f"Using cached prices: {hotel_key}")
        saved = json.loads(cache_file.read_text())
        return saved["response"], saved["retrieved_at"]

    # Wait at least 1 second between requests.
    if last_request_time is not None:
        elapsed = time.monotonic() - last_request_time
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

    print(f"Fetching live prices: {hotel_key}")

    last_request_time = time.monotonic()

    response = session.get(
        API_URL,
        params={
            "hotel_key": hotel_key,
            "chk_in": CHECK_IN,
            "chk_out": CHECK_OUT,
        },
        headers={
            "User-Agent": "SafeStay-CS4371/1.0"
        },
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("error") is not None:
        raise RuntimeError(str(data["error"]))

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    # Save the complete API response.
    cache_file.write_text(
        json.dumps(
            {
                "retrieved_at": retrieved_at,
                "response": data,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return data, retrieved_at


def extract_prices(data):
    result = data.get("result") or {}
    rates = result.get("rates") or []

    valid_rates = []

    for item in rates:
        rate = item.get("rate")

        if isinstance(rate, (int, float)) and rate > 0:
            valid_rates.append({
                "site": item.get("name"),
                "code": item.get("code"),
                "nightly_rate": rate,
                "tax": item.get("tax"),
            })

    if not valid_rates:
        return None, None, valid_rates

    cheapest = min(
        valid_rates,
        key=lambda item: item["nightly_rate"]
    )

    return (
        cheapest["site"],
        cheapest["nightly_rate"],
        valid_rates,
    )


def collect_all_prices():
    hotels = pd.read_csv(HOTELS_CSV)
    results = []

    total = len(hotels)

    for index, hotel in hotels.iterrows():
        hotel_key = hotel["hotel_key"]

        print(f"\nHotel {index + 1}/{total}: {hotel['name']}")

        cheapest_site = None
        cheapest_rate = None
        all_rates = []
        retrieved_at = None
        status = "no_rates"

        try:
            data, retrieved_at = fetch_hotel_prices(
                hotel_key
            )

            (
                cheapest_site,
                cheapest_rate,
                all_rates,
            ) = extract_prices(data)

            if cheapest_rate is not None:
                status = "success"
                print(
                    f"Cheapest: {cheapest_site} "
                    f"${cheapest_rate}/night"
                )
            else:
                print("No usable rates returned.")

        except (requests.RequestException, ValueError,
                RuntimeError) as error:
            status = "api_error"
            print(f"Error: {error}")

        total_cost = (
            cheapest_rate * NIGHTS
            if cheapest_rate is not None
            else None
        )

        results.append({
            "hotel_key": hotel_key,
            "name": hotel["name"],
            "latitude": hotel["latitude"],
            "longitude": hotel["longitude"],
            "neighborhood": None,
            "check_in": CHECK_IN,
            "check_out": CHECK_OUT,
            "nights": NIGHTS,
            "currency": "USD",
            "cheapest_site": cheapest_site,
            "cheapest_nightly_rate": cheapest_rate,
            "base_stay_total": total_cost,
            "prices_by_site": json.dumps(all_rates),
            "retrieved_at": retrieved_at,
            "status": status,
        })

        # Save partial results after every hotel.
        pd.DataFrame(results).to_csv(
            OUTPUT_CSV, index=False
        )

    df = pd.DataFrame(results)

    print("\n===== FINAL SUMMARY =====")
    print("Hotels processed:", len(df))
    print(
        "Hotels with prices:",
        (df["status"] == "success").sum()
    )

    priced = df.dropna(
        subset=["cheapest_nightly_rate"]
    )

    if not priced.empty:
        print("\n5 cheapest hotels:")
        print(
            priced.nsmallest(
                5, "cheapest_nightly_rate"
            )[
                [
                    "name",
                    "cheapest_site",
                    "cheapest_nightly_rate",
                    "base_stay_total",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    collect_all_prices()
