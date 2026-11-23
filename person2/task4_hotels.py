
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.xotelo.com/api/list"
LOCATION_KEY = "g60878"
CACHE_DIR = Path("cache/person2")
OUTPUT_DIR = Path("outputs")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_page(offset, limit=30):
    cache_file = CACHE_DIR / f"hotels_{offset}.json"

    if cache_file.exists():
        print(f"Loading cached page at offset {offset}")
        return json.loads(
            cache_file.read_text(encoding="utf-8")
        )["response"]

    print(f"Downloading page at offset {offset}")

    response = requests.get(
        URL,
        params={
            "location_key": LOCATION_KEY,
            "limit": limit,
            "offset": offset,
        },
        timeout=30,
        headers={"User-Agent": "SafeStay-CS4371/1.0"},
    )
    response.raise_for_status()
    data = response.json()

    if data.get("error") is not None:
        raise RuntimeError(f"Xotelo error: {data['error']}")

    cache_file.write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "response": data,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    time.sleep(1)
    return data


def collect_hotels(min_hotels=50):
    hotels_by_key = {}
    offset = 0

    while len(hotels_by_key) < min_hotels:
        data = fetch_page(offset)
        result = data.get("result", {})
        page = result.get("list", [])

        if not page:
            break

        for hotel in page:
            key = hotel.get("key")
            geo = hotel.get("geo") or {}

            if (
                key
                and hotel.get("name")
                and geo.get("latitude") is not None
                and geo.get("longitude") is not None
            ):
                hotels_by_key[key] = {
                    "hotel_key": key,
                    "name": hotel["name"],
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "accommodation_type": hotel.get(
                        "accommodation_type"
                    ),
                }

        page_offset = result.get("offset", offset)
        page_limit = result.get("limit", len(page))
        next_offset = page_offset + len(page)

        if (
            next_offset <= offset
            or len(page) < page_limit
            or next_offset >= result.get(
                "total_count", float("inf")
            )
        ):
            break

        offset = next_offset

    df = pd.DataFrame(hotels_by_key.values())

    output_path = OUTPUT_DIR / "seattle_hotels_basic.csv"
    df.to_csv(output_path, index=False)

    print(f"\nUsable unique hotels: {len(df)}")
    print(f"Saved to: {output_path}")

    if len(df) < min_hotels:
        print("WARNING: Fewer than 50 usable hotels.")

    if not df.empty:
        print(df.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    collect_hotels()
