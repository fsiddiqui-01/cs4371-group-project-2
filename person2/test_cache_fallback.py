
import requests
from unittest.mock import patch
from datetime import datetime, timezone

from person2.dynamic_hotel_search import get_prices

HOTEL_KEY = "g60878-d258824"
CHECK_IN = "2027-01-15"
CHECK_OUT = "2027-01-18"


def simulate_failure():
    print("===== SIMULATING XOTELO OUTAGE =====")

    # Prevent a real API call and simulate failure.
    with patch(
        "person2.dynamic_hotel_search.session.get",
        side_effect=requests.exceptions.ConnectionError(
        "Simulated outage"
        ),
    ):
        # refresh=True attempts live retrieval first,
        # then falls back to the existing cache.
        data, retrieved_at, source = get_prices(
            HOTEL_KEY,
            CHECK_IN,
            CHECK_OUT,
            refresh=True,
        )

    print("Source:", source)
    print("Retrieved at:", retrieved_at)

    if retrieved_at:
        retrieved = datetime.fromisoformat(retrieved_at)
        age = datetime.now(timezone.utc) - retrieved
        print(
            "Cache age (minutes):",
            round(age.total_seconds() / 60, 2),
        )

    if source == "cached_fallback" and data:
        print("PASS: Cached prices used during outage.")
    else:
        print("FAIL: Cached fallback did not work.")


if __name__ == "__main__":
    simulate_failure()
