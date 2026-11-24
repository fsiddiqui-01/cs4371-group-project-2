
from pathlib import Path

import pandas as pd

HOTELS_FILE = Path("outputs/t4_hotels.csv")
INCIDENTS_FILE = Path("neighbourhood_month.csv")


def get_top5_incident_neighborhoods(month):
    """
    Return the five Seattle neighborhoods with
    the highest average reported incidents for
    the selected month across 2022-2024.
    """
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12.")

    incidents = pd.read_csv(INCIDENTS_FILE)

    # Use the three historical years required
    # by the assignment.
    selected = incidents[
        incidents["year"].between(2022, 2024)
        & (incidents["month"] == month)
    ].copy()

    selected["neighborhood"] = (
        selected["neighborhood"]
        .str.upper()
        .str.strip()
    )

    averages = (
        selected.groupby("neighborhood")["total"]
        .mean()
        .sort_values(ascending=False)
    )

    return averages.head(5)


def find_cheapest_hotels(budget, month):
    """
    Find:
    1. Cheapest hotel within budget.
    2. Cheapest hotel within budget outside
       the selected month's top 5 incident areas.
    """
    if budget <= 0:
        raise ValueError("Budget must be positive.")

    hotels = pd.read_csv(HOTELS_FILE)

    hotels["neighborhood"] = (
        hotels["neighborhood"]
        .str.upper()
        .str.strip()
    )

    # Remove hotels without usable prices.
    hotels = hotels.dropna(
        subset=["cheapest_nightly_rate"]
    )

    # Filter hotels by nightly budget.
    affordable = hotels[
        hotels["cheapest_nightly_rate"] <= budget
    ].copy()

    affordable = affordable.sort_values(
        by=["cheapest_nightly_rate", "name"]
    )

    top5 = get_top5_incident_neighborhoods(month)

    print(
        f"\n===== TOP 5 REPORTED INCIDENT AREAS "
        f"FOR MONTH {month} ====="
    )

    print(top5.to_string())

    if affordable.empty:
        print("\nNo hotels found within your budget.")
        return None, None

    cheapest_overall = affordable.iloc[0]

    # Exclude the top-five incident neighborhoods.
    alternatives = affordable[
        ~affordable["neighborhood"].isin(top5.index)
    ]

    cheapest_alternative = (
        alternatives.iloc[0]
        if not alternatives.empty
        else None
    )

    print("\n===== CHEAPEST OVERALL HOTEL =====")
    print(cheapest_overall[
        ["name", "neighborhood",
         "cheapest_site", "cheapest_nightly_rate"]
    ].to_string())

    print(
        "\n===== CHEAPEST HOTEL OUTSIDE "
        "THE TOP-5 INCIDENT AREAS ====="
    )

    if cheapest_alternative is not None:
        print(cheapest_alternative[
            ["name", "neighborhood",
             "cheapest_site", "cheapest_nightly_rate"]
        ].to_string())
    else:
        print("No qualifying alternative found.")

    return cheapest_overall, cheapest_alternative


if __name__ == "__main__":
    find_cheapest_hotels(
        budget=150,
        month=12,
    )
