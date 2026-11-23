
from pathlib import Path

import pandas as pd

from utils.geography import haversine


INPUT_FILE = Path("outputs/t4_hotels.csv")
OUTPUT_DIR = Path("outputs")


def find_nearby_hotels(hotel_key, radius_km=2.0):
    hotels = pd.read_csv(INPUT_FILE)

    selected = hotels[
        hotels["hotel_key"] == hotel_key
    ]

    if selected.empty:
        raise ValueError(
            f"Hotel not found: {hotel_key}"
        )

    selected = selected.iloc[0]
    nearby = []

    for _, hotel in hotels.iterrows():

        # Don't compare a hotel with itself
        if hotel["hotel_key"] == hotel_key:
            continue

        distance = haversine(
            selected["latitude"],
            selected["longitude"],
            hotel["latitude"],
            hotel["longitude"],
        )

        if distance <= radius_km:
            nearby.append({
                "selected_hotel": selected["name"],
                "hotel_key": hotel["hotel_key"],
                "name": hotel["name"],
                "neighborhood": hotel["neighborhood"],
                "distance_km": round(distance, 3),
                "cheapest_site": hotel["cheapest_site"],
                "cheapest_nightly_rate": hotel[
                    "cheapest_nightly_rate"
                ],
                "base_stay_total": hotel[
                    "base_stay_total"
                ],
                "retrieved_at": hotel["retrieved_at"],
            })

    result = pd.DataFrame(
        nearby,
        columns=[
            "selected_hotel",
            "hotel_key",
            "name",
            "neighborhood",
            "distance_km",
            "cheapest_site",
            "cheapest_nightly_rate",
            "base_stay_total",
            "retrieved_at",
        ],
    )

    result = result.sort_values(
        by=["cheapest_nightly_rate", "distance_km"],
        na_position="last",
    )

    return selected["name"], result


def main():
    # Two chosen hotels required by Task 4
    selected_hotels = [
        "g60878-d258824",  # Mediterranean Inn
        "g60878-d100507",  # Inn at the Market
    ]

    all_results = []

    for hotel_key in selected_hotels:
        hotel_name, nearby = find_nearby_hotels(
            hotel_key
        )

        print(f"\n===== {hotel_name} =====")
        print("Nearby hotels within 2 km:", len(nearby))

        if not nearby.empty:
            print(
                nearby[
                    [
                        "name",
                        "neighborhood",
                        "distance_km",
                        "cheapest_nightly_rate",
                    ]
                ].to_string(index=False)
            )

        filename = (
            OUTPUT_DIR / f"t4_nearby_{hotel_key}.csv"
        )
        nearby.to_csv(filename, index=False)

        all_results.append(nearby)

    # Combined assignment output
    combined = pd.concat(
        all_results, ignore_index=True
    )

    combined.to_csv(
        OUTPUT_DIR / "t4_nearby.csv",
        index=False,
    )

    print("\nSaved nearby hotel results.")


if __name__ == "__main__":
    main()
