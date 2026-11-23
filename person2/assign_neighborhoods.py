
import pandas as pd

from utils.geography import (
    find_neighborhood,
    load_neighborhoods,
)


def main():
    geojson = load_neighborhoods()

    input_file = "outputs/t4_hotels.csv"
    output_file = "outputs/t4_hotels.csv"

    hotels = pd.read_csv(input_file)

    neighborhoods = []

    for _, hotel in hotels.iterrows():
        neighborhood = find_neighborhood(
            hotel["latitude"],
            hotel["longitude"],
            geojson,
        )

        neighborhoods.append(neighborhood)

        print(
            f"{hotel['name']} -> "
            f"{neighborhood or 'NOT FOUND'}"
        )

    hotels["neighborhood"] = neighborhoods

    hotels.to_csv(output_file, index=False)

    print("\n===== SUMMARY =====")
    print("Hotels processed:", len(hotels))
    print(
        "Hotels assigned to neighborhoods:",
        hotels["neighborhood"].notna().sum()
    )
    print(
        "Hotels without neighborhoods:",
        hotels["neighborhood"].isna().sum()
    )


if __name__ == "__main__":
    main()
