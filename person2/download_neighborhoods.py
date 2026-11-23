
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = (
    "https://services.arcgis.com/"
    "ZOyb2t4B0UYuYNYH/ArcGIS/rest/services/"
    "SPD_Boundaries/FeatureServer/0/query"
)

OUTPUT = Path("cache/person2/seattle_neighborhoods.geojson")
METADATA = Path("cache/person2/neighborhoods_metadata.json")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    data = json.loads(OUTPUT.read_text())
    print("Using cached Seattle neighborhoods.")
else:
    response = requests.get(
        URL,
        params={
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("type") != "FeatureCollection":
        raise RuntimeError(
            "Unexpected API response: "
            + json.dumps(data)[:1000]
        )

    OUTPUT.write_text(json.dumps(data, indent=2))

    METADATA.write_text(
        json.dumps({
            "source": URL,
            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "feature_count": len(data["features"]),
        }, indent=2)
    )

features = data.get("features", [])

print("Neighborhood polygons:", len(features))

if features:
    print("\nFirst neighborhood properties:")
    print(json.dumps(features[0]["properties"], indent=2))

    print("\nGeometry type:")
    print(features[0]["geometry"]["type"])

print("\nSaved to:", OUTPUT)
