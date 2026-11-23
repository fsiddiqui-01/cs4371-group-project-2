
import json
from pathlib import Path


def point_on_segment(px, py, x1, y1, x2, y2):
    """Check whether a point lies on an edge."""
    tolerance = 1e-10

    cross = (px - x1) * (y2 - y1) - (
        py - y1
    ) * (x2 - x1)

    if abs(cross) > tolerance:
        return False

    return (
        min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
    )


def point_in_ring(longitude, latitude, ring):
    """Ray casting for a single polygon ring."""
    inside = False
    n = len(ring)

    for i in range(n):
        x1, y1 = ring[i][:2]
        x2, y2 = ring[(i + 1) % n][:2]

        if point_on_segment(
            longitude, latitude, x1, y1, x2, y2
        ):
            return True

        if (y1 > latitude) != (y2 > latitude):
            x_intersection = x1 + (
                (latitude - y1) * (x2 - x1)
                / (y2 - y1)
            )

            if longitude < x_intersection:
                inside = not inside

    return inside


def point_in_polygon(longitude, latitude, coordinates):
    """
    GeoJSON polygon:
    First ring is exterior; remaining rings are holes.
    """
    if not coordinates:
        return False

    if not point_in_ring(
        longitude, latitude, coordinates[0]
    ):
        return False

    for hole in coordinates[1:]:
        if point_in_ring(longitude, latitude, hole):
            return False

    return True


def find_neighborhood(latitude, longitude, geojson):
    """Find a point's Seattle neighborhood."""
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            continue

        for polygon in polygons:
            if point_in_polygon(
                longitude, latitude, polygon
            ):
                return feature.get(
                    "properties", {}
                ).get("neighborhood")

    return None


def load_neighborhoods():
    path = Path(
        "cache/person2/seattle_neighborhoods.geojson"
    )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


from math import radians, sin, cos, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates.

    Returns:
        Distance in kilometers.
    """
    earth_radius = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    c = 2 * asin(sqrt(a))

    return earth_radius * c
