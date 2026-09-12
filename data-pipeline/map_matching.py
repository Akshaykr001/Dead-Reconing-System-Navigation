"""Snap latitude/longitude coordinates to the nearest drivable road."""
from __future__ import annotations

import requests


OSRM_NEAREST_URL = "https://router.project-osrm.org/nearest/v1/driving"
REQUEST_TIMEOUT_SECONDS = 10


def snap_to_road(lat: float, lon: float) -> dict[str, float]:
    """Return the nearest road coordinate and a distance-based confidence score."""
    fallback = {"matched_lat": lat, "matched_lon": lon, "confidence": 0.0}
    url = f"{OSRM_NEAREST_URL}/{lon},{lat}"
    print(f"Requesting OSRM URL: {url}")

    response = None
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        print(f"OSRM response status: {response.status_code}")
        if response.status_code != 200:
            print(f"OSRM response body: {response.text}")
        response.raise_for_status()
        waypoint = response.json()["waypoints"][0]
        matched_lon, matched_lat = waypoint["location"]
        distance = float(waypoint["distance"])
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        print(f"OSRM request failed: {error}")
        if response is not None:
            print(f"OSRM response body: {response.text}")
        return fallback

    if distance < 5:
        confidence = 1.0
    elif distance >= 50:
        confidence = 0.0
    else:
        confidence = (50 - distance) / 45

    return {
        "matched_lat": matched_lat,
        "matched_lon": matched_lon,
        "confidence": confidence,
    }


if __name__ == "__main__":
    print(snap_to_road(40.7128, -74.0060))