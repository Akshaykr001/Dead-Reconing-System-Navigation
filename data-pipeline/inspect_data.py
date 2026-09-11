"""Plot the route and print quick quality checks for a prepared segment."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def haversine_m(lat: pd.Series, lon: pd.Series) -> float:
    earth_radius_m = 6_371_000.0
    lat1, lat2 = np.radians(lat.iloc[:-1].to_numpy()), np.radians(lat.iloc[1:].to_numpy())
    delta_lat = lat2 - lat1
    delta_lon = np.radians(lon.iloc[1:].to_numpy() - lon.iloc[:-1].to_numpy())
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return float(np.sum(2 * earth_radius_m * np.arcsin(np.sqrt(a))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("driving_segment.csv"))
    parser.add_argument("--plot", type=Path, default=Path("gps_route.png"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"])
    duration = (frame.timestamp.iloc[-1] - frame.timestamp.iloc[0]).total_seconds()
    distance = haversine_m(frame.gps_lat, frame.gps_lon)
    print(f"Rows: {len(frame):,}")
    print(f"Duration: {duration / 60:.2f} minutes")
    print(f"Distance: {distance / 1000:.3f} km")
    print(f"Speed range: {frame.vehicle_speed.min():.2f} to {frame.vehicle_speed.max():.2f} km/h")
    print(f"Missing values: {int(frame.isna().sum().sum())}")

    plt.figure(figsize=(8, 6))
    plt.plot(frame.gps_lon, frame.gps_lat, linewidth=1.2)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("IO-VNBD driving segment GPS route")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Saved route plot to {args.plot}")


if __name__ == "__main__":
    main()
