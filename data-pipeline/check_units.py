"""Check likely units for vehicle speed and gyro Z-rate sensor data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def print_statistics(name: str, values: pd.Series) -> None:
    print(f"{name} statistics:")
    print(f"  Min:    {values.min():.6f}")
    print(f"  Max:    {values.max():.6f}")
    print(f"  Mean:   {values.mean():.6f}")
    print(f"  Median: {values.median():.6f}")


def main() -> None:
    data_dir = Path(__file__).parent
    outage_path = data_dir / "driving_segment_gnss_outage.csv"
    input_path = outage_path if outage_path.exists() else data_dir / "driving_segment.csv"
    frame = pd.read_csv(input_path)

    vehicle_speed = frame["vehicle_speed"]
    gyro_z = frame["gyro_z"]
    print(f"Loaded: {input_path.name}")
    print(f"Rows: {len(frame):,}\n")

    print_statistics("vehicle_speed", vehicle_speed)
    vehicle_speed_mean = vehicle_speed.mean()
    if 0 <= vehicle_speed_mean <= 35:
        vehicle_speed_verdict = "Likely meters/second (correct)"
    elif 35 < vehicle_speed_mean <= 120:
        vehicle_speed_verdict = (
            "Likely km/h (WRONG - needs conversion to m/s, divide by 3.6)"
        )
    else:
        vehicle_speed_verdict = "Unit unclear from the configured thresholds"
    print(f"\n!!! VEHICLE_SPEED VERDICT: {vehicle_speed_verdict} !!!\n")

    print_statistics("gyro_z", gyro_z)
    max_abs_gyro_z = gyro_z.abs().max()
    if max_abs_gyro_z < 5:
        gyro_z_verdict = "Likely radians/second (correct)"
    elif max_abs_gyro_z > 20:
        gyro_z_verdict = (
            "Likely degrees/second (WRONG - needs conversion to rad/s, "
            "multiply by pi/180)"
        )
    else:
        gyro_z_verdict = "Unit unclear from the configured thresholds"
    print(f"\n!!! GYRO_Z VERDICT: {gyro_z_verdict} !!!")


if __name__ == "__main__":
    main()