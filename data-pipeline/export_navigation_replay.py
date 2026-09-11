"""Export real segment and EKF output to the vehicle dashboard contract."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

REQUIRED_OUTPUT_COLUMNS = {
    "timestamp",
    "ground_truth_lat",
    "ground_truth_lon",
    "raw_imu_lat",
    "raw_imu_lon",
    "fused_lat",
    "fused_lon",
    "fused_heading",
    "fused_velocity",
    "position_uncertainty",
    "gnss_available",
    "road_match_confidence",
}


def _finite(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} contains NaN or infinity")
    return number


def build_replay(segment_path: Path, ekf_path: Path) -> list[dict[str, object]]:
    segment = pd.read_csv(segment_path, parse_dates=["timestamp"])
    estimates = pd.read_csv(ekf_path, parse_dates=["timestamp"])
    required_segment = {"timestamp", "gps_lat", "gps_lon", "vehicle_speed"}
    missing = sorted(required_segment - set(segment.columns))
    if missing:
        raise ValueError(f"Segment is missing required columns: {', '.join(missing)}")
    required_ekf = {
        "timestamp", "estimated_lat", "estimated_lon", "estimated_heading",
        "position_uncertainty", "gnss_available", "raw_dr_lat", "raw_dr_lon",
    }
    missing = sorted(required_ekf - set(estimates.columns))
    if missing:
        raise ValueError(f"EKF output is missing required columns: {', '.join(missing)}")

    merged = segment.merge(estimates, on="timestamp", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No timestamps overlap between the segment and EKF output")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # No map matcher exists in this project. Zero is an explicit, deterministic
    # unavailable state; it is not presented as a road-match result.
    merged["road_match_confidence"] = 0.0
    # The prepared IO-VNBD segment stores speed in km/h; the dashboard and EKF
    # contract use m/s.
    merged["fused_velocity"] = pd.to_numeric(merged["vehicle_speed"], errors="coerce") / 3.6
    output: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        record = {
            "timestamp": row.timestamp.isoformat(),
            "ground_truth_lat": _finite(row.gps_lat, "ground_truth_lat"),
            "ground_truth_lon": _finite(row.gps_lon, "ground_truth_lon"),
            "raw_imu_lat": _finite(row.raw_dr_lat, "raw_imu_lat"),
            "raw_imu_lon": _finite(row.raw_dr_lon, "raw_imu_lon"),
            "fused_lat": _finite(row.estimated_lat, "fused_lat"),
            "fused_lon": _finite(row.estimated_lon, "fused_lon"),
            "fused_heading": _finite(row.estimated_heading, "fused_heading"),
            "fused_velocity": _finite(row.fused_velocity, "fused_velocity"),
            "position_uncertainty": _finite(row.position_uncertainty, "position_uncertainty"),
            "gnss_available": bool(row.gnss_available),
            "road_match_confidence": 0.0,
        }
        output.append(record)

    if len(output) != len(merged) or not REQUIRED_OUTPUT_COLUMNS.issubset(output[0]):
        raise ValueError("Replay export did not produce the complete dashboard contract")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--ekf-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_replay(args.segment, args.ekf_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outage_frames = sum(not row["gnss_available"] for row in rows)
    print("Replay export successful")
    print(f"Frames: {len(rows):,}")
    print(f"GNSS outage frames: {outage_frames:,}")
    print(f"First timestamp: {rows[0]['timestamp']}")
    print(f"Last timestamp: {rows[-1]['timestamp']}")
    print(f"Output: {args.output}")
    print("Road match confidence: unavailable (no map-matching implementation found)")


if __name__ == "__main__":
    main()
