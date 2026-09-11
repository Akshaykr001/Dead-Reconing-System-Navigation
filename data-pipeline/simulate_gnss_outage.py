"""Mask GPS during a configurable outage window for EKF testing."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("driving_segment.csv"))
    parser.add_argument("--output", type=Path, default=Path("driving_segment_gnss_outage.csv"))
    parser.add_argument("--start-seconds", type=float, default=120.0)
    parser.add_argument("--duration-seconds", type=float, default=75.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, parse_dates=["timestamp"])
    frame["gnss_available"] = True
    elapsed = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds()
    outage = elapsed.between(args.start_seconds, args.start_seconds + args.duration_seconds, inclusive="left")
    frame.loc[outage, ["gps_lat", "gps_lon"]] = pd.NA
    frame.loc[outage, "gnss_available"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Masked {int(outage.sum()):,} rows; wrote {args.output}")


if __name__ == "__main__":
    main()
