"""Convert the prepared driving segment vehicle speed from km/h to m/s."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def print_statistics(label: str, values: pd.Series) -> None:
    print(
        f"{label}: min={values.min():.6f}, "
        f"max={values.max():.6f}, mean={values.mean():.6f}"
    )


def main() -> None:
    input_path = Path(__file__).parent / "driving_segment.csv"
    frame = pd.read_csv(input_path)
    before = frame["vehicle_speed"]
    print_statistics("Before conversion (km/h)", before)

    frame["vehicle_speed"] = before / 3.6
    frame.to_csv(input_path, index=False)

    print_statistics("After conversion (m/s)", frame["vehicle_speed"])
    print(f"Updated: {input_path}")


if __name__ == "__main__":
    main()