"""Prepare one 5-15 minute IO-VNBD driving segment for an EKF."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = [
    "timestamp", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z",
    "gps_lat", "gps_lon", "vehicle_speed", "heading",
]


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(frame: pd.DataFrame, names: Iterable[str]) -> str:
    aliases = {_normalise_name(name) for name in names}
    for column in frame.columns:
        if _normalise_name(column) in aliases:
            return column
    raise ValueError(f"Could not find one of {list(names)} in columns: {list(frame.columns)}")


def _read_csv(path: Path) -> pd.DataFrame:
    # Older IO-VNBD exports contain degree/micro symbols in a Windows-1252 header.
    return pd.read_csv(path, encoding="cp1252")


def _parse_phone_time(values: pd.Series) -> pd.Series:
    # IO-VNBD writes milliseconds after the seconds with a colon, e.g. ...55:816.
    cleaned = values.astype(str).str.replace(r"(?<=\d{2}):(\d{3})$", r".\1", regex=True)
    parsed = pd.to_datetime(cleaned, errors="coerce").astype("datetime64[ns]")
    if parsed.isna().all():
        raise ValueError("No smartphone timestamps could be parsed")
    return parsed


def _load_smartphone(path: Path) -> pd.DataFrame:
    raw = _read_csv(path)
    timestamp = _find_column(raw, ["DATE (YYYY-MO-DD HH-MI-SS_SSS)", "timestamp", "time"])
    result = pd.DataFrame({"timestamp": _parse_phone_time(raw[timestamp])})
    mappings = {
        "gyro_x": ["GYROSCOPE X (rad/s)", "gyro_x", "gyroscope_x"],
        "gyro_y": ["GYROSCOPE Y (rad/s)", "gyro_y", "gyroscope_y"],
        "gyro_z": ["GYROSCOPE Z (rad/s)", "gyro_z", "gyroscope_z"],
        "mag_x": ["MAGNETIC FIELD X (μT)", "MAGNETIC FIELD X (Î¼T)", "mag_x", "magnetic_x"],
        "mag_y": ["MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Y (Î¼T)", "mag_y", "magnetic_y"],
        "mag_z": ["MAGNETIC FIELD Z (μT)", "MAGNETIC FIELD Z (Î¼T)", "mag_z", "magnetic_z"],
        "gps_lat": ["GPS LATITUDE (degrees)", "latitude", "gps_lat", "lat"],
        "gps_lon": ["GPS LONGITUDE (degrees)", "longitude", "gps_lon", "lon", "lng"],
        "phone_speed": ["GPS SPEED (Kmh)", "gps_speed", "speed"],
    }
    for output, aliases in mappings.items():
        result[output] = pd.to_numeric(raw[_find_column(raw, aliases)], errors="coerce")
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp")
    return result.drop_duplicates("timestamp").reset_index(drop=True)


def _load_vehicle(path: Path, phone_start: pd.Timestamp) -> pd.DataFrame:
    raw = _read_csv(path)
    elapsed = _find_column(raw, ["Time Since Start of Day (seconds)", "elapsed_seconds", "time"])
    speed = _find_column(raw, ["Velocity (km/hr)", "vehicle_speed", "speed"])
    elapsed_values = pd.to_numeric(raw[elapsed], errors="coerce")
    result = pd.DataFrame({
        "timestamp": phone_start + pd.to_timedelta(elapsed_values - elapsed_values.min(), unit="s"),
        "vehicle_speed": pd.to_numeric(raw[speed], errors="coerce"),
    })
    result["timestamp"] = result["timestamp"].astype("datetime64[ns]")
    return result.dropna().sort_values("timestamp").drop_duplicates("timestamp")


def _bearing(lat: pd.Series, lon: pd.Series) -> pd.Series:
    lat1 = np.radians(lat)
    lat2 = lat1.shift(-1)
    delta_lon = np.radians(lon.shift(-1) - lon)
    angle = np.degrees(np.arctan2(
        np.sin(delta_lon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon),
    ))
    # The final point has no consecutive point; carry the previous valid heading.
    return (angle + 360.0) % 360.0


def build_segment(smartphone_path: Path, vehicle_path: Path | None, duration_minutes: float,
                  start_seconds: float, max_gap_seconds: float) -> pd.DataFrame:
    phone = _load_smartphone(smartphone_path)
    if vehicle_path:
        vehicle = _load_vehicle(vehicle_path, phone["timestamp"].iloc[0])
        merged = pd.merge_asof(phone.sort_values("timestamp"), vehicle, on="timestamp", direction="nearest", tolerance=pd.Timedelta("150ms"))
    else:
        # This fallback supports smartphone-only files, while preferring vehicle speed when available.
        merged = phone.rename(columns={"phone_speed": "vehicle_speed"})
    merged["vehicle_speed"] = merged["vehicle_speed"].fillna(merged.get("phone_speed", np.nan))
    merged = merged.set_index("timestamp").sort_index()

    start = merged.index[0] + pd.Timedelta(seconds=start_seconds)
    end = start + pd.Timedelta(minutes=duration_minutes)
    segment = merged.loc[start:end].copy()
    if segment.empty or (segment.index[-1] - segment.index[0]).total_seconds() < duration_minutes * 0.9:
        raise ValueError("Requested segment is outside the available recording")

    # Put the recording on its nominal 10 Hz grid, then interpolate only short outages.
    grid = pd.date_range(start=segment.index[0].ceil("100ms"), end=segment.index[-1].floor("100ms"), freq="100ms")
    # Keep the original off-grid samples as interpolation anchors, then select the grid.
    segment = segment.reindex(segment.index.union(grid)).sort_index()
    numeric = [column for column in OUTPUT_COLUMNS if column != "timestamp" and column != "heading"]
    segment[numeric] = segment[numeric].interpolate(method="time", limit=int(max_gap_seconds * 10), limit_area="inside")
    segment = segment.reindex(grid)
    segment = segment.dropna(subset=numeric)
    segment = segment.reset_index(names="timestamp")
    segment["heading"] = _bearing(segment["gps_lat"], segment["gps_lon"]).bfill().ffill()
    segment = segment.dropna(subset=OUTPUT_COLUMNS).sort_values("timestamp")
    if segment.empty:
        raise ValueError("Requested segment has no rows after gap handling; try another file or start time")
    return segment[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smartphone", type=Path, required=True, help="Smartphone S-*.csv file")
    parser.add_argument("--vehicle", type=Path, help="Matching vehicle V-*.csv file")
    parser.add_argument("--output", type=Path, default=Path("driving_segment.csv"))
    parser.add_argument("--duration-minutes", type=float, default=10.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if not 5 <= args.duration_minutes <= 15:
        parser.error("--duration-minutes must be between 5 and 15")
    result = build_segment(args.smartphone, args.vehicle, args.duration_minutes, args.start_seconds, args.max_gap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    print(f"Wrote {len(result):,} rows to {args.output} ({result.timestamp.iloc[0]} to {result.timestamp.iloc[-1]})")


if __name__ == "__main__":
    main()
