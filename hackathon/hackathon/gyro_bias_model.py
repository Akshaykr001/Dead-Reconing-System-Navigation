"""Train and evaluate a small, explainable gyro-drift model.

The target is a heading correction in radians over a short window:
    correction = true_heading_change - sum(raw_gyro_z * dt)

This sign convention is useful to the EKF: add the predicted correction to the
raw gyro heading change. It is not a deep-learning model; a small random
forest is fast to train and its feature importance is easy to explain.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_NAMES = [
    "gyro_variance",
    "vehicle_speed",
    "gyro_turn_rate",
    "time_since_gps_s",
]
EARTH_RADIUS_M = 6_371_000.0


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def bearing_radians(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate clockwise-from-north GPS bearings between consecutive points."""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    delta_lon = np.deg2rad(lon2 - lon1)
    y = np.sin(delta_lon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lon)
    return np.asarray(wrap_angle(np.arctan2(y, x)))


def load_data(path: Path) -> pd.DataFrame:
    """Load the Person 1 export and choose its GPS heading source."""
    data = pd.read_csv(path)
    required = {"timestamp", "gyro_z", "gps_lat", "gps_lon", "vehicle_speed"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    numeric_columns = ["gyro_z", "gps_lat", "gps_lon", "vehicle_speed"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=numeric_columns).reset_index(drop=True)

    # Prefer Person 1's GPS-derived heading column when present. Otherwise
    # calculate bearings directly from the GPS coordinates.
    if "heading" in data.columns:
        data["truth_heading_rad"] = np.unwrap(np.deg2rad(data["heading"].to_numpy()))
    else:
        lat = data["gps_lat"].to_numpy()
        lon = data["gps_lon"].to_numpy()
        bearings = np.zeros(len(data))
        if len(data) > 1:
            bearings[1:] = bearing_radians(lat[:-1], lon[:-1], lat[1:], lon[1:])
            bearings[0] = bearings[1]
        data["truth_heading_rad"] = np.unwrap(bearings)

    if "gnss_available" not in data.columns:
        data["gnss_available"] = True
    else:
        data["gnss_available"] = data["gnss_available"].astype(bool)
    return data


def add_outage_flag(data: pd.DataFrame, start_s: float | None, duration_s: float) -> pd.DataFrame:
    """Mask GPS availability for a chosen interval without deleting truth GPS."""
    result = data.copy()
    elapsed = (result["timestamp"] - result["timestamp"].iloc[0]).dt.total_seconds()
    if start_s is None:
        start_s = max((float(elapsed.iloc[-1]) - duration_s) / 2.0, 0.0)
    result["gnss_available"] = ~elapsed.between(start_s, start_s + duration_s)
    return result


def build_training_table(data: pd.DataFrame, history_s: float = 1.0) -> pd.DataFrame:
    """Build causal features and the measured short-window gyro correction."""
    rows: list[dict[str, float]] = []
    timestamps = data["timestamp"].tolist()
    truth = data["truth_heading_rad"].to_numpy()
    gyro = data["gyro_z"].to_numpy()
    speed = data["vehicle_speed"].to_numpy()
    gps_available = data["gnss_available"].to_numpy(dtype=bool)
    last_gps_time = timestamps[0]

    for index in range(1, len(data)):
        now = timestamps[index]
        if gps_available[index]:
            last_gps_time = now
        start = index
        while start > 0 and (now - timestamps[start - 1]).total_seconds() <= history_s:
            start -= 1
        dt = np.diff(np.array([timestamp.value for timestamp in timestamps[start : index + 1]], dtype=float)) / 1e9
        raw_change = float(np.sum(gyro[start:index] * dt))
        truth_change = float(wrap_angle(truth[index] - truth[start]))
        window_gyro = gyro[start : index + 1]
        rows.append({
            "gyro_variance": float(np.var(window_gyro)),
            "vehicle_speed": float(speed[index]),
            "gyro_turn_rate": float(gyro[index]),
            "time_since_gps_s": max((now - last_gps_time).total_seconds(), 0.0),
            "bias_target": truth_change - raw_change,
            "timestamp": now,
            "index": float(index),
        })
    return pd.DataFrame(rows)


def train_model(training_table: pd.DataFrame, output_path: Path, random_state: int = 7) -> dict[str, object]:
    """Fit a small random forest and save model plus feature metadata."""
    if len(training_table) < 10:
        raise ValueError("At least 10 usable rows are needed to train the bias model.")
    split = max(int(len(training_table) * 0.8), 1)
    features = training_table[FEATURE_NAMES]
    targets = training_table["bias_target"]
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=4,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features.iloc[:split], targets.iloc[:split])
    validation_prediction = model.predict(features.iloc[split:]) if split < len(features) else np.array([])
    validation_target = targets.iloc[split:].to_numpy()
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "target_definition": "true heading change minus raw gyro_z integrated over history_s",
        "history_s": float(training_table.attrs.get("history_s", 1.0)),
        "naive_bias": float(targets.iloc[:split].median()),
        "validation_mae": float(mean_absolute_error(validation_target, validation_prediction)) if len(validation_target) else None,
        "validation_rmse": float(np.sqrt(mean_squared_error(validation_target, validation_prediction))) if len(validation_target) else None,
    }
    joblib.dump(artifact, output_path)
    return artifact


def predict_bias(recent_sensor_window: pd.DataFrame, model_path: str | Path = "gyro_bias_model.joblib") -> float:
    """Predict one heading correction in radians for the EKF.

    `recent_sensor_window` should contain recent `gyro_z`, `vehicle_speed`,
    `timestamp`, and optionally `gnss_available` columns. The returned value
    is added to the raw gyro heading change for the next short window.
    """
    artifact = joblib.load(model_path)
    window = recent_sensor_window.copy()
    if len(window) == 0:
        return float(artifact.get("naive_bias", 0.0))
    timestamps = pd.to_datetime(window["timestamp"])
    gyro = pd.to_numeric(window["gyro_z"]).to_numpy()
    speed = float(pd.to_numeric(window["vehicle_speed"]).iloc[-1])
    available = window.get("gnss_available", pd.Series(True, index=window.index)).astype(bool)
    last_gps = timestamps.iloc[0]
    for timestamp, is_available in zip(timestamps, available):
        if is_available:
            last_gps = timestamp
    feature_row = pd.DataFrame([{
        "gyro_variance": float(np.var(gyro)),
        "vehicle_speed": speed,
        "gyro_turn_rate": float(gyro[-1]),
        "time_since_gps_s": max((timestamps.iloc[-1] - last_gps).total_seconds(), 0.0),
    }], columns=artifact["feature_names"])
    return float(artifact["model"].predict(feature_row)[0])


def local_xy(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert GPS to local east/north metres for error calculation."""
    lat0, lon0 = data.loc[0, ["gps_lat", "gps_lon"]]
    lat_scale = np.deg2rad(1.0) * EARTH_RADIUS_M
    lon_scale = lat_scale * np.cos(np.deg2rad(lat0))
    east = (data["gps_lon"].to_numpy() - lon0) * lon_scale
    north = (data["gps_lat"].to_numpy() - lat0) * lat_scale
    return east, north


def evaluate_outage(data: pd.DataFrame, artifact: dict[str, object], history_s: float) -> dict[str, float]:
    """Compare fixed-bias and learned-bias runs of the actual EKF."""
    from ekf_dead_reckoning import load_input, run_ekf

    # Reuse the EKF's normalizer so raw magnetometer-axis exports are handled
    # exactly the same way as a normal filter run.
    ekf_data = load_input(Path(data.attrs["input_path"]))
    ekf_data["gnss_available"] = data["gnss_available"].to_numpy(dtype=bool)
    _, baseline_errors = run_ekf(ekf_data)

    model = artifact["model"]
    def learned_predictor(window: pd.DataFrame) -> float:
        timestamps = pd.to_datetime(window["timestamp"])
        available = window["gnss_available"].astype(bool)
        last_gps = timestamps.iloc[0]
        for timestamp, is_available in zip(timestamps, available):
            if is_available:
                last_gps = timestamp
        features = pd.DataFrame([{
            "gyro_variance": float(np.var(window["gyro_z"])),
            "vehicle_speed": float(window["vehicle_speed"].iloc[-1]),
            "gyro_turn_rate": float(window["gyro_z"].iloc[-1]),
            "time_since_gps_s": max((timestamps.iloc[-1] - last_gps).total_seconds(), 0.0),
        }])
        return float(model.predict(features[FEATURE_NAMES])[0])

    # The fixed baseline uses the EKF's normal bias state; the learned run
    # receives the model correction through the same EKF prediction path.
    _, learned_errors = run_ekf(ekf_data, learned_predictor, history_s)
    outage = ~baseline_errors["gnss_available"]
    if not outage.any():
        raise ValueError("No GNSS outage rows found. Pass --outage-duration to create one.")
    naive_error = float(baseline_errors.loc[outage, "ekf_error_m"].mean())
    learned_error = float(learned_errors.loc[outage, "ekf_error_m"].mean())
    return {
        "naive_mean_error_m": naive_error,
        "learned_mean_error_m": learned_error,
        "improvement_percent": (naive_error - learned_error) / naive_error * 100.0 if naive_error else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, nargs="?", default=Path("driving_segment.csv"))
    parser.add_argument("--model", type=Path, default=Path("gyro_bias_model.joblib"))
    parser.add_argument("--history-s", type=float, default=1.0)
    parser.add_argument("--outage-start", type=float, default=None, help="Outage start in seconds from the first row")
    parser.add_argument("--outage-duration", type=float, default=60.0)
    args = parser.parse_args()

    data = load_data(args.input_csv)
    data = add_outage_flag(data, args.outage_start, args.outage_duration)
    data.attrs["input_path"] = str(args.input_csv)
    # Keep the artificial outage out of training so the comparison measures
    # generalization into a denied-GNSS interval rather than memorization.
    training_data = data.loc[data["gnss_available"]].reset_index(drop=True)
    if len(training_data) < 10:
        raise ValueError("Not enough GNSS-available rows remain for training; shorten the outage.")
    training_table = build_training_table(training_data, args.history_s)
    training_table.attrs["history_s"] = args.history_s
    artifact = train_model(training_table, args.model)
    metrics = evaluate_outage(data, artifact, args.history_s)
    print(f"Saved model: {args.model}")
    print(f"Validation MAE: {artifact['validation_mae']}")
    print(f"Naive outage mean position error: {metrics['naive_mean_error_m']:.2f} m")
    print(f"Learned outage mean position error: {metrics['learned_mean_error_m']:.2f} m")
    print(f"Position-error improvement: {metrics['improvement_percent']:.1f}%")


if __name__ == "__main__":
    main()
    print(f"Feature importances: {dict(zip(FEATURE_NAMES, artifact['model'].feature_importances_))}")


if __name__ == "__main__":
    main()
