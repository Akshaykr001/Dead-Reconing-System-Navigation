"""Extended Kalman filter for GNSS-denied vehicle dead reckoning.

Input CSV columns:
    timestamp, gyro_z, mag_heading, gps_lat, gps_lon,
    vehicle_speed, gnss_available

Angles in the input are degrees/seconds unless documented otherwise:
    gyro_z is radians/second; mag_heading is degrees clockwise from north;
    vehicle_speed is metres/second; GPS is decimal degrees.

The EKF state is [latitude, longitude, heading, velocity, gyro_bias].
Latitude/longitude are stored in a local tangent-plane coordinate system in
metres internally. This is equivalent to the requested geographic state while
keeping the covariance numerically well-conditioned. The output converts the
position back to decimal degrees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLUMNS = [
    "timestamp",
    "gyro_z",
    "mag_heading",
    "gps_lat",
    "gps_lon",
    "vehicle_speed",
    "gnss_available",
]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi), avoiding discontinuities at north."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def gps_to_local(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert decimal-degree GPS positions to a local east/north frame."""
    lat_scale = np.deg2rad(1.0) * EARTH_RADIUS_M
    lon_scale = lat_scale * np.cos(np.deg2rad(lat0))
    north = (lat_deg - lat0) * lat_scale
    east = (lon_deg - lon0) * lon_scale
    return east, north


def local_to_gps(east: float, north: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert one local east/north position back to decimal degrees."""
    lat_scale = np.deg2rad(1.0) * EARTH_RADIUS_M
    lon_scale = lat_scale * np.cos(np.deg2rad(lat0))
    return lat0 + north / lat_scale, lon0 + east / lon_scale


def bearing_from_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the clockwise-from-north bearing between two GPS points."""
    phi1, phi2 = np.deg2rad([lat1, lat2])
    delta_lon = np.deg2rad(lon2 - lon1)
    y = np.sin(delta_lon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lon)
    return wrap_angle(float(np.arctan2(y, x)))


def load_input(path: Path) -> pd.DataFrame:
    """Load, validate, and normalize the driving segment."""
    data = pd.read_csv(path)
    # Person 1's export contains the full smartphone IMU and GPS stream but
    # does not yet have an outage flag. Treat finite GPS rows as available so
    # the real export can be processed directly; outage simulation can later
    # replace this column with an intentional mask.
    if "mag_heading" not in data.columns and {"mag_x", "mag_y"}.issubset(data.columns):
        data["mag_heading"] = np.rad2deg(np.arctan2(data["mag_x"], data["mag_y"])) % 360.0
    if "gnss_available" not in data.columns:
        data["gnss_available"] = data[["gps_lat", "gps_lon"]].notna().all(axis=1)

    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    data["gnss_available"] = data["gnss_available"].astype(bool)
    for column in REQUIRED_COLUMNS[1:-1]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    sensor_columns = ["gyro_z", "mag_heading", "vehicle_speed"]
    if data[sensor_columns].isna().any().any():
        raise ValueError("The EKF input contains missing or non-numeric sensor values.")
    gps_missing_while_available = data["gnss_available"] & data[["gps_lat", "gps_lon"]].isna().any(axis=1)
    if gps_missing_while_available.any():
        raise ValueError("The EKF input contains missing GPS values on GNSS-available rows.")
    if data[["gps_lat", "gps_lon"]].notna().sum().min() == 0:
        raise ValueError("The EKF input needs at least one valid GPS position.")
    return data


def propagate_covariance(
    covariance: np.ndarray,
    state: np.ndarray,
    speed: float,
    dt: float,
    gyro_noise_std: float,
    speed_noise_std: float,
    bias_walk_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the state covariance with the nonlinear motion model.

    Motion model, using heading psi measured clockwise from north:
        east'  = east  + v sin(psi) dt
        north' = north + v cos(psi) dt
        psi'   = psi + (gyro_z - bias) dt
        v'     = measured vehicle speed
        b'     = b

    F is the Jacobian of that model. Q maps gyro, speed, and bias-walk noise
    into the state. The covariance prediction is P' = F P F^T + Q.
    """
    _, _, heading, _, _ = state
    sin_heading = np.sin(heading)
    cos_heading = np.cos(heading)

    # Full state-transition Jacobian for [east, north, heading, speed, bias].
    transition = np.eye(5)
    transition[0, 2] = speed * cos_heading * dt
    transition[0, 3] = sin_heading * dt
    transition[1, 2] = -speed * sin_heading * dt
    transition[1, 3] = cos_heading * dt
    transition[2, 4] = -dt

    # Discrete process noise for gyro rate, speed input, and bias random walk.
    q_gyro = (gyro_noise_std * dt) ** 2
    q_speed = speed_noise_std**2
    q_bias = (bias_walk_std * np.sqrt(max(dt, 1e-9))) ** 2
    process_noise = np.diag([0.0, 0.0, q_gyro, q_speed, q_bias])

    return transition @ covariance @ transition.T + process_noise, transition


def update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_noise: np.ndarray,
    angle_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the standard EKF measurement update.

    Innovation: y = z - h(x)
    Gain:      K = P H^T (H P H^T + R)^-1
    State:     x = x + K y
    Covariance uses the Joseph form, which is algebraically equivalent to
    (I-KH)P but is more stable and preserves positive semidefiniteness.
    """
    innovation = measurement - measurement_matrix @ state
    if angle_index is not None:
        innovation[angle_index] = wrap_angle(innovation[angle_index])

    innovation_covariance = measurement_matrix @ covariance @ measurement_matrix.T + measurement_noise
    kalman_gain = np.linalg.solve(innovation_covariance, measurement_matrix @ covariance).T
    state = state + kalman_gain @ innovation
    if angle_index is not None:
        state[angle_index] = wrap_angle(state[angle_index])

    identity = np.eye(len(state))
    residual_matrix = identity - kalman_gain @ measurement_matrix
    covariance = (
        residual_matrix @ covariance @ residual_matrix.T
        + kalman_gain @ measurement_noise @ kalman_gain.T
    )
    return state, (covariance + covariance.T) / 2.0


def run_ekf(
    data: pd.DataFrame,
    bias_predictor: Callable[[pd.DataFrame], float] | None = None,
    bias_history_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run EKF and an uncorrected gyro+speed dead-reckoning trajectory."""
    lat0 = float(data.loc[0, "gps_lat"])
    lon0 = float(data.loc[0, "gps_lon"])
    gps_east, gps_north = gps_to_local(data["gps_lat"].to_numpy(), data["gps_lon"].to_numpy(), lat0, lon0)

    first_heading = bearing_from_points(
        data.loc[0, "gps_lat"], data.loc[0, "gps_lon"],
        data.loc[min(1, len(data) - 1), "gps_lat"], data.loc[min(1, len(data) - 1), "gps_lon"],
    )
    # State is [east_m, north_m, heading_rad, velocity_mps, gyro_bias_radps].
    state = np.array([gps_east[0], gps_north[0], first_heading, data.loc[0, "vehicle_speed"], 0.0])
    covariance = np.diag([4.0**2, 4.0**2, np.deg2rad(15.0) ** 2, 2.0**2, np.deg2rad(1.0) ** 2])
    raw_state = state.copy()
    rows: list[dict[str, object]] = []
    gps_error_rows: list[dict[str, float | bool]] = []

    # Tunable sensor-noise assumptions; replace with sensor specs during EKF tuning.
    gps_noise = np.diag([5.0**2, 5.0**2])
    mag_noise = np.array([[np.deg2rad(12.0) ** 2]])
    gyro_noise_std = np.deg2rad(1.5)
    speed_noise_std = 0.5
    bias_walk_std = np.deg2rad(0.03)

    timestamps = data["timestamp"].tolist()
    for index, row in data.iterrows():
        dt = 0.1 if index == 0 else max((timestamps[index] - timestamps[index - 1]).total_seconds(), 1e-3)
        gyro_z = float(row["gyro_z"])
        speed = float(row["vehicle_speed"])

        # Predict using the EKF bias state. An optional learned correction is
        # a heading correction over its recent history window, so convert it
        # to a rate before applying it at this timestep.
        learned_rate_correction = 0.0
        if bias_predictor is not None:
            history_start = index
            while history_start > 0 and (
                timestamps[index] - timestamps[history_start - 1]
            ).total_seconds() <= bias_history_s:
                history_start -= 1
            history_window = data.iloc[history_start : index + 1]
            correction = float(bias_predictor(history_window))
            history_duration = max(
                (timestamps[index] - timestamps[history_start]).total_seconds(), dt
            )
            learned_rate_correction = correction / history_duration
        heading_rate = gyro_z - state[4] + learned_rate_correction
        state[0] += speed * np.sin(state[2]) * dt
        state[1] += speed * np.cos(state[2]) * dt
        state[2] = wrap_angle(state[2] + heading_rate * dt)
        state[3] = speed
        covariance, _ = propagate_covariance(
            covariance, state, speed, dt, gyro_noise_std, speed_noise_std, bias_walk_std
        )

        # Magnetometer is always a heading correction, including during outage.
        mag_heading = np.deg2rad(float(row["mag_heading"]))
        mag_matrix = np.zeros((1, 5))
        mag_matrix[0, 2] = 1.0
        state, covariance = update(state, covariance, np.array([mag_heading]), mag_matrix, mag_noise, 0)

        # GPS is a position correction only when the availability flag is true.
        if bool(row["gnss_available"]):
            gps_matrix = np.zeros((2, 5))
            gps_matrix[0, 0] = 1.0
            gps_matrix[1, 1] = 1.0
            state, covariance = update(
                state,
                covariance,
                np.array([gps_east[index], gps_north[index]]),
                gps_matrix,
                gps_noise,
            )

        # Raw dead reckoning uses the same gyro and speed but never corrects.
        raw_state[0] += speed * np.sin(raw_state[2]) * dt
        raw_state[1] += speed * np.cos(raw_state[2]) * dt
        raw_state[2] = wrap_angle(raw_state[2] + gyro_z * dt)
        raw_state[3] = speed
        estimated_lat, estimated_lon = local_to_gps(state[0], state[1], lat0, lon0)
        raw_lat, raw_lon = local_to_gps(raw_state[0], raw_state[1], lat0, lon0)
        position_uncertainty = float(np.sqrt(max(np.trace(covariance[:2, :2]), 0.0)))

        rows.append({
            "timestamp": row["timestamp"],
            "estimated_lat": estimated_lat,
            "estimated_lon": estimated_lon,
            "estimated_heading": np.rad2deg(state[2]) % 360.0,
            "position_uncertainty": position_uncertainty,
            "gnss_available": bool(row["gnss_available"]),
            "raw_dr_lat": raw_lat,
            "raw_dr_lon": raw_lon,
        })
        gps_error_rows.append({
            "ekf_error_m": float(np.hypot(state[0] - gps_east[index], state[1] - gps_north[index])),
            "raw_error_m": float(np.hypot(raw_state[0] - gps_east[index], raw_state[1] - gps_north[index])),
            "gnss_available": bool(row["gnss_available"]),
        })

    return pd.DataFrame(rows), pd.DataFrame(gps_error_rows)


def plot_paths(data: pd.DataFrame, estimates: pd.DataFrame, output_path: Path) -> None:
    """Plot ground truth, uncorrected dead reckoning, and EKF paths together."""
    plt.figure(figsize=(9, 7))
    plt.plot(data["gps_lon"], data["gps_lat"], label="Ground truth GPS", linewidth=2)
    plt.plot(estimates["raw_dr_lon"], estimates["raw_dr_lat"], label="Raw dead reckoning", alpha=0.8)
    plt.plot(estimates["estimated_lon"], estimates["estimated_lat"], label="EKF", linewidth=2)
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title("GNSS-denied vehicle dead reckoning")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", type=Path, default=Path("driving_segment.csv"))
    parser.add_argument("--output", type=Path, default=Path("ekf_output.csv"))
    parser.add_argument("--plot", type=Path, default=Path("ekf_paths.png"))
    parser.add_argument("--bias-model", type=Path, default=None)
    parser.add_argument("--bias-history-s", type=float, default=1.0)
    args = parser.parse_args()

    data = load_input(args.input_csv)
    predictor = None
    if args.bias_model is not None:
        from gyro_bias_model import predict_bias

        predictor = lambda window: predict_bias(window, args.bias_model)
    estimates, errors = run_ekf(data, predictor, args.bias_history_s)
    output_columns = [
        "timestamp", "estimated_lat", "estimated_lon", "estimated_heading",
        "position_uncertainty", "gnss_available",
    ]
    estimates[output_columns].to_csv(args.output, index=False)
    plot_paths(data, estimates, args.plot)

    outage = errors.loc[~errors["gnss_available"], "ekf_error_m"].dropna()
    available = errors.loc[errors["gnss_available"], "ekf_error_m"].dropna()
    print(f"Wrote {args.output} ({len(estimates)} rows)")
    print(f"Wrote {args.plot}")
    print(f"Mean EKF position error during GNSS outage: {outage.mean():.2f} m")
    print(f"Mean EKF position error with GNSS available: {available.mean():.2f} m")


if __name__ == "__main__":
    main()
