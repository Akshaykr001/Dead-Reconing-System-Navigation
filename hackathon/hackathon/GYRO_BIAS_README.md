# Explainable gyro-bias model

`gyro_bias_model.py` trains a small `RandomForestRegressor` to estimate the short-window heading correction needed by the EKF.

## Target

For each row and a causal history window (default 1 second):

```text
bias_correction = GPS heading change - sum(raw gyro_z * dt)
```

The correction is in radians over the history window. It is added to the gyro-implied heading change. This sign is intentional: if the raw gyro turns too far, the learned correction is negative.

## Features

- `gyro_variance`: recent gyro noise level; unstable sensors usually need more correction.
- `vehicle_speed`: motion regime affects vibration and gyro quality.
- `gyro_turn_rate`: current rotational motion, which distinguishes turns from straight driving.
- `time_since_gps_s`: correction confidence can change as the last GNSS update becomes stale.

The model is trained only on GNSS-available rows. The artificial outage is held out for evaluation.

## Run

Install the extra lightweight ML dependency:

```powershell
python -m pip install scikit-learn joblib
```

Train and evaluate a 60-second outage:

```powershell
python gyro_bias_model.py driving_segment.csv --outage-duration 60
```

This writes `gyro_bias_model.joblib` and prints naive versus learned mean position error and percentage improvement.

## EKF handoff

The teammate's EKF can call:

```python
from gyro_bias_model import predict_bias

correction_rad = predict_bias(recent_sensor_window, "gyro_bias_model.joblib")
```

`recent_sensor_window` is a pandas DataFrame containing recent `timestamp`, `gyro_z`, `vehicle_speed`, and optionally `gnss_available`. Apply `correction_rad` to the raw gyro heading change for the same short history interval. Do not apply it repeatedly as a full one-second correction without scaling it to the EKF timestep.

This is intentionally a small, inspectable model rather than a neural network. The saved artifact also contains validation error, the naive median baseline, feature names, and the feature importances.
