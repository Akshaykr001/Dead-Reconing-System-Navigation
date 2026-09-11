# Vehicle Dead-Reckoning EKF

This folder contains a standalone Extended Kalman Filter for the CSV exported by Person 1.

## Install

```powershell
python -m pip install numpy pandas matplotlib
```

## Run

```powershell
python ekf_dead_reckoning.py driving_segment.csv
```

Outputs:

- `ekf_output.csv`: timestamp, EKF latitude/longitude, heading, position uncertainty, and GNSS availability.
- `ekf_paths.png`: ground-truth GPS, raw gyro+speed dead reckoning, and EKF paths.

The included `driving_segment.csv` is a small synthetic fixture with an 8-second GNSS outage. Person 1's export may instead contain the raw IMU axes and a `heading` column:

```text
timestamp,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,gps_lat,gps_lon,vehicle_speed,heading
```

The EKF loader accepts both forms. For Person 1's form it derives `mag_heading` from the magnetometer axes and marks finite GPS rows as GNSS available. To copy and run the real export from the Downloads folder in PowerShell:

```powershell
Copy-Item "$env:USERPROFILE\Downloads\driving_segment.csv" .\driving_segment.csv -Force
python .\ekf_dead_reckoning.py .\driving_segment.csv
```

`gyro_z` must be radians/second, magnetometer values must be in a consistent frame, and `vehicle_speed` must be metres/second. The script prints mean position error separately during outage rows and GNSS-available rows.

## Learned gyro drift

`gyro_bias_model.py` trains the lightweight Random Forest described in [GYRO_BIAS_README.md](GYRO_BIAS_README.md):

```powershell
python -m pip install scikit-learn joblib
python .\gyro_bias_model.py .\driving_segment.csv --outage-duration 60
python .\ekf_dead_reckoning.py .\driving_segment.csv --bias-model .\gyro_bias_model.joblib
```

The first command trains and evaluates fixed versus learned bias on the masked interval. The second feeds the saved model into the EKF.
