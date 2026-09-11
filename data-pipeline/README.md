# IO-VNBD vehicle dead-reckoning pipeline

This small pipeline prepares one synchronized 10 Hz smartphone/vehicle segment from the [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) dataset. It keeps the original sensor units: gyroscope in rad/s, magnetometer in microtesla, GPS in degrees, and speed in km/h.

## Setup

Install the minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and extract the synchronized archive (it is large):

```bash
python download_dataset.py
```

Then point the preprocessor at matching `S-*.csv` and `V-*.csv` files. The archive contains paths with spaces, so quoting them is useful on Windows:

```bash
python preprocess_iovnbd.py --smartphone "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Vw1.csv" --vehicle "data/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-Vw1.csv" --duration-minutes 10 --output driving_segment.csv
```

The output is `driving_segment.csv`, with a regular 10 Hz timestamp, the requested IMU/GPS/speed columns, and `heading` in degrees clockwise from north. Small gaps up to 0.5 seconds are time-interpolated; rows with larger or unresolved gaps are removed. The final GPS point uses the preceding valid bearing because a consecutive point is required.

Create a synthetic GNSS outage by masking GPS for 75 seconds starting 2 minutes into the segment:

```bash
python simulate_gnss_outage.py driving_segment.csv --start-seconds 120 --duration-seconds 75
```

Inspect the route and basic statistics:

```bash
python inspect_data.py driving_segment.csv
```

This writes `gps_route.png` and prints duration, haversine distance, speed range, row count, and missing-value count.

## Complete project run

From the repository root in PowerShell, activate the existing virtual environment and run:

```powershell
cd C:\path\to\Hackathon_Project
.venv\Scripts\activate
python data-pipeline\run_pipeline.py
```

The runner masks a 75-second GNSS outage, trains the existing Random Forest gyro-bias model,
runs the EKF with that correction, and writes the validated replay to
`hackathon\hackathon\vehicle-dashboard\public\data\navigation_replay.json`.

Start the dashboard in a second terminal:

```powershell
cd C:\path\to\Hackathon_Project\hackathon\hackathon\vehicle-dashboard
npm install
npm run dev
```

The current project has no map-matching implementation. The exporter therefore sets
`road_match_confidence` to the documented unavailable value `0.0`; it does not fabricate road data.
