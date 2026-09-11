# Dead Reckoning Replay Console

A React + TypeScript + Vite dashboard for replaying the precomputed Python EKF output. It does not calculate navigation, drift, or Kalman updates in the browser.

## Run

```powershell
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Data contract

The app fetches exactly one static file:

```text
public/data/navigation_replay.json
```

It accepts either a JSON array or `{ "rows": [...] }`. Each row must contain:

```json
{
  "timestamp": "2020-01-08T10:15:46.700Z",
  "ground_truth_lat": 52.554695,
  "ground_truth_lon": -1.463064,
  "raw_imu_lat": 52.554695,
  "raw_imu_lon": -1.463064,
  "fused_lat": 52.554695,
  "fused_lon": -1.463064,
  "fused_heading": 0.0,
  "fused_velocity": 0.0,
  "position_uncertainty": 4.0,
  "gnss_available": true,
  "road_match_confidence": 0.0
}
```

The example above documents the schema only. Do not use it as replay data. Copy the real values from the Python pipeline output. Until the JSON exists, the dashboard intentionally shows a clear no-data state and never creates fallback numbers.

## Views

- Leaflet route replay: GPS truth, raw IMU dead reckoning, EKF fused path, moving vehicle marker, uncertainty radius, and outage highlighting.
- Playback: play/pause, scrub, 1x/2x/4x speed, and outage jump.
- Live stats: GNSS state, uncertainty, speed, heading, outage distance, and map-match confidence.
- Recharts error comparison: raw and fused position error against GPS truth.
- Pipeline strip: Gyroscope through Output.

The UI uses dark Carto tiles and loads no backend data. The `public/data` directory is intentionally empty in this workspace because no verified precomputed EKF JSON has been produced yet.
