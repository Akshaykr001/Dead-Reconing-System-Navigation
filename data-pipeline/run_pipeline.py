"""Run the prepared data, ML bias, EKF, and dashboard export flow."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "data-pipeline"
BACKEND_DIR = PROJECT_ROOT / "hackathon" / "hackathon"
SEGMENT_PATH = PIPELINE_DIR / "driving_segment.csv"
OUTAGE_PATH = PIPELINE_DIR / "driving_segment_gnss_outage.csv"
EKF_SEGMENT_PATH = PIPELINE_DIR / "driving_segment_ekf.csv"
EKF_INPUT_PATH = PIPELINE_DIR / "driving_segment_gnss_outage_ekf.csv"
MODEL_PATH = BACKEND_DIR / "gyro_bias_model.joblib"
EKF_OUTPUT_PATH = BACKEND_DIR / "ekf_output_real.csv"
EKF_PLOT_PATH = BACKEND_DIR / "ekf_paths_real.png"
REPLAY_PATH = BACKEND_DIR / "vehicle-dashboard" / "public" / "data" / "navigation_replay.json"


def run_script(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    if not SEGMENT_PATH.exists():
        raise FileNotFoundError(
            f"Prepared segment not found: {SEGMENT_PATH}. Run preprocess_iovnbd.py first."
        )

    segment = pd.read_csv(SEGMENT_PATH, parse_dates=["timestamp"])
    spacing = segment["timestamp"].diff().dropna().dt.total_seconds()
    if spacing.empty or abs(float(spacing.median()) - 0.1) > 0.001:
        raise ValueError("The prepared segment is not a 10 Hz recording")

    print("[1/5] Simulating GNSS outage while preserving truth coordinates")
    run_script(
        PIPELINE_DIR / "simulate_gnss_outage.py",
        str(SEGMENT_PATH),
        "--output", str(OUTAGE_PATH),
        "--start-seconds", "120",
        "--duration-seconds", "75",
    )

    ekf_segment = pd.read_csv(SEGMENT_PATH)
    ekf_segment.to_csv(EKF_SEGMENT_PATH, index=False)
    ekf_input = pd.read_csv(OUTAGE_PATH)
    ekf_input.to_csv(EKF_INPUT_PATH, index=False)

    sys.path.insert(0, str(BACKEND_DIR))
    import ekf_dead_reckoning as ekf
    import gyro_bias_model as bias

    print("[2/5] Training the existing Random Forest gyro-bias model")
    model_data = bias.load_data(EKF_SEGMENT_PATH)
    model_data = bias.add_outage_flag(model_data, 120.0, 75.0)
    model_data.attrs["input_path"] = str(EKF_INPUT_PATH)
    training_data = model_data.loc[model_data["gnss_available"]].reset_index(drop=True)
    training_table = bias.build_training_table(training_data, 1.0)
    training_table.attrs["history_s"] = 1.0
    artifact = bias.train_model(training_table, MODEL_PATH)
    metrics = bias.evaluate_outage(model_data, artifact, 1.0)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Naive outage mean position error: {metrics['naive_mean_error_m']:.2f} m")
    print(f"Learned outage mean position error: {metrics['learned_mean_error_m']:.2f} m")
    print(f"Position-error improvement: {metrics['improvement_percent']:.1f}%")

    print("[3/5] Running the EKF with learned gyro-bias correction")
    outage_data = ekf.load_input(EKF_INPUT_PATH)
    predictor = lambda window: bias.predict_bias(window, MODEL_PATH)
    estimates, _ = ekf.run_ekf(outage_data, predictor, 1.0)
    EKF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(EKF_OUTPUT_PATH, index=False)
    ekf.plot_paths(outage_data, estimates, EKF_PLOT_PATH)
    print(f"Wrote EKF output: {EKF_OUTPUT_PATH} ({len(estimates):,} rows)")

    print("[4/5] Exporting the dashboard replay contract")
    run_script(
        PIPELINE_DIR / "export_navigation_replay.py",
        "--segment", str(SEGMENT_PATH),
        "--ekf-output", str(EKF_OUTPUT_PATH),
        "--output", str(REPLAY_PATH),
    )

    print("[5/5] Validating replay timing")
    replay = pd.read_json(REPLAY_PATH)
    replay["timestamp"] = pd.to_datetime(replay["timestamp"])
    replay_spacing = replay["timestamp"].diff().dropna().dt.total_seconds()
    print(f"Replay median sample interval: {replay_spacing.median():.3f} seconds")
    print(f"Replay output ready: {REPLAY_PATH}")


if __name__ == "__main__":
    main()
