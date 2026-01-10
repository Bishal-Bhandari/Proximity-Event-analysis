import pandas as pd
import numpy as np
import os


def parse_vm2_file(input_path, output_path, confirmed_csv=None):
    # Read file
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Find separator line
    sep_idx = None
    for i, line in enumerate(lines):
        if "=========================" in line:
            sep_idx = i
            break

    if sep_idx is None:
        raise ValueError("Sensor data separator not found")

    # Sensor data starts after separator
    sensor_lines = lines[sep_idx + 1:]

    # Detect header or data format
    header = None
    data_start = None
    delimiter = None

    for i, line in enumerate(sensor_lines):
        line = line.strip()
        if not line or line.startswith("="):
            continue

        # Detect delimiter
        if "," in line:
            parts = line.split(",")
            delimiter = ","
        else:
            parts = line.split()
            delimiter = None  # whitespace

        # Check if line looks like a header
        non_numeric = sum(
            not p.replace(".", "", 1).isdigit() for p in parts
        )

        if len(parts) > 3 and non_numeric > 0:
            header = parts
            data_start = i + 1
        else:
            header = None
            data_start = i
        break

    if data_start is None:
        raise ValueError("No sensor data found")

    # Parse data rows
    parsed_rows = []
    for row in sensor_lines[data_start:]:
        row = row.strip()
        if not row:
            continue

        if delimiter == ",":
            values = row.split(",")
        else:
            values = row.split()

        parsed_rows.append(values)

    if not parsed_rows:
        raise ValueError("No data rows parsed")

    # Generate header if missing
    if header is None:
        n_cols = max(len(r) for r in parsed_rows)
        header = [f"col_{i}" for i in range(n_cols)]

    # Normalize row lengths
    normalized_rows = []
    for r in parsed_rows:
        if len(r) < len(header):
            r = r + [np.nan] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[:len(header)]
        normalized_rows.append(r)

    # Create DataFrame
    df_raw = pd.DataFrame(normalized_rows, columns=header)
    df_raw.columns = df_raw.columns.str.strip()

    # Detect timestamp column
    timestamp_col = None
    for c in df_raw.columns:
        if c.lower() in ["timestamp", "time", "time_stamp", "col_0"]:
            timestamp_col = c
            break

    if timestamp_col is None:
        raise ValueError(f"No timestamp column found: {df_raw.columns.tolist()}")

    # Convert timestamp
    df_raw["timestamp"] = pd.to_datetime(
        pd.to_numeric(df_raw[timestamp_col], errors="coerce"),
        unit="s",
        errors="coerce"
    )

    # Drop rows without valid timestamp
    df_raw = df_raw.dropna(subset=["timestamp"])

    # Convert common numeric columns if present
    for col in ["lat", "lon", "obsDistanceLeft", "obsDistanceRight"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Rename distance columns
    df_raw = df_raw.rename(
        columns={
            "obsDistanceLeft": "left_dist",
            "obsDistanceRight": "right_dist"
        }
    )

    # Ensure lat/lon exist
    for col in ["lat", "lon"]:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    # Sort by timestamp
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    # Derived features
    if {"left_dist", "right_dist"}.issubset(df_raw.columns):
        df_raw["min_dist"] = df_raw[["left_dist", "right_dist"]].min(axis=1)
        df_raw["asymmetry"] = (df_raw["left_dist"] - df_raw["right_dist"]).abs()
    else:
        df_raw["min_dist"] = np.nan
        df_raw["asymmetry"] = np.nan

    df_raw["delta_min_dist"] = df_raw["min_dist"].diff().fillna(0)
    df_raw["delta_recovery"] = (
        df_raw["min_dist"].shift(-1) - df_raw["min_dist"]
    ).fillna(0)

    # Optional motion magnitude
    motion_cols = [
        c for c in df_raw.columns
        if c.lower() in ["x", "y", "z", "acc", "a", "b", "c"]
    ]

    if motion_cols:
        df_raw["acc_magnitude"] = np.sqrt(
            np.sum(np.power(df_raw[motion_cols].fillna(0), 2), axis=1)
        )

    # Merge confirmed labels if provided
    if confirmed_csv is not None:
        df_confirmed = pd.read_csv(
            confirmed_csv, parse_dates=["timestamp"]
        )

        df = pd.merge_asof(
            df_raw.sort_values("timestamp"),
            df_confirmed.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1s")
        )

        df["Confirmed"] = df["Confirmed"].fillna(0).astype(int)
    else:
        df = df_raw.copy()
        df["Confirmed"] = 0

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned CSV to: {output_path}")


# Example usage
parse_vm2_file(
    input_path="For_parsing1/VM2_3220027",
    output_path="Parsed_file/ride123_clean.csv"
)
