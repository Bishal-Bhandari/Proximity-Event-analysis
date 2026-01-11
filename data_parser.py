import pandas as pd
import numpy as np
import os


def parse_vm2_file(input_path, output_path, confirmed_csv=None):

    #Read all lines
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Locate the sensor header row
    sensor_header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("lat,lon,") and "timeStamp" in line:
            sensor_header_idx = i
            break

    if sensor_header_idx is None:
        raise ValueError("Could not find sensor header row in the file.")

    #Extract the header and following lines as sensor data
    header_line = lines[sensor_header_idx].strip()
    data_lines = lines[sensor_header_idx + 1 :]

    #Build a DataFrame from sensor data
    cols = header_line.split(",")
    parsed_rows = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")

        if len(parts) < len(cols):
            parts += [None] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            parts = parts[:len(cols)]
        parsed_rows.append(parts)

    df = pd.DataFrame(parsed_rows, columns=cols)

    # Convert numeric columns
    numeric_cols = [
        "lat", "lon",
        "X", "Y", "Z",
        "timeStamp",
        "acc", "a", "b", "c",
        "obsDistanceLeft1", "obsDistanceLeft2",
        "obsDistanceRight1", "obsDistanceRight2"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward fill lat/lon where they are empty
    df["lat"] = df["lat"].ffill()
    df["lon"] = df["lon"].ffill()

    # Compute combined left/right distances
    # If both channels exist, take the minimum (closest object)
    if "obsDistanceLeft1" in df.columns and "obsDistanceLeft2" in df.columns:
        df["left_dist"] = df[["obsDistanceLeft1", "obsDistanceLeft2"]].min(axis=1)
    else:
        df["left_dist"] = df["obsDistanceLeft1"]

    if "obsDistanceRight1" in df.columns and "obsDistanceRight2" in df.columns:
        df["right_dist"] = df[["obsDistanceRight1", "obsDistanceRight2"]].min(axis=1)
    else:
        df["right_dist"] = df["obsDistanceRight1"]

    # Convert timestamp in ms to proper datetime
    # Your sample clearly uses ms since epoch
    df["timestamp"] = pd.to_datetime(df["timeStamp"], unit="ms", errors="coerce")

    #Keep rows with a valid timestamp
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    #Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derived features for ML
    df["min_dist"] = df[["left_dist", "right_dist"]].min(axis=1)
    df["asymmetry"] = (df["left_dist"] - df["right_dist"]).abs()
    df["delta_min_dist"] = df["min_dist"].diff().fillna(0)
    df["delta_recovery"] = df["min_dist"].shift(-1) - df["min_dist"]
    df["delta_recovery"] = df["delta_recovery"].fillna(0)

    # Optional motion intensity
    if {"acc", "X", "Y", "Z"}.issubset(df.columns):
        df["motion_mag"] = np.sqrt(
            df["acc"].fillna(0) ** 2 +
            df["X"].fillna(0) ** 2 +
            df["Y"].fillna(0) ** 2 +
            df["Z"].fillna(0) ** 2
        )

    # Merge Confirmed labels (if provided)
    if confirmed_csv is not None:
        df_confirmed = pd.read_csv(confirmed_csv, parse_dates=["timestamp"])
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_confirmed.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1s")
        )
        df["Confirmed"] = df["Confirmed"].fillna(0).astype(int)
    else:
        df["Confirmed"] = 0

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✔ Saved parsed CSV to: {output_path}")

parse_vm2_file(
    input_path="For_parsing1/VM2_3220027",
    output_path="Parsed/ride3220027_clean.csv",
    confirmed_csv=None
)
