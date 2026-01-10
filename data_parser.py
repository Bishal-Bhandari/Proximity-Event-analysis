import pandas as pd
import numpy as np
import os

def parse_vm2_file(input_path, output_path, confirmed_csv=None):

    # Read raw file and extract only sensor lines
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Find sensor data starts pint
    sep_idx = None
    for i, line in enumerate(lines):
        if "=========================" in line:
            sep_idx = i
            break

    if sep_idx is None:
        raise ValueError("Couldn't find sensor data separator in the file.")

    # Sensor data starts right after the separator line
    sensor_lines = lines[sep_idx + 1 :]

    # identify header row n load sensor data into DataFrame
    header = None
    for i, line in enumerate(sensor_lines):
        # First non-empty line is the header
        if line.strip() and not line.startswith("==="):
            header = line.strip().split(",")
            data_start = i + 1
            break

    if header is None:
        raise ValueError("Couldn't find header row for sensor table.")

    # Read the rest of sensor rows into a df
    data_rows = sensor_lines[data_start:]
    parsed_rows = []

    for row in data_rows:
        row = row.strip()
        if not row:
            continue

        values = row.split(",")

        # Fix row length mismatch
        if len(values) < len(header):
            values += [np.nan] * (len(header) - len(values))
        elif len(values) > len(header):
            values = values[:len(header)]

        parsed_rows.append(values)

    df_raw = pd.DataFrame(parsed_rows, columns=header)

    # Convert numeric columns to appropriate types
    def to_numeric(col):
        return pd.to_numeric(col, errors="coerce")

    # Convert common numeric columns
    for col in ["lat", "lon", "obsDistanceLeft", "obsDistanceRight"]:
        if col in df_raw.columns:
            df_raw[col] = to_numeric(df_raw[col])

    # Convert timestamp
    if "timeStamp" in df_raw.columns:
        try:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timeStamp"], unit="s", errors="coerce")
        except Exception as e:
            # If timeStamp is a formatted string, try generic parse
            df_raw["timestamp"] = pd.to_datetime(df_raw["timeStamp"], errors="coerce")

    # Rename columns consistently
    rename_map = {
        "obsDistanceLeft": "left_dist",
        "obsDistanceRight": "right_dist"
    }
    df_raw = df_raw.rename(columns=rename_map)

    # SSort & cleanup
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    # ensure lat/lon exist
    for col in ["lat", "lon"]:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    # derived features
    df_raw["min_dist"] = df_raw[["left_dist", "right_dist"]].min(axis=1)

    df_raw["asymmetry"] = (df_raw["left_dist"] - df_raw["right_dist"]).abs()

    # rate of change
    df_raw["delta_min_dist"] = df_raw["min_dist"].diff().fillna(0)

    # Recovery proxy: next minus current
    df_raw["delta_recovery"] = df_raw["min_dist"].shift(-1) - df_raw["min_dist"]
    df_raw["delta_recovery"] = df_raw["delta_recovery"].fillna(0)

    motion_cols = [c for c in df_raw.columns if c.lower() in ["acc", "x", "y", "z", "a", "b", "c"]]
    if motion_cols:
        df_raw["acc_magnitude"] = np.sqrt(np.sum(np.power(df_raw[motion_cols].fillna(0), 2), axis=1))

    if confirmed_csv is not None:
        df_confirmed = pd.read_csv(confirmed_csv, parse_dates=["timestamp"])
        df = pd.merge_asof(df_raw.sort_values("timestamp"),
                           df_confirmed.sort_values("timestamp"),
                           on="timestamp",
                           direction="nearest",
                           tolerance=pd.Timedelta("1s"))
        df["Confirmed"] = df["Confirmed"].fillna(0).astype(int)
    else:
        df = df_raw.copy()
        df["Confirmed"] = 0

    # Save output
    df.to_csv(output_path, index=False)
    print(f"\nParser finished \nSaved cleaned CSV to:\n   {output_path}\n")

parse_vm2_file(
    input_path="For_parsing1/VM2_3220027",
    output_path="Parsed_file/ride123_clean.csv"
)


