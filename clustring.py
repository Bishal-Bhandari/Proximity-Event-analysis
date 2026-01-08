import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = "Data/Exp01_Map-matched&DTW.csv"
OUTPUT_DIR = "PlotsImg"
LAT_COL = "lat"
LON_COL = "lon"
CONFIRMED_COL = "Confirmed"
LEFT_COL = "Left"
RIGHT_COL = "Right"

df = pd.read_csv(CSV_PATH)

# handle gps
if LON_COL not in df.columns and "lan" in df.columns:
    df = df.rename(columns={"lan": "lon"})

required_cols = [LAT_COL, "lon", CONFIRMED_COL, LEFT_COL, RIGHT_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LAT vs LON

plt.figure()
sc = plt.scatter(
    df["lon"],
    df[LAT_COL],
    c=df[CONFIRMED_COL],
    alpha=0.6
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Latitude vs Longitude (Confirmed)")
plt.colorbar(sc, label="Confirmed")
plt.savefig(f"{OUTPUT_DIR}/lat_lon_vs_confirmed.png", dpi=300)
plt.close()


# ONFIRMED vs LEFT & RIGHT
plt.figure()
plt.scatter(df[CONFIRMED_COL], df[LEFT_COL], alpha=0.6, label="Left")
plt.scatter(df[CONFIRMED_COL], df[RIGHT_COL], alpha=0.6, label="Right")
plt.xlabel("Confirmed")
plt.ylabel("Value")
plt.title("Confirmed vs Left and Right")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/confirmed_vs_left_right.png", dpi=300)
plt.close()

# LAT, LON, CONFIRMED
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["lon"],
    df[LAT_COL],
    df[CONFIRMED_COL],
    c=df[CONFIRMED_COL],
    alpha=0.6
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Confirmed")
ax.set_title("3D: Lat-Lon-Confirmed")

plt.savefig(f"{OUTPUT_DIR}/3d_lat_lon_confirmed.png", dpi=300)
plt.close()

print("✅ Plots saved successfully in:", OUTPUT_DIR)
