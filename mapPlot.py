import pandas as pd
import folium

# CONFIGURATION
CSV_PATH = "Data/Exp50_Map-matched&DTW.csv"
LAT_COL = "lat"
LON_COL = "lon"
CONFIRMED_COL = "Confirmed"
OUTPUT_MAP = "PlotsImg/confirmed_events_map.html"

# LOAD DATA
df = pd.read_csv(CSV_PATH)

# Handle 'lan' vs 'lon'
if LON_COL not in df.columns and "lan" in df.columns:
    df = df.rename(columns={"lan": "lon"})

# CREATE BASE MAP CENTERED ON DATA
center_lat = df[LAT_COL].mean()
center_lon = df[LON_COL].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# ADD POINTS TO MAP
for _, row in df.iterrows():
    lat = row[LAT_COL]
    lon = row[LON_COL]
    confirmed = row[CONFIRMED_COL]

    # Color based on confirmed
    color = 'red' if confirmed > 0 else 'green'

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Confirmed: {confirmed}"
    ).add_to(m)

# SAVE MAP
m.save(OUTPUT_MAP)
print(f"✅ Map saved as {OUTPUT_MAP}")
