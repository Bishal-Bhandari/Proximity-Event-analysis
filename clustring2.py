import pandas as pd
import folium

# Load CSV
df = pd.read_csv("Data/Exp01_Map-matched&DTW.csv")

# Center map on mean lat/lon
center_lat = df['lat'].mean()
center_lon = df['lon'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Add points to map
for _, row in df.iterrows():
    color = 'red' if row['Confirmed'] > 0 else 'blue'
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5 + row['Confirmed'],  # optional: bigger circle if Confirmed > 1
        color=color,
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# Save map
m.save("PlotsImg/lat_lon_confirmed_map.html")

print("Map saved as lat_lon_confirmed_map.html")
