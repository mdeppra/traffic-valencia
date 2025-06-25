import streamlit as st
import pandas as pd
import folium
from folium import GeoJson
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import json

# Load the data
traffic_segments = pd.read_csv("estat-transit-temps-real-estado-trafico-tiempo-real.csv", sep=';')
predictions = pd.read_csv("valencia_traffic_predictions.csv")

# Prepare and clean
traffic_segments["Id. Tram / Id. Tramo"] = traffic_segments["Id. Tram / Id. Tramo"].astype(float)
predictions["segment_id"] = predictions["segment_id"].astype(float)

# Merge by segment ID
merged = pd.merge(
    traffic_segments,
    predictions,
    left_on="Id. Tram / Id. Tramo",
    right_on="segment_id",
    how="inner"
)

# Parse geometry strings to JSON
def parse_linestring(geo_shape_str):
    try:
        return json.loads(geo_shape_str.replace("'", '"'))
    except:
        return None

merged["geometry"] = merged["geo_shape"].apply(parse_linestring)
merged = merged[merged["geometry"].notnull()]

# Set up map
valencia_center = [39.4702, -0.3768]
m = folium.Map(location=valencia_center, zoom_start=13)
Fullscreen().add_to(m)

# Define color function
def status_color(status):
    if status == 0:
        return "green"
    elif status == 1:
        return "orange"
    elif status == 2:
        return "red"
    else:
        return "gray"

# Add line segments
for _, row in merged.iterrows():
    coords = row["geometry"]["coordinates"]
    coords_latlon = [(lat, lon) for lon, lat in coords]  # Flip order
    color = status_color(row["predicted_status"])
    folium.PolyLine(
        coords_latlon,
        color=color,
        weight=5,
        opacity=0.8,
        popup=row["name"]
    ).add_to(m)

# Render in Streamlit
st.title("Valencia Real-Time Traffic Predictions (Streets)")
st_data = st_folium(m, width=1000, height=600)
