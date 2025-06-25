import streamlit as st
import pandas as pd
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import json
import requests
import os
from datetime import datetime
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Constants
CSV_FILE = "valencia_traffic_data.csv"
MODEL_FILE = "traffic_model.pkl"
PREDICTION_FILE = "valencia_traffic_predictions.csv"

# Function Definitions
@st.cache_data
def fetch_traffic_data():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=estat-transit-temps-real-estado-trafico-tiempo-real&rows=1000"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["records"]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

def parse_records(records):
    parsed = []
    now = datetime.utcnow()
    for record in records:
        fields = record.get("fields", {})
        parsed.append({
            "timestamp": now.isoformat(),
            "segment_id": fields.get("idtramo"),
            "status": fields.get("estado"),
            "name": fields.get("denominacion"),
            "lat": fields.get("geo_point_2d", [None, None])[0],
            "lon": fields.get("geo_point_2d", [None, None])[1],
        })
    return parsed

def save_to_csv(data):
    df_new = pd.DataFrame(data)
    if os.path.exists(CSV_FILE):
        df_old = pd.read_csv(CSV_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CSV_FILE, index=False)

def train_model():
    df = pd.read_csv(CSV_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["segment_id", "status"])
    df["segment_id"] = df["segment_id"].astype(float)
    df["status"] = df["status"].astype(float)

    # Create prediction target
    df_sorted = df.sort_values(["segment_id", "timestamp"])
    df_sorted["future_status"] = df_sorted.groupby("segment_id")["status"].shift(-2)
    df_model = df_sorted.dropna(subset=["future_status"])
    df_model["hour"] = df_model["timestamp"].dt.hour
    df_model["minute"] = df_model["timestamp"].dt.minute
    df_model["weekday"] = df_model["timestamp"].dt.weekday

    X = df_model[["segment_id", "hour", "minute", "weekday"]]
    y = df_model["future_status"].astype(int)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

    latest_time = df["timestamp"].max()
    current_batch = df[df["timestamp"] == latest_time].copy()
    current_batch["hour"] = current_batch["timestamp"].dt.hour
    current_batch["minute"] = current_batch["timestamp"].dt.minute
    current_batch["weekday"] = current_batch["timestamp"].dt.weekday

    X_current = current_batch[["segment_id", "hour", "minute", "weekday"]]
    current_batch["predicted_status"] = model.predict(X_current)

    current_batch[["segment_id", "predicted_status", "name", "lat", "lon"]].to_csv(PREDICTION_FILE, index=False)

def load_predictions():
    return pd.read_csv(PREDICTION_FILE)

# App UI
st.title("🚦 Valencia Real-Time Traffic App")

tab1, tab2 = st.tabs(["📈 Model Training", "🗺️ Map View"])

with tab1:
    if st.button("Fetch Data and Train Model"):
        with st.spinner("Fetching data..."):
            records = fetch_traffic_data()
            parsed = parse_records(records)
            save_to_csv(parsed)
        with st.spinner("Training model..."):
            train_model()
        st.success("Data fetched and model trained. Predictions saved.")

with tab2:
    st.subheader("Traffic Map with Predicted Congestion")

    traffic_segments = pd.read_csv("estat-transit-temps-real-estado-trafico-tiempo-real.csv", sep=';')
    predictions = load_predictions()

    traffic_segments["Id. Tram / Id. Tramo"] = traffic_segments["Id. Tram / Id. Tramo"].astype(float)
    predictions["segment_id"] = predictions["segment_id"].astype(float)

    merged = pd.merge(
        traffic_segments,
        predictions,
        left_on="Id. Tram / Id. Tramo",
        right_on="segment_id",
        how="inner"
    )

    def parse_linestring(geo_shape_str):
        try:
            return json.loads(geo_shape_str.replace("'", '"'))
        except:
            return None

    merged["geometry"] = merged["geo_shape"].apply(parse_linestring)
    merged = merged[merged["geometry"].notnull()]

    def status_color(status):
        return {0: "green", 1: "orange", 2: "red"}.get(status, "gray")

    m = folium.Map(location=[39.4702, -0.3768], zoom_start=13)
    Fullscreen().add_to(m)

    for _, row in merged.iterrows():
        coords = row["geometry"]["coordinates"]
        coords_latlon = [(lat, lon) for lon, lat in coords]
        color = status_color(row["predicted_status"])
        folium.PolyLine(coords_latlon, color=color, weight=5, opacity=0.8, popup=row["name"]).add_to(m)

    st_folium(m, width=1000, height=600)
