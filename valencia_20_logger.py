import requests
import pandas as pd
from datetime import datetime
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import SMOTE

CSV_FILE = "valencia_traffic_data.csv"
MODEL_FILE = "traffic_model.pkl"
PREDICTION_FILE = "valencia_traffic_predictions.csv"

def fetch_traffic_data():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=estat-transit-temps-real-estado-trafico-tiempo-real&rows=1000"
    try:
        response = requests.get(url)
        response.raise_for_status()
        records = response.json()["records"]
        return records
    except Exception as e:
        print(f"❌ Error during fetch: {e}")
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
    print(f"💾 Data saved to {CSV_FILE} ({len(data)} new rows)")

def train_model():
    try:
        df = pd.read_csv(CSV_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["segment_id", "status"])
        df["segment_id"] = df["segment_id"].astype(float)
        df["status"] = df["status"].astype(float)

        # Sort and create target: status 6 minutes ahead (assuming data every 3 mins, shift by 2)
        df_sorted = df.sort_values(["segment_id", "timestamp"])
        df_sorted["future_status"] = df_sorted.groupby("segment_id")["status"].shift(-2)
        df_model = df_sorted.dropna(subset=["future_status"])

        # Fix SettingWithCopyWarning by using .loc
        df_model.loc[:, "hour"] = df_model["timestamp"].dt.hour
        df_model.loc[:, "minute"] = df_model["timestamp"].dt.minute
        df_model.loc[:, "weekday"] = df_model["timestamp"].dt.weekday

        X = df_model[["segment_id", "hour", "minute", "weekday"]]
        y = df_model["future_status"].astype(int)

        # Handle imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_FILE)
        print("🤖 Model trained and saved.")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

        # Predict current status for latest timestamp batch
        latest_time = df["timestamp"].max()
        current_batch = df[df["timestamp"] == latest_time].copy()

        current_batch.loc[:, "hour"] = current_batch["timestamp"].dt.hour
        current_batch.loc[:, "minute"] = current_batch["timestamp"].dt.minute
        current_batch.loc[:, "weekday"] = current_batch["timestamp"].dt.weekday

        X_current = current_batch[["segment_id", "hour", "minute", "weekday"]]
        current_batch["predicted_status"] = model.predict(X_current)

        current_batch[["segment_id", "predicted_status", "name", "lat", "lon"]].to_csv(PREDICTION_FILE, index=False)
        print(f"📈 Predictions saved to {PREDICTION_FILE}")

    except Exception as e:
        print(f"⚠️ Error training model: {e}")

def main():
    print("🚦 Starting Valencia traffic logger with 6-minute future prediction...")
    while True:
        records = fetch_traffic_data()
        if records:
            parsed = parse_records(records)
            save_to_csv(parsed)
            train_model()
        else:
            print("⚠️ No new data.")
        time.sleep(180)  # Wait 3 minutes (180 seconds)

if __name__ == "__main__":
    main()
