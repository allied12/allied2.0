# traffic_full.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import asyncio
import threading
import time
import os
import datetime
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
from pathlib import Path
import sqlite3

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

# ---------------------------
# Shared data
# ---------------------------
traffic_data = {"left_lane": 0, "right_lane": 0, "priority": "Unknown"}
traffic_predictions = {"left_lane": 0, "right_lane": 0}
signal_state = {"left_lane": "red", "right_lane": "red"}
is_priority_vehicle_present = {"left_lane": False, "right_lane": False}
priority_vehicle_type = "None"

# ---------------------------
# Paths inside the container
# ---------------------------
VIDEO_PATH = "video/input.mp4"         # Place your video here
YOLO_MODEL_PATH = "yolov8n.pt"        # Place your model in repo
DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

DB_FILE = DATA_PATH / "traffic_project.db"

# ---------------------------
# Initialize capture
# ---------------------------
def make_capture(path):
    cap_obj = cv2.VideoCapture(path)
    if cap_obj.isOpened():
        return cap_obj
    cap_obj.release()
    return cv2.VideoCapture(0)

cap = make_capture(VIDEO_PATH)
latest_frame = None

# ---------------------------
# YOLO and GRU setup
# ---------------------------
model = YOLO(YOLO_MODEL_PATH)

scaler = MinMaxScaler(feature_range=(0, 1))
gru_model = Sequential([
    GRU(32, activation='relu', input_shape=(None, 2)),
    Dense(2)
])
gru_model.compile(optimizer='adam', loss='mean_squared_error')


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# ---------------------------
# Database setup
# ---------------------------
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            timestamp TEXT,
            left_lane INTEGER,
            right_lane INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# ---------------------------
# Detection thread
# ---------------------------
def detect_traffic():
    global traffic_data, latest_frame, cap, is_priority_vehicle_present, priority_vehicle_type

    print("[detector] started")
    log_timer = time.time()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    while not cap.isOpened():
        print("[detector] capture not opened. Trying to open...")
        cap = make_capture(VIDEO_PATH)
        time.sleep(1)

    frame_counter = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = make_capture(VIDEO_PATH)
                time.sleep(0.5)
                continue

            latest_frame = frame.copy()
            frame_counter += 1
            if frame_counter % 5 != 0:
                continue

            small_frame = cv2.resize(frame, (640, 480))
            results = model.track(small_frame, persist=True, classes=[2,3,5,7], verbose=False)
            left_lane_count, right_lane_count = 0, 0

            is_priority_vehicle_present["left_lane"] = False
            is_priority_vehicle_present["right_lane"] = False
            priority_vehicle_type = "None"

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()

                original_h, original_w = frame.shape[:2]
                scale_x = original_w / 640
                scale_y = original_h / 480

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x_scaled = box[0] * scale_x
                    y_scaled = box[1] * scale_y
                    w_scaled = box[2] * scale_x
                    h_scaled = box[3] * scale_y
                    cx = x_scaled

                    if cls == 5:
                        priority_vehicle_type = "Ambulance"
                        if cx < original_w // 2:
                            is_priority_vehicle_present["left_lane"] = True
                        else:
                            is_priority_vehicle_present["right_lane"] = True
                    elif cls == 7:
                        priority_vehicle_type = "Firetruck"
                        if cx < original_w // 2:
                            is_priority_vehicle_present["left_lane"] = True
                        else:
                            is_priority_vehicle_present["right_lane"] = True

                    if cx < original_w // 2:
                        left_lane_count += 1
                        color = (255, 0, 0)
                    else:
                        right_lane_count += 1
                        color = (0, 255, 0)

                    cv2.rectangle(latest_frame,
                                  (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)),
                                  (int(x_scaled+w_scaled/2), int(y_scaled+h_scaled/2)),
                                  color, 2)
                    cv2.putText(latest_frame,
                                f"ID: {track_id}",
                                (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            priority = "Left" if left_lane_count > right_lane_count else "Right"
            traffic_data = {"left_lane": left_lane_count, "right_lane": right_lane_count, "priority": priority}

            if time.time() - log_timer >= 5:
                cursor.execute("INSERT INTO traffic_data VALUES (?, ?, ?)",
                               (datetime.datetime.now().isoformat(), left_lane_count, right_lane_count))
                conn.commit()
                log_timer = time.time()

            time.sleep(0.01)

        except Exception as e:
            print(f"[detector] Error: {e}")
            time.sleep(1)

# ---------------------------
# Signal management
# ---------------------------
def manage_signals():
    global signal_state, traffic_data, is_priority_vehicle_present
    while True:
        if is_priority_vehicle_present["left_lane"]:
            signal_state["left_lane"] = "green"
            signal_state["right_lane"] = "red"
        elif is_priority_vehicle_present["right_lane"]:
            signal_state["left_lane"] = "red"
            signal_state["right_lane"] = "green"
        else:
            priority = traffic_data.get("priority", "Unknown")
            if priority == "Left":
                signal_state["left_lane"] = "green"
                signal_state["right_lane"] = "red"
            elif priority == "Right":
                signal_state["left_lane"] = "red"
                signal_state["right_lane"] = "green"
        time.sleep(2)

# ---------------------------
# Prediction thread
# ---------------------------
def predict_traffic_patterns():
    global traffic_predictions, gru_model, scaler, DB_FILE
    look_back = 10

    while True:
        try:
            conn = sqlite3.connect(DB_FILE)
            df = pd.read_sql_query("SELECT left_lane, right_lane FROM traffic_data ORDER BY timestamp DESC LIMIT 15", conn)
            conn.close()
            if len(df) >= look_back + 1:
                data = df.values.astype(np.float32)
                scaler.fit(data)
                scaled_data = scaler.transform(data)
                X, y = create_sequences(scaled_data, look_back)
                X = X.reshape(X.shape[0], look_back, 2)
                gru_model.fit(X, y, epochs=1, verbose=0)
                last_sequence = scaled_data[-look_back:].reshape(1, look_back, 2)
                prediction_scaled = gru_model.predict(last_sequence)
                prediction = scaler.inverse_transform(prediction_scaled)
                traffic_predictions = {
                    "left_lane": int(np.maximum(0, prediction[0][0])),
                    "right_lane": int(np.maximum(0, prediction[0][1]))
                }
        except Exception as e:
            print(f"[predictor] Error: {e}")
        time.sleep(300)

# ---------------------------
# Frame generator
# ---------------------------
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            try:
                _, buffer = cv2.imencode(".jpg", latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception as e:
                print(f"[stream] encode error: {e}")
        else:
            time.sleep(0.05)

# ---------------------------
# FastAPI endpoints
# ---------------------------
app.mount("/dashboard_assets", StaticFiles(directory="dashboard"), name="dashboard_assets")

@app.get("/")
def root():
    return {"message": "🚦 AI Traffic System Running. Go to /dashboard"}

@app.get("/traffic_status")
def get_status():
    return traffic_data

@app.get("/traffic_predictions")
def get_predictions():
    return traffic_predictions

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "realtime": traffic_data,
                "predicted": traffic_predictions,
                "signal": signal_state,
                "priority_alert": is_priority_vehicle_present,
                "priority_vehicle_type": priority_vehicle_type
            })
            await asyncio.sleep(1)
    except Exception as e:
        print(f"[ws] connection closed: {e}")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/dashboard")
def get_dashboard():
    return FileResponse("dashboard/index.html")

# ---------------------------
# Start threads
# ---------------------------
if __name__ == "__main__":
    setup_database()
    threading.Thread(target=detect_traffic, daemon=True).start()
    threading.Thread(target=predict_traffic_patterns, daemon=True).start()
    threading.Thread(target=manage_signals, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
