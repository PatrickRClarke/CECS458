# GUI for LSTM
# 11/6/'25

import tkinter as tk
import random
import datetime

#import requests
#import threading

# Double check the logic for this trend. It should be only be changing color if all three of the 
# values in that window are either above the threshold and match the pattern or below the threshold
# and match the pattern. 

#
# Note: If I don't want to mess with the colors then I should have a way so that the size of
# of the range is displayed.
#
#SERVER_URL = "http://0.0.0.0:8000/prediction"

# === Layout Config ===
UPDATE_INTERVAL = 4000  # ms (4 sec for demo; later 300000 = 5 min)
NUM_PREDICTIONS = 5
THRESH_HIGH = 0.6
THRESH_LOW = 0.4

LABELS = ["BTC", "ETH", "DNUT", "TSLA", "AAPL"]

# === GUI Setup ===
root = tk.Tk()
root.title("Prediction Dashboard")
root.geometry("540x420")
root.configure(bg="#1e1e1e")

title = tk.Label(root, text="Model Prediction Monitor", font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#1e1e1e")
title.pack(pady=(15, 5))

timestamp_label = tk.Label(root, text="Last update: —", font=("Segoe UI", 10), fg="#bbbbbb", bg="#1e1e1e")
timestamp_label.pack()

frames = []
history_labels = []

for label_text in LABELS:
    frame = tk.Frame(root, bg="#252526", bd=2, relief="ridge")
    frame.pack(pady=8, padx=20, fill="x")

    title = tk.Label(frame, text=label_text, font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#252526")
    title.pack(anchor="w", padx=8, pady=4)

    hist_frame = tk.Frame(frame, bg="#252526")
    hist_frame.pack(pady=5)

    boxes = []
    for i in range(NUM_PREDICTIONS):
        box = tk.Label(
            hist_frame,
            text="—",
            width=6,
            height=2,
            font=("Consolas", 14, "bold"),
            fg="#ffffff",
            bg="#3a3a3a",
            bd=2,
            relief="groove",
        )
        box.pack(side="left", padx=4)
        boxes.append(box)

    history_labels.append(boxes)
    frames.append(frame)

status_label = tk.Label(root, text="Status: Running (mock data)", font=("Segoe UI", 10), fg="#aaaaaa", bg="#1e1e1e")
status_label.pack(side="bottom", pady=(10, 10))

# === Prediction Data ===
prediction_histories = {lbl: [] for lbl in LABELS}


def check_pattern(values):
    """Return 'green' if down,down,up < 0.4; 'red' if up,up,down > 0.6; else None."""
    patterns = []
    for i in range(len(values) - 2):
        a, b, c = values[i:i + 3]
        # Down, down, up (green)
        if a > b and b > c and all(v < THRESH_LOW for v in (a, b, c)):
            patterns.append("green")
        # Up, up, down (red)
        elif a < b and b < c and all(v > THRESH_HIGH for v in (a, b, c)):
            patterns.append("red")
    return patterns[-1] if patterns else None


def updatePredictions():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    timestamp_label.config(text=f"Last update: {now}")

    # Part 2: This for label_text, boxes in zip( VV becomes replaced with :
    """
    # This should not be inside of here.
    def update_gui_from_server(data):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    timestamp_label.config(text=f"Last update: {now}")

    if not data:
        status_label.config(text="Status: Server offline or error.")
        return

    for label_text, boxes in zip(LABELS, history_labels):
        history = prediction_histories[label_text]
        new_value = float(data.get(label_text, random.uniform(0, 1)))  # fallback if missing
        history.append(new_value)
        if len(history) > NUM_PREDICTIONS:
            history.pop(0)

        # Update GUI cells
        for i, val in enumerate(history):
            boxes[i].config(text=f"{val:.3f}")
        for i in range(len(history), NUM_PREDICTIONS):
            boxes[i].config(text="—", bg="#3a3a3a")

        boxes[-1].config(bg="#3a3a3a")
        if len(history) == NUM_PREDICTIONS:
            pattern_color = check_pattern(history)
            if pattern_color == "green":
                boxes[-1].config(bg="#006400")
            elif pattern_color == "red":
                boxes[-1].config(bg="#8B0000")

    status_label.config(text="Status: Connected and updated.")"""

    for label_text, boxes in zip(LABELS, history_labels):
        history = prediction_histories[label_text]
        new_value = round(random.uniform(0, 1), 3)
        # Part1:This VV should replace this^^.
        # Launch a thread to fetch data without blocking
        #threading.Thread(target=fetch_prediction_from_server, daemon=True).start()

        history.append(new_value)
        if len(history) > NUM_PREDICTIONS:
            history.pop(0)

        # Fill boxes left → right; newest is rightmost
        for i, val in enumerate(history):
            boxes[i].config(text=f"{val:.3f}")
        for i in range(len(history), NUM_PREDICTIONS):
            boxes[i].config(text="—", bg="#3a3a3a")

        # Reset rightmost color to neutral before updating
        boxes[-1].config(bg="#3a3a3a")

        # Apply pattern detection only if 5 predictions exist
        if len(history) == NUM_PREDICTIONS:
            pattern_color = check_pattern(history)
            if pattern_color == "green":
                boxes[-1].config(bg="#00FF00")  # neon green
            elif pattern_color == "red":
                boxes[-1].config(bg="#FF0000")  # neon red

    root.after(UPDATE_INTERVAL, updatePredictions)

"""
# This is a background thread.
def fetch_prediction_from_server():
    #Background thread: fetch prediction, then update GUI safely.
    try:
        response = requests.get(SERVER_URL, timeout=5)
        if response.status_code == 200:
            # Example: the server returns a JSON dict like {"BTC": 0.82, "ETH": 0.61, ...}
            data = response.json()
        else:
            data = None
    except Exception as e:
        data = None

    # Push GUI update back to main thread
    root.after(0, lambda: update_gui_from_server(data))
"""


updatePredictions()
root.mainloop()

