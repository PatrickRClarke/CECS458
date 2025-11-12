import json, time
from tensorflow.keras.models import load_model
#from model_loader import load_model, make_prediction, fetch_latest_data

model = load_model("best_lstm_model.h5")

while True:
    #data = fetch_latest_data()
    #pred = make_prediction(model, data)
    pred = 0xC0FFEE
    with open("latest_prediction.json", "w") as f:
        json.dump({"prediction": float(pred)}, f)
        
    time.sleep(300)
    