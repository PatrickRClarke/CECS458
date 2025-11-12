from fastapi import FastAPI
import uvicorn
import json

app = FastAPI()

@app.get("/")
def root():
    return {"status": "alive"}

@app.get("/prediction")
def get_prediction():
    with open("latest_prediction.json", "r") as f:
        data = json.load(f)
    return data

if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0", port =8000)