import uvicorn
from fastapi import FastAPI
from rssi import Rssi
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/predict")
def predict(data: Rssi):
    data = data.dict()
    rssi1 = data['rssiae']
    rssi2 = data['rssi0e']
    rssi3 = data['rssi0f']
    print(model.predict([[rssi1, rssi2, rssi3]]))
    prediction = model.predict([[rssi1, rssi2, rssi3]])
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
