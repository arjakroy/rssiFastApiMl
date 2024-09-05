import uvicorn
from fastapi import FastAPI
from rssi import Rssi
import numpy as np
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"]

)

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
