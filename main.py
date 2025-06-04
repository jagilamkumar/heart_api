from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["heart_db"]
collection = db["predictions"]

# Input schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: HeartData):
    try:
        input_data = np.array([[v for v in data.dict().values()]])
        prediction = model.predict(input_data)[0]
        result = "Disease" if prediction == 1 else "No Disease"

        # Save to MongoDB
        record = data.dict()
        record["prediction"] = result
        collection.insert_one(record)

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
