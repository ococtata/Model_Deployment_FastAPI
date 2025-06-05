from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AbaloneInput(BaseModel):
    Sex: int  
    Length: float
    Diameter: float
    Height: float
    WholeWeight: float
    ShuckedWeight: float
    VisceraWeight: float
    ShellWeight: float

@app.get("/")
def read_root():
    return {"message": "Abalone age prediction API is live!"}

@app.post("/predict")
def predict(data: AbaloneInput):
    input_array = np.array([[data.Sex, data.Length, data.Diameter, data.Height,
                             data.WholeWeight, data.ShuckedWeight,
                             data.VisceraWeight, data.ShellWeight]])
    
    scaled = scaler.transform(input_array)
    prediction = model.predict(scaled)[0]
    return {
        "predicted_rings": round(prediction, 2),
        "estimated_age": round(prediction + 1.5, 2)
    }
