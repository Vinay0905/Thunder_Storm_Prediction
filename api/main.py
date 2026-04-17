"""
Reason: 
    This is the entry point. 
    It keeps the routes clean by calling the functions in utils.py. 
    We use a Lifespan event (or startup) to load the model only once when the server starts, 
    rather than loading it for every request (which would be very slow).
"""

from fastapi import FastAPI,HTTPException

from api.models import PredictionRequest,PrediuctionResponse

from api.utils import load_artifacts,engineer_feature

import pandas as pd 


app = FastAPI(title="Thunderstorm Prediction API")
# Global variables to store our model and metadata
model = None
medians = None
feature_names = None
@app.on_event("startup")
def startup_event():
    global model, medians, feature_names
    model, medians, feature_names = load_artifacts()
@app.get("/")
def read_root():
    return {"message": "Thunderstorm Prediction API is online"}
@app.post("/predict", response_model=PrediuctionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert request to dict and engineer features
        input_data = request.dict()
        processed_df = engineer_feature(input_data, medians)
        
        # Ensure column order matches training
        processed_df = processed_df[feature_names]
        
        # Make Prediction
        prediction = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))