"""
Reason: 
    This is the entry point. 
    It keeps the routes clean by calling the functions in utils.py. 
    We use a Lifespan event (or startup) to load the model only once when the server starts, 
    rather than loading it for every request (which would be very slow).
"""

"""
Command to run the server:
    uvicorn api.main:app
"""
from fastapi import FastAPI,HTTPException

from api.models import PredictionRequest,PrediuctionResponse

#from api.utils import load_artifacts,prepare_data_for_Prediction

import pandas as pd 

from pipelines.InferencePipeline.inference import InferencePipeline

app = FastAPI(title="Thunderstorm Prediction API")
# Initialize our pipeline once at startup
pipeline = None
@app.on_event("startup")
def startup_event():
    global pipeline
    # This now handles loading model, medians, scaler, and feature names automatically!
    pipeline = InferencePipeline()
@app.get("/")
def read_root():
    return {"message": "Thunderstorm Prediction API is online"}
@app.post("/predict", response_model=PrediuctionResponse)
def predict(request: PredictionRequest):
    try:
        # One simple call handles preprocessing, guardrails, and prediction!
        result = pipeline.predict(request.dict())
        
        return {
            "prediction": result["prediction"],
            "probability": result["Probability"],
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
