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

from api.utils import load_artifacts,prepare_data_for_Prediction

import pandas as pd 


app = FastAPI(title="Thunderstorm Prediction API")
# Global variables to store our model and metadata
model,medians,feature_names,scaler=None,None,None,None
@app.on_event("startup")
def startup_event():
    global model, medians, feature_names,scaler
    model, medians, feature_names,scaler = load_artifacts()
@app.get("/")
def read_root():
    return {"message": "Thunderstorm Prediction API is online"}
@app.post("/predict", response_model=PrediuctionResponse)
def predict(request: PredictionRequest):
    try:
        # 1. Process and Engineer features
        input_data = request.dict()
        processed_df = prepare_data_for_Prediction(input_data, medians,feature_names,scaler)

        # 2. Ensure column order matches exactly what the model was trained on
        # processed_df = processed_df[feature_names]
        
        # 3. Make Prediction
        prediction = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")