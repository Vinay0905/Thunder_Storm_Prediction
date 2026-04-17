"""
Reason: 
    This is the "brain" of your API. 
    It handles Model Loading (using joblib), 
    Imputation (filling missing values with the medians we calculated), 
    and Feature Engineering (creating the complex indices like Environmental Stability that your model expects).
"""

import joblib
 
import pandas as pd 
import numpy as np 
from pathlib import Path

MODEL_PATH = Path("models/xgboost_model.joblib")
MEDIANS_PATH = Path("models/medians.joblib")
FEATURE_NAMES_PATH = Path("models/Feature_names.joblib")


def load_artifacts():
    model=joblib.load(MODEL_PATH)
    medians=joblib.load(MEDIANS_PATH)
    feature_names=joblib.load(FEATURE_NAMES_PATH)

    if 'target' in feature_names :
         feature_names.remove('target')

    return model,medians,feature_names

def engineer_feature(raw_data:dict,medians:pd.Series):
    # 1. Convert to DataFrame

    df=pd.DataFrame([raw_data])

    #2 Impute 
    for col,val in medians.items():
        if col in df.columns and df[col].isnull().any():
            df[col]=df[col].fillna(val)

    # 3. Feature Engineering
    df['Environmental Stability'] = df['showalter_index'] + df['lifted_index']
    df['Convective Potential'] = df['cape'] + df['cine']
    
    # Rename raw columns to match what the model expects
    df = df.rename(columns={
        'sweat_index': 'SWEAT index',
        'k_index': 'K index',
        'totals_totals_index': 'Totals totals index',
        'precipitable_water': 'Moisture Indices',
        'thickness_1000_500': 'Temperature Pressure',
        'plcl': 'Moisture Temperature Profiles'
    })
    
    return df    


    