import joblib
from numpy.linalg import cond
import pandas as pd 
import joblib 
from pathlib import Path 
import yaml
from typing import List,Dict,Any

import sys
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from pipelines.FeaturePipeline.FeatureEngineering import apply_feature_engineering

class InferencePipeline:
    def __init__(self,confg_path:str="config.yaml"):
        self.project_root=Path(__file__).resolve().parents[2]
        with open(self.project_root/confg_path,'r')as f:
            self.config=yaml.safe_load(f)

        self.model=None
        self.medians=None 
        self.feature_name=None
        self.scaler=None
        self.load_artifacts()

    def load_artifacts(self):
        model_path=self.project_root/self.config['model']['path']
        base_path=model_path.parent
        self.model=joblib.load(model_path)
        self.medians=joblib.load(base_path/"medians.joblib")
        self.feature_name=joblib.load(base_path/"Feature_names.joblib")
        self.scaler=joblib.load(base_path/"scaler.joblib")
        if 'target' in self.feature_name:
            self.feature_name.remove('target')
    
    def preprocess(self,raw_data:Dict[str,Any])->pd.DataFrame:
        df=pd.DataFrame([raw_data])
        df=apply_feature_engineering(df)
        for col,val in self.medians.items():
            if col in df.columns and df[col].isnull().any():
                
                df[col]=df[col].fillna(val)

        df=df.reindex(columns=self.feature_name,fill_value=0)
        df_scaled=pd.DataFrame(self.scaler.transform(df),columns=self.feature_name)
        return df_scaled

    

    def predict(self,raw_data:Dict[str,Any])-> Dict[str,Any]:
        processed_df=self.preprocess(raw_data)
        prediction=int(self.model.predict(processed_df)[0])
        probability=float(self.model.predict_proba(processed_df)[0][1])    
        return{
            "prediction":prediction,
            "Probability":round(probability,4)

        }



if __name__=="__main__":

    pipeline=InferencePipeline()

    sample_data = {
        "showalter_index": 4.12,
        "LIFTED index": 2.5,
        "SWEAT index": 320.5,
        "K index": 28.0,
        "Totals totals index": 45.5,
        "CAPE": 1500.0,
        "CINE": -50.0,
        "PRECIPITABLE WATER": 35.0,
        "1000-500 THICKNESS": 5700.0,
        "PLCL": 850.0
    }


    res=pipeline.predict(sample_data)
    print(f"prediction result: {res}")

