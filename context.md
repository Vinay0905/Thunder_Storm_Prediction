# Project Context: Thunderstorm Prediction System

## 1. Objective
A binary classification system to predict thunderstorms in North East India using 40 years of IMD data.

## 2. Technical Stack
- **Model**: XGBoost (SMOTE-resampled)
- **API**: FastAPI (Uvicorn)
- **UI**: Streamlit
- **Packaging**: Joblib (for model, medians, and scaler)

## 3. The "Battle Scars" (Key Challenges Faced)

### A. Column Shift Syndrome
Raw CSV files had mismatched headers and data. 
- **Fixed by**: Surgical loading using `index_col=False` and explicit column realignment in `preprocess.py`.

### B. Training-Serving Skew
The API initially gave different results than the notebook for the same logic.
- **Root Cause**: Redundant feature engineering and unit mismatches.
- **Fixed by**: Creating a shared `FeatureEngineering.py` module as the "Single Source of Truth."

### C. The "Thickness" Trap (Data Scale)
Inputting `570` (decameters) instead of `5700` (meters) for atmospheric thickness crashed the model's confidence.
- **Fixed by**: Implementing **Heuristic Guardrails** in the pipeline to auto-correct units.

### D. Invariant Predictions (Scaler Mismatch)
Applying a `StandardScaler` to a model trained on raw data caused identical/wrong predictions (mostly 0.01 probability).
- **Fixed by**: Re-training the model (v2) with the scaler integrated into the training pipeline and exported as `scaler.joblib`.

## 4. Current Architecture
- **`pipelines/FeaturePipeline/FeatureEngineering.py`**: Contains `apply_guardrails` and calculation logic.
- **`api/utils.py`**: Loads model, medians, feature names, and scaler. Standardizes request data.
- **`api/main.py`**: FastAPI routes.
- **`streamlit/streamlit_ui.py`**: The final user-facing dashboard.

## 5. Handoff Note
Always ensure `scaler.joblib` and `xgboost_model.joblib` are synchronized. If the model is re-trained, the scaler MUST be re-fitted on the same training set.
