# Interview Preparation Guide: Thunderstorm Prediction System

## 1. The "Elevator Pitch" (30-60 Seconds)
"I built an end-to-end MLOps system for predicting thunderstorms in North East India using 40 years of historical meteorological data. The system doesn't just train a model; it's a production-grade pipeline that handles raw data cleaning, automated feature engineering, hyperparameter tuning with Optuna, and serves real-time predictions via a FastAPI backend and a Streamlit dashboard. The most challenging part was ensuring 'Training-Serving' consistency, which I solved by creating a centralized inference layer with heuristic guardrails."

## 2. Key Technical Talking Points

### A. Data Realignment (The Detective Work)
- **Challenge**: Raw IMD data had shifted columns and header mismatches.
- **Solution**: Implemented surgical parsing and realignment logic during the preprocessing phase.

### B. MLOps Core (The Infrastructure)
- **Feature Pipeline**: A shared module that serves as the "Single Source of Truth."
- **Inference Pipeline**: A class-based runner that handles artifacts (scalers/medians) and guardrails automatically.
- **Training Suite**: Modular scripts for training, evaluation, and tuning (using MLflow for tracking).

### C. Guardrails & Robustness
- **Problem**: Models often fail on noisy real-world data (e.g., unit mismatches).
- **Solution**: Built "Heuristic Guardrails" into the inference pipeline to auto-correct common input errors (like atmospheric thickness units).

## 3. Potential Interview Questions

### Technical / ML
1. "How did you handle class imbalance in the thunderstorm dataset?" (Answer: SMOTE)
2. "Why choose XGBoost for this specific problem?" (Answer: Non-linear relationships in thermodynamics, handling sparse data).
3. "What is Training-Serving skew, and how did you prevent it?" (Answer: Shared feature engineering modules and the InferencePipeline class).

### Architecture / MLOps
1. "Walk me through how a raw data point becomes a prediction in your system."
2. "How would you safely retrain this model without breaking the API?"
3. "Why use MLflow and Optuna instead of just grid searching in a notebook?"

### Problem Solving
1. "Tell me about a time you found a data quality issue and how you fixed it." (The Column Shift Syndrome).
