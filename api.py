import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib  # For loading machine learning models
import pandas as pd  # For data manipulation
import json  # For reading JSON files
from typing import Optional

app = FastAPI(title="Heart Disease Prediction API")

# Get allowed origins from environment variable or use defaults
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,https://frontend-heart-predictor-5-git-530957-arturos-projects-65f856a2.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Only allow needed methods
    allow_headers=["Content-Type", "Authorization"],  # Only needed headers
)



# --- Load Model Artifacts ---
# These are loaded once when the application starts

# 1. Load the trained machine learning model
model = joblib.load("model_rf.joblib")

# 2. Load the scaler used for feature normalization
scaler = joblib.load("scaler.joblib")

# 3. Load the feature column names that the model expects
with open("feature_cols.json", "r") as f:
    FEATURE_COLS = json.load(f)

# Continuous features list that need scaling
CONTINUOUS_FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# --- Pydantic model for input validation ---
class FrontendData(BaseModel):
    # Continuous features
    age: float
    restingBP: float
    cholesterol: float
    maxHR: float
    oldPeak: float
    
    # Categorical features
    sex: int  # 0 = female, 1 = male
    fastingBS: int  # 0 = fasting BS <= 120 mg/dl, 1 = > 120 mg/dl
    exerciseAngina: int  # 0 = no, 1 = yes
    chestPainType: int  # Encoded as: 0=TA, 1=ATA, 2=NAP, 3=ASY
    st_Slope: int  # Encoded as: 0=Down, 1=Flat, 2=Up
    restingECG: int  # Encoded as: 0=LVH, 1=Normal, 2=ST
    
    fullName: Optional[str] = None

# --- Prediction Endpoint ---
@app.post("/predict")
def predict_frontend(data: FrontendData):
    """
    Endpoint that takes patient data, processes it, and returns a heart disease prediction.
    
    Steps:
    1. Map frontend input to model features
    2. Create DataFrame with expected feature columns
    3. Scale continuous features
    4. Make prediction
    5. Return results
    """
    
    # --- Feature Mapping ---
    # Convert frontend input to the format expected by the model
    mapping = {
        # Continuous features (direct mapping)
        "Age": data.age,
        "RestingBP": data.restingBP,
        "Cholesterol": data.cholesterol,
        "MaxHR": data.maxHR,
        "Oldpeak": data.oldPeak,
        
        # Categorical features (direct mapping)
        "Sex": data.sex,
        "FastingBS": data.fastingBS,
        "ExerciseAngina": data.exerciseAngina,
        
        # One-hot encoded chest pain types
        "chestPain_TA": 1 if data.chestPainType == 0 else 0,   # Typical Angina
        "chestPain_ATA": 1 if data.chestPainType == 1 else 0,  # Atypical Angina
        "chestPain_NAP": 1 if data.chestPainType == 2 else 0,  # Non-Anginal Pain
        "chestPain_ASY": 1 if data.chestPainType == 3 else 0,  # Asymptomatic
        
        # One-hot encoded ST slope features
        "ST_Slope_Down": 1 if data.st_Slope == 0 else 0,  # Downsloping
        "ST_Slope_Flat": 1 if data.st_Slope == 1 else 0,  # Flat
        "ST_Slope_Up": 1 if data.st_Slope == 2 else 0,    # Upsloping
        
        # One-hot encoded Resting ECG results
        "RestingECG_LVH": 1 if data.restingECG == 0 else 0,     # Left Ventricular Hypertrophy
        "RestingECG_Normal": 1 if data.restingECG == 1 else 0,  # Normal
        "RestingECG_ST": 1 if data.restingECG == 2 else 0       # ST-T wave abnormality
    }
    # --- Data Preparation ---
    # Create DataFrame with exactly the columns the model expects
    df_row = pd.DataFrame([mapping])[FEATURE_COLS]
    
    # Scale continuous features using the same scaler from training
    df_row[CONTINUOUS_FEATURES] = scaler.transform(df_row[CONTINUOUS_FEATURES])

   # --- Make Prediction ---
    # Get the predicted class (0 = no heart disease, 1 = heart disease)
    pred = model.predict(df_row)[0]
    
    # Get the probability of heart disease
    proba = model.predict_proba(df_row)[0][1]

    # --- Return Results ---
    return {
        "prediction": int(pred),        # Convert numpy int to Python int
        "probability": float(proba),     # Convert numpy float to Python float
        "fullName": data.fullName        # Include patient name
    }