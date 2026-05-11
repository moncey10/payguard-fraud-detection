from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Detect fraudulent credit card transactions using machine learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
threshold = 0.5

feature_names = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]

@app.on_event("startup")
def load_model():
    global model, threshold
    try:
        with open('artifacts/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('artifacts/threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
        print(f"Model loaded! Threshold: {threshold:.2f}")
    except FileNotFoundError:
        raise RuntimeError("Model file not found. Run preprocessing.py first.")

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float
    V7: float; V8: float; V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float; V17: float; V18: float
    V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

class PredictionResponse(BaseModel):
    prediction: int
    fraud: bool
    probability: float
    message: str

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Fraud detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        txn_dict = transaction.model_dump()
    except AttributeError:
        txn_dict = transaction.dict()
    data = pd.DataFrame([txn_dict], columns=feature_names)

    try:
        probability = model.predict_proba(data)[0][1]
        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "fraud": bool(prediction == 1),
            "probability": float(probability),
            "message": "Fraudulent transaction detected" if prediction else "Legitimate transaction"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Real fraud sample from dataset
@app.get("/fraud_sample")
def fraud_sample():
    return {
        "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
        "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292,
        "V7": 0.239598554061257, "V8": 0.0986979012610507, "V9": 0.363786969611213,
        "V10": 0.0907941719789316, "V11": -0.551599533260813, "V12": -0.617800855762348,
        "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
        "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801985591,
        "V19": 0.403992960255733, "V20": 0.251412098239705, "V21": -0.018306777944153,
        "V22": 0.277837575558899, "V23": -0.110473910188767, "V24": 0.0669280749146731,
        "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 0.133558376740387,
        "V28": -0.0210530534538215
    }