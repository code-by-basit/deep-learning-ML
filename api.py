from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from regression import MLP, scalar 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
model.eval() 

app = FastAPI(title="Property Price Prediction Engine")

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

class PredictionResponse(BaseModel):
    input_features: list[float]
    predicted_target: float
    status: str


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        raw_features = [[request.feature1, request.feature2, request.feature3]]        
        scaled_features = scalar.transform(raw_features)
        
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            raw_prediction = model(input_tensor)
            final_score = raw_prediction.cpu().item()
        
        return {
            "input_features": raw_features[0],
            "predicted_target": round(final_score, 4),
            "status": "SUCCESS"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Error: {str(e)}")
    

