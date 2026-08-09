from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Model.predict import predict_output, MODEL_VERSION, model
from pydantic import BaseModel,Field
from typing import Literal,Annotated


class SMSRequest(BaseModel):
    data: str

class APiResponse(BaseModel):
    result:Annotated[Literal[0,1], Field(..., description="The predicted insurance premium category")]
    probabilities: Annotated[dict [str, float], Field(..., description="Probabilities of each category")]

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the SMS Spam Prediction API. Use the /predict endpoint to get predictions."}

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION ,
        "model_loaded": model is not None
    }

@app.post("/predict",response_model=APiResponse) 
def predict_premium(request: SMSRequest):
    text =request.data

    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded")

    try:
        result = predict_output(text)
        return result
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    