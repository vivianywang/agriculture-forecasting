import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.api.prediction import PredictionEngine

app = FastAPI(title="AI Wheat Forecaster for Canada")
engine = PredictionEngine()


class PredictionRequest(BaseModel):
    city: str
    year: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return engine.predict(request.city, request.year)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))