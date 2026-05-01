from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_PATH = Path("data/pond_iot_2023.csv")
MODEL_PATH = Path("models/pond_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")

app = FastAPI(title="Pond Water Quality API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def score_quality(pH: float, TDS: float, temp: float) -> int:
    # pH scoring
    if 6.8 <= pH <= 8.2:
        pH_score = 2
    elif 6.0 <= pH < 6.8 or 8.2 < pH <= 8.6:
        pH_score = 1
    else:
        pH_score = 0

    # TDS scoring
    if TDS <= 450:
        tds_score = 2
    elif 450 < TDS <= 600:
        tds_score = 1
    else:
        tds_score = 0

    # temperature scoring
    if 23 <= temp <= 28:
        temp_score = 2
    elif 20 <= temp < 23 or 28 < temp <= 32:
        temp_score = 1
    else:
        temp_score = 0

    total = pH_score + tds_score + temp_score

    if total >= 5:
        return 2   # Good
    elif total >= 3:
        return 1   # Moderate
    return 0       # Bad

def label_text(label: int) -> str:
    return ["Bad", "Moderate", "Good"][label]

def advice_text(label: int) -> str:
    if label == 2:
        return "Water quality is good. No action needed."
    if label == 1:
        return "Water quality is moderate. Monitor and consider mild treatment."
    return "Water quality is bad. Immediate action/treatment is required."

class RecordCreate(BaseModel):
    water_pH: float
    TDS: float
    water_temp: float

class Record(BaseModel):
    id: int
    created_date: str
    water_pH: float
    TDS: float
    water_temp: float
    label: int
    label_text: str

records_db: List[dict] = []

model: Optional[object] = None
scaler: Optional[object] = None


def record_from_internal(index: int, data: dict) -> Record:
    """Internal dict -> Record (id = index)"""
    return Record(
        id=index +1,
        created_date=data["created_date"],
        water_pH=data["water_pH"],
        TDS=data["TDS"],
        water_temp=data["water_temp"],
        label=data["label"],
        label_text=data["label_text"],
    )


def load_initial_data():
    global records_db

    if not DATA_PATH.exists():
        records_db = []
        return

    raw = pd.read_csv(DATA_PATH)

    water_pH_list = raw["water_pH"].tolist()
    TDS_list = raw["TDS"].tolist()
    temp_list = raw["water_temp"].tolist()

    records_db = []

    for p, tds, temp in zip(water_pH_list, TDS_list, temp_list):
        lbl = score_quality(p, tds, temp)
        records_db.append(
            {
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "water_pH": float(p),
                "TDS": float(tds),
                "water_temp": float(temp),
                "label": lbl,
                "label_text": label_text(lbl),
            }
        )

def load_model_and_scaler():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

    try:
        scaler_local = joblib.load(SCALER_PATH)
        scaler = scaler_local
    except Exception:
        scaler = None

@app.on_event("startup")
def on_startup():
    load_model_and_scaler()

@app.post("/predict")
def predict(sample: RecordCreate):
    
    rule_label = score_quality(sample.water_pH, sample.TDS, sample.water_temp)

    model_label = rule_label
    if model is not None and scaler is not None:
        X = np.array([[sample.water_pH, sample.TDS, sample.water_temp]])
        X_scaled = scaler.transform(X)
        model_label = int(model.predict(X_scaled)[0])

    final_label = rule_label  

    return {
        "label": final_label,
        "label_text": label_text(final_label),
        "advice": advice_text(final_label),
        "rule_label": rule_label,
        "model_label": model_label,
    }

# CRUD ENDPOINTS
@app.get("/records", response_model=List[Record])
def get_records():
    return [record_from_internal(i, r) for i, r in enumerate(records_db)]

@app.get("/records/{record_id}", response_model=Record)
def get_record(record_id: int):
    if 0 <= record_id < len(records_db):
        return record_from_internal(record_id, records_db[record_id])
    raise HTTPException(status_code=404, detail="Record not found")

@app.post("/records", response_model=Record)
def create_record(rec: RecordCreate):
    lbl = score_quality(rec.water_pH, rec.TDS, rec.water_temp)
    internal = {
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "water_pH": rec.water_pH,
        "TDS": rec.TDS,
        "water_temp": rec.water_temp,
        "label": lbl,
        "label_text": label_text(lbl),
    }
    records_db.append(internal)
    new_index = len(records_db) - 1
    return record_from_internal(new_index, internal)

@app.put("/records/{record_id}", response_model=Record)
def update_record(record_id: int, rec: RecordCreate):
    if not (0 <= record_id < len(records_db)):
        raise HTTPException(status_code=404, detail="Record not found")

    lbl = score_quality(rec.water_pH, rec.TDS, rec.water_temp)
    internal = records_db[record_id]
    internal.update(
        {
            "water_pH": rec.water_pH,
            "TDS": rec.TDS,
            "water_temp": rec.water_temp,
            "label": lbl,
            "label_text": label_text(lbl),
        }
    )
    return record_from_internal(record_id, internal)

@app.delete("/records/{record_id}")
def delete_record(record_id: int):
    if not (0 <= record_id < len(records_db)):
        raise HTTPException(status_code=404, detail="Record not found")
    del records_db[record_id]
    return {"detail": "Record deleted"}
