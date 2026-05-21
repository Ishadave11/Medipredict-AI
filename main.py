from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import sqlite3

import datetime

app = FastAPI(title="Diabetes Clinical API")

# Load model
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")


# ----------------------
# DATABASE
# ----------------------
conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    glucose REAL,
    bmi REAL,
    age REAL,
    risk REAL
)
""")
conn.commit()


# ----------------------
# REQUEST FORMAT
# ----------------------
class Patient(BaseModel):
    glucose: float
    bp: float
    skin: float
    insulin: float
    bmi: float
    dpf: float
    age: float


# ----------------------
# PREDICTION ENDPOINT
# ----------------------
@app.post("/predict")
def predict(patient: Patient):

    data = np.array([[
        patient.glucose,
        patient.bp,
        patient.skin,
        patient.insulin,
        patient.bmi,
        patient.dpf,
        patient.age
    ]])

    data = imputer.transform(data)
    data = scaler.transform(data)

    prob = model.predict_proba(data)[0][1]
    risk = float(prob * 100)

    # Save to database
    cursor.execute("""
        INSERT INTO predictions (timestamp, glucose, bmi, age, risk)
        VALUES (?, ?, ?, ?, ?)
    """, (
        str(datetime.datetime.now()),
        patient.glucose,
        patient.bmi,
        patient.age,
        risk
    ))
    conn.commit()

    return {
        "risk_score": risk,
        "risk_level": "HIGH" if risk > 70 else "MODERATE" if risk > 30 else "LOW"
    }
