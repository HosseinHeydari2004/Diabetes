from pydantic import BaseModel
from fastapi import FastAPI
import joblib
from numpy import array

app = FastAPI()

class DiabetesData(BaseModel):
    chol: float
    stab_glu: float
    hdl: float
    ratio: float
    glyhb: float
    age: int
    height: float
    weight: float
    bp_1s: float
    bp_1d: float
    waist: float
    hip: float
    Bmi: float
    gender_female: bool
    gender_male: bool

model = joblib.load("my_model.joblib")

@app.post("/predict")
def predict(data: DiabetesData):
    features = [[
    data.chol,
    data.stab_glu,
    data.hdl,
    data.ratio,
    data.glyhb,
    data.age,
    data.height,
    data.weight,
    data.bp_1s,
    data.bp_1d,
    data.waist,
    data.hip,
    data.Bmi,
    int(data.gender_female),
    int(data.gender_male)
    ]]
    pred = model.predict(array(features))
    return {
        "prediction":int(pred[0])
    }
    
