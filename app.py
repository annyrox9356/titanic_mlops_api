from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
import pickle
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd
from src.transforming_inputs import TransformInputs

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    with open("models/model.pkl", "rb") as file:
        artifacts = pickle.load(file)

    model = artifacts["model"]
    transformer=TransformInputs(age_imputer = artifacts["age_imputer"],encoder = artifacts["encoder"],scaler = artifacts["scaler"])

except FileNotFoundError:
    model=None
    print("warning! model.pkl file not found")


class TitanicInput(BaseModel):
        Pclass: Literal[1, 2, 3] = Field(..., description="Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)")
        Age: float = Field(..., gt=0, description="Age in years")
        SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
        Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
        Fare: float = Field(..., ge=0, description="Passenger fare")
        Sex: Literal["male", "female"] = Field(..., description="Passenger gender")
        Embarked: Literal["C", "Q", "S"] = Field(..., description="Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)")

@app.post("/prediction-survival") 
def predict_survival(data:TitanicInput): 
     if model==None:
          raise HTTPException(status_code=500,detail="model is not available on server")

     try:
          # 1. Raw incoming data
          input_df = pd.DataFrame([data.model_dump()])

          trans_input=transformer.transform(input_df)

          prediction=model.predict(trans_input)

          final_output=int(prediction[0])

          return{
               "Passenger class":data.Pclass,
               "passenger Age":data.Age,
               "Ticket Fare":data.Fare,
               "Gender":data.Sex,
               "End result(survived or not)": "Survived" if final_output==1 else "Not survived"
          }      
     except Exception as err_msg:
          raise HTTPException(status_code=400, detail=f"during prediction error that came:{str(err_msg)}")
     
                   
    

