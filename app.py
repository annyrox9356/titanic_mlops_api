from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
import pickle
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd

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
    age_imputer = artifacts["age_imputer"]
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]

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
          input_df = pd.DataFrame([{
            "Pclass": data.Pclass,
            "Age": data.Age,
            "SibSp": data.SibSp,
            "Parch": data.Parch,
            "Fare": data.Fare,
            "Sex": data.Sex,
            "Embarked": data.Embarked
            }])
          # 2. Age imputation
          input_df[['Age']] = age_imputer.transform(
                input_df[['Age']]
            )

            # 3. One Hot Encoding
          cat_cols = ['Sex', 'Embarked']

          encoded_cats = encoder.transform(
                input_df[cat_cols]
            )

          encoded_df = pd.DataFrame(
                encoded_cats,
                columns=encoder.get_feature_names_out(cat_cols)
            )

            # 4. Remove original categorical columns
          input_df = input_df.drop(columns=cat_cols)

            # 5. Add encoded columns
          input_df = pd.concat(
                [input_df.reset_index(drop=True), encoded_df],
                axis=1
            )

            # 6. Scaling
          input_df[['Age', 'Fare']] = scaler.transform(
                input_df[['Age', 'Fare']]
            )

          prediction=model.predict(input_df)

          final_output=int(prediction[0])

          return{
               "Passenger class":data.Pclass,
               "passenger Age":data.Age,
               "Ticket Fare":data.Fare,
               "Gender":data.Sex,
               "End result(survived or not)":final_output
          }      
     except Exception as err_msg:
          raise HTTPException(status_code=400, detail=f"during prediction error that came:{str(err_msg)}")
     
                   
    

