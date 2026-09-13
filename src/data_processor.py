import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:

    def __init__(self):
        pass

    def loader(self,filename:str):
        cols_to_drop=['PassengerId','Name','Ticket','Cabin']
        data=pd.read_csv(filename)
        data = data.dropna(subset=['Embarked']).reset_index(drop=True)
        self.X=data.drop(["Survived"] + cols_to_drop, axis=1)
        self.y=data["Survived"]
        

    def processor(self):
        # droping blank row in embarked
        self.X.dropna(subset=['Embarked'], inplace=True)       

        #filling age values with median
        age_imputer=SimpleImputer(strategy='median')
        self.X[['Age']]=age_imputer.fit_transform(self.X[['Age']])

        #Charecter Encoding
        cat_cols = ['Sex', 'Embarked']
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        encoded_cats = encoder.fit_transform(self.X[cat_cols])


        # Convert the encoded array into a DataFrame
        encoded_df = pd.DataFrame(
                                    encoded_cats,
                                    columns=encoder.get_feature_names_out(cat_cols)
                                )
        # Drop the original columns
        self.X = self.X.drop(columns=cat_cols).reset_index(drop=True)

        # Add the encoded columns
        self.X = pd.concat([self.X, encoded_df], axis=1)

        # print(self.X.head())

        #Feature Scaling
        cols_to_feature=['Age','Fare']
        ss=StandardScaler()
        self.X[cols_to_feature]=ss.fit_transform(self.X[cols_to_feature])
        print(self.X.head())

        #Training model on XGBoost 

        #Train test split
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=0.25,random_state=0)

        return X_train,X_test,y_train,y_test
 


    
    