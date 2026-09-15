from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
from data_processor import DataProcessor


processor=DataProcessor()

class ModelTrainer:

        def model_training(self,X_train,X_test,y_train,y_test):

            
                # Model initialize karna
                self.xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

                # Model ko data par train karna
                self.xgb_model.fit(X_train, y_train)

                y_pred_XGB=self.xgb_model.predict(X_test)
                cm_XGB=confusion_matrix(y_test,y_pred_XGB)
                print(cm_XGB)
                acuracy_XGB=accuracy_score(y_test,y_pred_XGB)
                print(acuracy_XGB)

        def pickling(self):
               with open(r"models\model.pkl","wb")as file:
                       artifacts = {
                        "model": self.xgb_model,
                        "age_imputer": processor.age_imputer,
                        "encoder": processor.encoder,
                        "scaler": processor.ss
                        }
                       pickle.dump(artifacts,file)
                       print("Pickle file created Succesfully!")       
  
