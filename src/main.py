from data_processor import DataProcessor
from model_trainer import ModelTrainer

processor=DataProcessor()
processor.loader(r"data\train.csv")
X_train,X_test,y_train,y_test=processor.processor()


trainer=ModelTrainer()
trainer.model_training(X_train,X_test,y_train,y_test)
trainer.pickling()

