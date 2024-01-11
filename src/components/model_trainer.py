import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_filepath: str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],#Take everything except the last column
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regression" : KNeighborsRegressor(),
                "Adaboost Regression" : AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score<=0.6:
                raise CustomException("No best model found!")
            logging.info("Found best model on both training and test dataset")
            save_object(file_path=self.model_trainer_config.trained_model_filepath,obj=best_model)

            predicted = best_model.predict(X_test)
            r2_scr = r2_score(y_test,predicted)

            return r2_scr
        
        except Exception as e:
            raise CustomException(e,sys)