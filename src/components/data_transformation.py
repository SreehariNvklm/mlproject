# To do feature engineering, data cleaning, on categorical as well as numerical features
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Imputer handles missing values and data on dataset
from sklearn.preprocessing import OneHotEncoder,StandardScaler # Scaling

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    '''
    This function is responsible for data transformation    
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_features = ["writing_score","reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical Columns scaling Completed")
            logging.info("Categorical Columns Encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_col = 'math_score'

            X_train = train_df.drop(target_col,axis=1)
            y_train = train_df[target_col]

            X_test = test_df.drop(target_col,axis=1)
            y_test = test_df[target_col]

            logging.info("Applying preprocessing object on training and test datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(X_train)
            input_feature_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[input_feature_train_arr,np.array(y_train)]
            test_arr = np.c_[input_feature_test_arr,np.array(y_test)]

            logging.info('Saved Preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)