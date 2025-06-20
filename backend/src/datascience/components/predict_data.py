import joblib 
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from src.datascience.entity.config_entity import PredictionConfig
from src.datascience.constant import *
from src.datascience.utils.common import save_json

logger = logging.getLogger(__name__)

class PredictionDataset:
    def __init__(self, config: PredictionConfig, data: np.array):
        self.config = config
        self.model = None
        self.data = data

        try:
            logger.info(f"Attempting to load model from: {self.config.model_path.resolve()}")
            self.model = joblib.load(self.config.model_path)
            logger.info("Model Loaded Successfully")
        except FileNotFoundError:
            logger.error(
                f"Error: Model not found at {self.config.model_path.resolve()}. "
                "Did you run the Training pipeline?"
            )
        except Exception as e:
            logger.error(
                f"An Unexpected error occured while loading the model from {self.config.model_path}"
                f"the error seems is: {str(e)}",
                exc_info=True
                )
            raise Exception(f"Failed to load prediction model: {str(e)}")
        
    def predict(self):
        """
        doing the prediction with loaded model
        
        Args:
            data: numpy array data non-null
            
        Return:
            prediction: result from the data prediction.
        """
        if self.model is None:
            logger.error("The model is not loaded. cannot perform prediction")
            raise RuntimeError("Prediction model is not loaded. please check the model loading error.")
        
        logger.info(f"Received data shape for prediction :{self.data.shape}")

        expected_features = 8
        if self.data.ndim != 2 or self.data.shape[1] != expected_features:
            logger.error(
                f"input data has invalid shape. Expected(1, {expected_features})"
                f"But received data shape {self.data.shape}. Data dimensions {self.data.ndim}"
            )
            raise ValueError(
                f"Input data must be 2D numpy array with {expected_features} features"
                f"received shape {self.data.shape}"
            )
        
        try:
            prediction = self.model.predict(self.data)
            logger.info(f"Prediction performed successfully, got result {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error during prediction by the loaded moel: {str(e)}", exc_info=True)
            raise Exception(f"prediction failed: {str(e)}")