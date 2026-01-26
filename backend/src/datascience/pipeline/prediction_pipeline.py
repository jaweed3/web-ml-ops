import joblib 
import os
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
import dagshub
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
from pathlib import Path
from src.datascience.entity.config_entity import PredictionConfig

env_path = Path(__file__).parents[3] / ".env"
load_dotenv(env_path)
dagshub_repo_name = os.environ.get("DAGSHUB_REPO_NAME")
dagshub_owner_name = os.environ.get("DAGSHUB_REPO_OWNER")

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        cwd = os.getcwd()
        logger.info(f"[DEBUG] Current Working Directory: {cwd}")
        logger.info(f"[DEBUG] Looking for local model at: {os.path.join(cwd, self.config.model_path)}")
        if getattr(self.config, "mlflow_model_name", None):
            try:
                dagshub.init(repo_owner=dagshub_owner_name, repo_name=dagshub_repo_name, mlflow=True)
                logger.info(f"attempting to load model from MLflow Registry. => {self.config.mlflow_model_name}")
                client = MlflowClient()
                filter_string = f"name='{self.config.mlflow_model_name}'"
                results = client.search_model_versions(filter_string)

                if results:
                    latest_version_obj = max(results,key=lambda x: int(x.version))
                    version = latest_version_obj.version
                    logger.info(f"Found version {version}.loading from remote repository!")
                    return load_model(model_uri=f"models:/{self.config.mlflow_model_name}/{version}")
                else:
                    logger.warning(f"Model {self.config.mlflow_model_name} not found in registry!")
            except Exception as e:
                logger.error(f"failed to connect to MLflow Registry: {str(e)}. Switching to local model fallback.")

        if getattr(self.config, "model_path", None):
            if os.path.exists(self.config.model_path):
                logger.info(f"loading local model from {self.config.model_path}")
                return joblib.load(self.config.model_path)
            else:
                raise FileNotFoundError(f"CRITICAL: Local model not found at {self.config.model_path}. Prediction Impossible")
        else:
            raise ValueError("No valid model configuration found.")
        
    def predict(self, data: np.array):
        """
        Perform prediction using MLflow PyFunc loaded model.
        
        Args:
            data: 2D numpy array, shape (1, expected_features)
            
        Return:
            prediction: np.array or list
        """
        if self.model is None:
            logger.error("The model is not loaded. cannot perform prediction")
            raise RuntimeError("Prediction model is not loaded. please check the model loading error.")
        
        logger.info(f"Received data shape for prediction :{data.shape}")

        expected_features = self.config.expected_features
        if data.ndim != 2 or data.shape[1] != expected_features:
            logger.error(
                f"input data has invalid shape. Expected(1, {expected_features})"
                f"But received data shape {data.shape}. Data dimensions {data.ndim}"
            )
            raise ValueError(
                f"Input data must be 2D numpy array with {expected_features} features"
                f"received shape {data.shape}"
            )

        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age"
        ]
        df = pd.DataFrame(data, columns=self.config.feature_names)
        
        try:
            prediction = self.model.predict(df)
            logger.info(f"Prediction performed successfully, got result {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error during prediction by the loaded moel: {str(e)}", exc_info=True)
            raise Exception(f"prediction failed: {str(e)}")
