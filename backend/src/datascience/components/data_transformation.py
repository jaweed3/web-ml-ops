import os
import pandas as pd
from src.datascience import logger
from sklearn.model_selection import train_test_split
from src.datascience.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        # We can add more different data transformation progress.
        # Like Scaler, PCA, and another EDA processes to the data

    def train_test_dataset(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25)
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info(f"Splitted data into training and test dataset")
        logger.info(f"Training dataset shape : {train.shape}")
        logger.info(f"Training dataset shape : {train.shape}")