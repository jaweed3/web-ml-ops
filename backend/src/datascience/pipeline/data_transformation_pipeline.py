from pathlib import Path
from src.datascience.config.configuration import ConfigurationManager
from src.datascience import logger
from src.datascience.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
            if status == 'True':
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_dataset()
            else:
                raise Exception("Your Data Schema is Not Valid")
        except Exception as e:
            raise e