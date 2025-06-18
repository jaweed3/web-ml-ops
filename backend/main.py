from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.datascience.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} is Started <<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>> Stage {STAGE_NAME} is Completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} is Started <<<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>> Stage {STAGE_NAME} is Completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} is Started <<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.initiate_model_training_pipeline()
    logger.info(f">>>>> Stage {STAGE_NAME} is Completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} is Started <<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.initiate_model_evaluation_pipeline()
    logger.info(f">>>>> Stage {STAGE_NAME} is Completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e