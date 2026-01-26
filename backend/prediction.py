from fastapi import FastAPI
:q
:q
from pydantic import BaseModel
import numpy as np

from src.datascience.entity.config_entity import PredictionConfig
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()


