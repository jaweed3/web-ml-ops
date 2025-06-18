import os
import urllib.request as request
from src.datascience import logger
import zipfile

import urllib.request as request
import os
import zipfile
from src.datascience import logger
from src.datascience.entity.config_entity import (DataIngestionConfig)

# Component Data Ingestion
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_dir,
                filename=self.config.local_data_file,
            )
            logger.info(f"{filename} downloaded! with following information: \n{headers}")
        else:
            logger.info(f"File Already exists!")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extract the zip file into data directory
        Function Returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)