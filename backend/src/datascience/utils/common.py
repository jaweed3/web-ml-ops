import os
import yaml
import json
import sys
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.datascience import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and returns
    
    Args:
        path_to_yaml (str): path like input
        
    Raises:
        ValueError: if yaml file is empty
        e: empty file
        
    Returns:
        ConfigBox: ConfigBox Type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully...")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml File is Empty")
    
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories
    
    Args:
        path_to_directories (lsit): list os path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Create directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved as json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON files data
    
    Args:
        path (Path): path to json file
    
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f'JSON file loaded sucessfully from: {path}')
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary file
    
    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary data
    
    Args:
        path (Path): path to binary file
        
    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data