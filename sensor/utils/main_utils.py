import os
import sys
import yaml
import dill
import numpy as np
from sensor.exception import SensorException
from sensor.logger import logging

def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise SensorException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise SensorException(e, sys)

def save_array(file_path: str, array: np.array) -> None:
    """
    This function takes file path and an array.
    Saves numpy array data to file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)
    
    except Exception as e:
        raise SensorException(e, sys)
    
def load_array(file_path: str) -> np.array:
    """
    This function takes in the file path.
    Returns: the numpy array saved in the file path.
    """
    try:
        with open(file_path, "rb") as file:
            return np.load(file)
        
    except Exception as e:
        raise SensorException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    
    except Exception as e:
        raise SensorException(e, sys)
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file:
            return dill.load(file)
        
    except Exception as e:
        raise SensorException(e, sys)