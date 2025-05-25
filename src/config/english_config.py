# configs/english_config.py
import os
from .config_utils import BaseConfig

class EnglishConfig(BaseConfig):
    """Configuration for english sentiment analysis model"""
    
    LANGUAGE = "english"
    
    # You can override any base settings here
    MODEL_NAME = "roberta-base"  # english language model
    MAX_LENGTH = 150  # If english needs longer sequences
    
    # Raw data paths
    RAW_DATA_DIR = os.path.join(BaseConfig.BASE_DATA_DIR, "raw", "english")
    RAW_TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
    RAW_VAL_PATH = os.path.join(RAW_DATA_DIR, "val.csv")
    RAW_TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
    
    # Processed data paths
    PROCESSED_DATA_DIR = os.path.join(BaseConfig.BASE_DATA_DIR, "processed", "english")
    PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    PROCESSED_VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    
    # Use processed data for training
    TRAIN_DATA_PATH = PROCESSED_TRAIN_PATH
    VAL_DATA_PATH = PROCESSED_VAL_PATH
    TEST_DATA_PATH = PROCESSED_TEST_PATH
    
    # Output directory
    OUTPUT_DIR = os.path.join(BaseConfig.BASE_OUTPUT_DIR, "english")