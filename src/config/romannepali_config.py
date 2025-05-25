# configs/romannepali_config.py
import os
from .config_utils import BaseConfig

class RomanNepaliConfig(BaseConfig):
    """Configuration for Roman Nepali sentiment analysis model"""
    
    LANGUAGE = "roman_nepali"
    
    # You can override any base settings here
    MODEL_NAME = 'bert-base-multilingual-cased' #"xlm-roberta-base"  # Multilingual model
    MAX_LENGTH = 150  # If Nepali needs longer sequences
    
    # Raw data paths
    RAW_DATA_DIR = os.path.join(BaseConfig.BASE_DATA_DIR, "raw", "roman_nepali")
    RAW_TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
    RAW_VAL_PATH = os.path.join(RAW_DATA_DIR, "val.csv")
    RAW_TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
    
    # Processed data paths
    PROCESSED_DATA_DIR = os.path.join(BaseConfig.BASE_DATA_DIR, "processed", "roman_nepali")
    PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    PROCESSED_VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    
    # Use processed data for training
    TRAIN_DATA_PATH = PROCESSED_TRAIN_PATH
    VAL_DATA_PATH = PROCESSED_VAL_PATH
    TEST_DATA_PATH = PROCESSED_TEST_PATH
    
    # Output directory
    OUTPUT_DIR = os.path.join(BaseConfig.BASE_OUTPUT_DIR, "roman_nepali")