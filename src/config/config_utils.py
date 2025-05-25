class BaseConfig:
    """Base configuration class with common settings"""
    
    # Model settings
    MODELS = ['xlm-roberta-base', 'bert-base-multilingual-cased', 'distilbert/distilbert-base-multilingual-cased', 'roberta-base', 'ai4bharat/indic-bert', 'Shushant/nepaliBERT']
    MODEL_NAME = "xlm-roberta-base"  # Multilingual model
    NUM_LABELS = 3
    MAX_LENGTH = 128
    CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 5
    WARMUP_RATIO = 0.1
    EARLY_STOPPING_PATIENCE = 4
    
    # Column names (common across datasets)
    TEXT_COLUMN = "review"
    LABEL_COLUMN = "label"
    
    # Base paths
    BASE_OUTPUT_DIR = "model"
    BASE_DATA_DIR = "data"
    BASE_RESULT_DIR = "result"