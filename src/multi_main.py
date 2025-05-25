import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared training function
from trainer import train_model

# Import language-specific components
from common.sentiment_dataset import SentimentDataset
from config.multi_config import MultiConfig

def main():
    # Load configuration for Multilingual
    config = MultiConfig()
    
    # Create output directory with language identifier
    #output_dir = config.OUTPUT_DIR #os.path.join(config.BASE_OUTPUT_DIR, "multi")
    output_dir = os.path.join(config.OUTPUT_DIR, "mBERT")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer - choose appropriate multilingual model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )
    
    # Load datasets
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)
    
    # Create datasets
    train_dataset = SentimentDataset(
        texts=train_df[config.TEXT_COLUMN].values,
        labels=train_df[config.LABEL_COLUMN].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = SentimentDataset(
        texts=val_df[config.TEXT_COLUMN].values,
        labels=val_df[config.LABEL_COLUMN].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Train the model
    model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        output_dir=output_dir,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        epochs=config.EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE #,
        #class_names=config.CLASS_NAMES
    )
    
    print(f"Training complete for {config.LANGUAGE} language!")

if __name__ == "__main__":
    main()