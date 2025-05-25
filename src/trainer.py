import os
import numpy as np
import torch
import pandas as pd
from transformers import get_linear_schedule_with_warmup

# Use one of these alternatives:
try:
    from transformers.optimization import AdamW
except ImportError:
    # For newer transformers versions
    from torch.optim import AdamW
    print("Using AdamW from torch.optim")

# Import functions from other files
from common.metrics_utils import compute_metrics, print_epoch_metrics
from train_utils import train_epoch, validate, save_best_model

def train_model(model, train_loader, val_loader, tokenizer, output_dir, 
               learning_rate=2e-5, weight_decay=0.01, epochs=5, warmup_ratio=0.1,
               early_stopping_patience=3, gradient_clip=1.0):
    """
    Train a transformer model for sentiment analysis with comprehensive metrics tracking.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer for text processing
        output_dir: Directory to save models and metrics
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        epochs: Number of training epochs
        warmup_ratio: Ratio of steps for learning rate warmup
        early_stopping_patience: Number of epochs to wait before early stopping
        gradient_clip: Maximum gradient norm for clipping
        
    Returns:
        model: The trained model
        metrics: Dictionary containing all tracked metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Get number of labels from model config
    num_labels = model.config.num_labels
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
    
    # Calculate class weights for imbalanced data
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    
    # Get class counts and compute weights
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
    class_weights = class_weights.to(device)
    print("Class distribution:", class_counts)
    print("Class weights:", class_weights)
    
    # Define class names for easier reference
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize metrics tracking
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    # Add class-wise metrics tracking
    for class_name in class_names:
        lower_name = class_name.lower()
        for prefix in ['train', 'val']:
            for metric in ['f1', 'precision', 'recall', 'accuracy']:
                metrics[f'{prefix}_{lower_name}_{metric}'] = []
    
    # Early stopping setup
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, scheduler, class_weights, device, num_labels
        )
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, device)
        
        # Compute metrics
        train_metrics, _ = compute_metrics(train_labels, train_preds, class_names)
        val_metrics, _ = compute_metrics(val_labels, val_preds, class_names)
        
        # Add loss to metrics
        train_metrics['loss'] = train_loss
        val_metrics['loss'] = val_loss
        
        # Update metrics tracking
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        
        print("train_metrics keys:", train_metrics.keys())
        print("metrics keys:", metrics.keys())

        # Update all other metrics
        for key, value in train_metrics.items():
            if key != 'loss':  # Already added loss above
                #metrics[f'train_{key}'].append(value)
                if f'train_{key}' not in metrics:
                    metrics[f'train_{key}'] = []
                metrics[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            if key != 'loss':  # Already added loss above
                #metrics[f'val_{key}'].append(value)
                if f'val_{key}' not in metrics:
                    metrics[f'val_{key}'] = []
                metrics[f'val_{key}'].append(value)
        
        # Print metrics
        print_epoch_metrics(epoch, epochs, train_metrics, val_metrics, class_names)
        
        # Save metrics to CSV
        pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        
        # Check for early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # Save best model
            save_best_model(model, tokenizer, optimizer, metrics, val_metrics, epoch, output_dir, class_names)
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Print final summary
    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {metrics['val_accuracy'][-1]:.4f}")
    print(f"Final validation F1 (weighted): {metrics['val_f1'][-1]:.4f}")
    
    print("\nFinal class-wise F1 scores:")
    for class_name in class_names:
        lower_name = class_name.lower()
        print(f"{class_name:<10} {metrics[f'val_{lower_name}_f1'][-1]:.4f}")
    
    print("\nFinal class-wise Accuracy:")
    for class_name in class_names:
        lower_name = class_name.lower()
        print(f"{class_name:<10} {metrics[f'val_{lower_name}_accuracy'][-1]:.4f}")
    
    return model, metrics