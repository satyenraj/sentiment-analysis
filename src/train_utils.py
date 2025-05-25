import os
import numpy as np
import torch
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, scheduler, class_weights, device, num_labels):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        class_weights: Weights for handling class imbalance
        device: Device to train on (cuda/cpu)
        num_labels: Number of classification classes
        
    Returns:
        train_loss: Average training loss for the epoch
        train_preds: List of predictions
        train_labels: List of true labels
    """
    model.train()
    train_losses = []
    train_preds, train_labels = [], []
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        train_losses.append(loss.item())
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Using default clip value
        optimizer.step()
        scheduler.step()
        
        # Collect predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    train_loss = np.mean(train_losses)
    
    return train_loss, train_preds, train_labels


def validate(model, val_loader, device):
    """
    Validate the model on validation data.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to validate on (cuda/cpu)
        
    Returns:
        val_loss: Average validation loss
        val_preds: List of predictions
        val_labels: List of true labels
    """
    model.eval()
    val_losses = []
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            val_losses.append(loss.item())
            
            # Collect predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    val_loss = np.mean(val_losses)
    
    return val_loss, val_preds, val_labels


def save_best_model(model, tokenizer, optimizer, metrics, val_metrics, epoch, output_dir, class_names):
    """
    Save the best model based on validation performance.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        optimizer: The optimizer state to save
        metrics: Overall metrics dictionary
        val_metrics: Current validation metrics
        epoch: Current epoch number
        output_dir: Directory to save the model
        class_names: Names of the classes
    """
    # Save best model
    model_path = os.path.join(output_dir, "best_model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Also save in PyTorch format
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_f1': val_metrics['f1'],
        'class_f1': {
            class_name: val_metrics[f'{class_name.lower()}_f1']
            for class_name in class_names
        }
    }, os.path.join(output_dir, "best_model.pt"))
    
    print(f"Saved best model at epoch {epoch+1}")