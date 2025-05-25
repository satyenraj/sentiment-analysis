import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

def compute_metrics(true_labels, predictions, class_names):
    """
    Compute comprehensive classification metrics.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        class_names: Names of the classes
        
    Returns:
        Dictionary of metrics and classification report dictionary
    """
    # Overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    # Detailed classification report
    report = classification_report(
        true_labels, 
        predictions, 
        labels=range(len(class_names)), 
        target_names=class_names, 
        output_dict=True
    )
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        # Find indices where true label is this class
        class_mask = np.array(true_labels) == i
        if np.sum(class_mask) > 0:
            class_accuracy[class_name] = np.mean(np.array(predictions)[class_mask] == i)
        else:
            class_accuracy[class_name] = 0.0
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1': f1_weighted,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }
    
    # Add class-specific metrics
    for class_name in class_names:
        metrics[f'{class_name.lower()}_f1'] = report[class_name]['f1-score']
        metrics[f'{class_name.lower()}_precision'] = report[class_name]['precision']
        metrics[f'{class_name.lower()}_recall'] = report[class_name]['recall']
        metrics[f'{class_name.lower()}_accuracy'] = class_accuracy[class_name]
        metrics[f'{class_name.lower()}_support'] = report[class_name]['support']
    
    return metrics, report


def print_epoch_metrics(epoch, epochs, train_metrics, val_metrics, class_names):
    """
    Print metrics for the current epoch.
    
    Args:
        epoch: Current epoch number
        epochs: Total number of epochs
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        class_names: Names of the classes
    """
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['f1_weighted']:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_weighted']:.4f}")
    
    # Print class-wise F1 and accuracy
    print("\nClass-wise F1 scores:")
    for class_name in class_names:
        lower_name = class_name.lower()
        print(f"{class_name:<10} Train: {train_metrics[f'{lower_name}_f1']:.4f}, Val: {val_metrics[f'{lower_name}_f1']:.4f}")
    
    print("\nClass-wise Accuracy:")
    for class_name in class_names:
        lower_name = class_name.lower()
        print(f"{class_name:<10} Train: {train_metrics[f'{lower_name}_accuracy']:.4f}, Val: {val_metrics[f'{lower_name}_accuracy']:.4f}")


def print_test_metrics(test_metrics, class_names):
    """
    Print metrics for test set evaluation.
    
    Args:
        test_metrics: Test metrics dictionary
        class_names: Names of the classes
    """
    print("\n===== TEST SET EVALUATION RESULTS =====")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['f1_macro']:.4f}, Precision: {test_metrics['precision_macro']:.4f}, Recall: {test_metrics['recall_macro']:.4f}")
    print(f"Weighted F1: {test_metrics['f1_weighted']:.4f}, Precision: {test_metrics['precision_weighted']:.4f}, Recall: {test_metrics['recall_weighted']:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10} {'Support':<10}")
    print("-" * 60)
    for class_name in class_names:
        lower_name = class_name.lower()
        print(f"{class_name:<10} "
              f"{test_metrics[f'{lower_name}_precision']:<10.4f} "
              f"{test_metrics[f'{lower_name}_recall']:<10.4f} "
              f"{test_metrics[f'{lower_name}_f1']:<10.4f} "
              f"{test_metrics[f'{lower_name}_accuracy']:<10.4f} "
              f"{test_metrics[f'{lower_name}_support']:<10d}")


def calculate_confusion_matrix(true_labels, predictions, class_names):
    """
    Calculate confusion matrix.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        class_names: Names of the classes
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(true_labels, predictions, labels=range(len(class_names)))


def plot_confusion_matrix(cm, class_names, output_dir, title_suffix="", timestamp=None):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of the classes
        output_dir: Directory to save the plot
        title_suffix: Suffix for the plot title (e.g., "Test Set")
        timestamp: Timestamp for filename (optional)
        
    Returns:
        Path to saved confusion matrix image
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Create filename with suffix
    suffix = title_suffix.lower().replace(" ", "_")
    if suffix:
        cm_path = os.path.join(output_dir, f"confusion_matrix_{suffix}_{timestamp}.png")
    else:
        cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_path


def calculate_and_plot_curves(true_labels, logits, class_names, output_dir, title_suffix="", timestamp=None):
    """
    Calculate and plot ROC and PR curves for multi-class classification.
    
    Args:
        true_labels: Ground truth labels
        logits: Model logits for each class
        class_names: Names of the classes
        output_dir: Directory to save the plots
        title_suffix: Suffix for the plot titles (e.g., "Test Set")
        timestamp: Timestamp for filenames (optional)
        
    Returns:
        Dictionary with curve metrics
    """
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Create filename suffix
    suffix = title_suffix.lower().replace(" ", "_")
    if suffix:
        suffix = f"_{suffix}"
    
    metrics = {}
    
    # Binarize the labels for multi-class ROC
    y_bin = label_binarize(true_labels, classes=range(len(class_names)))
    
    # ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(12, 10))
    
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(
            fpr[i], 
            tpr[i], 
            lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.4f})'
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve {title_suffix}')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, f"roc_curve{suffix}_{timestamp}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and store AUC values
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name.lower()}_auc'] = roc_auc[i]
    
    # Precision-Recall curve
    plt.figure(figsize=(12, 10))
    
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i, class_name in enumerate(class_names):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], logits[:, i])
        avg_precision[i] = average_precision_score(y_bin[:, i], logits[:, i])
        
        plt.plot(
            recall[i], 
            precision[i], 
            lw=2,
            label=f'{class_name} (AP = {avg_precision[i]:.4f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {title_suffix}')
    plt.legend(loc="best")
    
    pr_path = os.path.join(output_dir, f"precision_recall_curve{suffix}_{timestamp}.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add average precision to metrics
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name.lower()}_avg_precision'] = avg_precision[i]
    
    return metrics


def save_metrics_to_csv(metrics, output_dir, filename_prefix="metrics", timestamp=None):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Directory to save the CSV
        filename_prefix: Prefix for the filename
        timestamp: Timestamp for filename (optional)
        
    Returns:
        Path to saved CSV file
    """
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    return csv_path


def save_predictions(true_labels, predictions, logits, class_names, output_dir, filename_prefix="predictions", timestamp=None):
    """
    Save predictions and logits to CSV for further analysis.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        logits: Model logits
        class_names: Names of the classes
        output_dir: Directory to save the CSV
        filename_prefix: Prefix for the filename
        timestamp: Timestamp for filename (optional)
        
    Returns:
        Path to saved predictions CSV
    """
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame
    predictions_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions
    })
    
    # Add logits for each class
    for i, class_name in enumerate(class_names):
        predictions_df[f'logit_{class_name.lower()}'] = logits[:, i]
    
    # Add class names for readability
    predictions_df['true_class'] = predictions_df['true_label'].apply(lambda x: class_names[x])
    predictions_df['predicted_class'] = predictions_df['predicted_label'].apply(lambda x: class_names[x])
    
    # Save to CSV
    predictions_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    return predictions_path


def evaluate_test_set(true_labels, predictions, logits, class_names, output_dir, timestamp=None):
    """
    Comprehensive evaluation of test set, with metrics calculation, visualization, and saving.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        logits: Model logits
        class_names: Names of the classes
        output_dir: Directory to save results
        timestamp: Timestamp for filenames (optional)
        
    Returns:
        Dictionary with all test metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate metrics
    test_metrics, report = compute_metrics(true_labels, predictions, class_names)
    
    # Print metrics
    print_test_metrics(test_metrics, class_names)
    
    # Calculate and plot confusion matrix
    cm = calculate_confusion_matrix(true_labels, predictions, class_names)
    cm_path = plot_confusion_matrix(
        cm, class_names, output_dir, "Test Set", timestamp
    )
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Calculate and plot ROC and PR curves
    curve_metrics = calculate_and_plot_curves(
        true_labels, logits, class_names, output_dir, "Test Set", timestamp
    )
    # Add curve metrics to test metrics
    test_metrics.update(curve_metrics)
    
    # Save metrics to CSV
    metrics_path = save_metrics_to_csv(
        test_metrics, output_dir, "test_metrics", timestamp
    )
    print(f"Test metrics saved to: {metrics_path}")
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, f"classification_report_test_{timestamp}.csv")
    report_df.to_csv(report_path)
    print(f"Classification report saved to: {report_path}")
    
    # Save predictions
    predictions_path = save_predictions(
        true_labels, predictions, logits, class_names, output_dir, "test_predictions", timestamp
    )
    print(f"Test predictions saved to: {predictions_path}")
    
    return test_metrics