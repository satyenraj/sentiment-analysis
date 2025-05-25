"""
Model Testing Module
===================

This module provides comprehensive testing functionality for trained sentiment classification models.
Supports multilingual models with English and Nepali text.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report)
from torch.utils.data import Dataset, DataLoader
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TestDataset(Dataset):
    """Custom dataset for testing"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelTester:
    """Class for comprehensive model testing"""
    
    def __init__(self, model_path, tokenizer_path=None, device=None):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to tokenizer (if different from model_path)
            device: Device to run inference on
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            print("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully!")
            print(f"Number of labels: {self.model.config.num_labels}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_test_data(self, test_data_path=None, test_df=None, text_column='text', label_column='label'):
        """
        Prepare test data for evaluation
        
        Args:
            test_data_path: Path to test CSV file
            test_df: DataFrame with test data (alternative to file path)
            text_column: Name of text column
            label_column: Name of label column
        """
        
        if test_df is not None:
            self.test_df = test_df
        elif test_data_path is not None:
            self.test_df = pd.read_csv(test_data_path)
        else:
            raise ValueError("Either test_data_path or test_df must be provided")
        
        print(f"Test data loaded: {len(self.test_df)} samples")
        print(f"Class distribution:\n{self.test_df[label_column].value_counts()}")
        
        # Extract texts and labels
        self.test_texts = self.test_df[text_column].tolist()
        self.test_labels = self.test_df[label_column].tolist()
        
        # Convert string labels to integers if necessary
        if isinstance(self.test_labels[0], str):
            self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(self.test_labels)))}
            self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
            self.test_labels = [self.label_mapping[label] for label in self.test_labels]
            print(f"Label mapping: {self.label_mapping}")
        else:
            # Assume labels are already integers
            unique_labels = sorted(set(self.test_labels))
            self.reverse_label_mapping = {idx: f"class_{idx}" for idx in unique_labels}
        
        # Create dataset
        self.test_dataset = TestDataset(
            texts=self.test_texts,
            labels=self.test_labels,
            tokenizer=self.tokenizer
        )
        
        return self.test_dataset
    
    def predict_batch(self, batch_size=32):
        """
        Make predictions on the test dataset
        
        Args:
            batch_size: Batch size for inference
            
        Returns:
            predictions, probabilities
        """
        
        print("Making predictions...")
        
        dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        
        print(f"Predictions completed for {len(self.predictions)} samples")
        
        return self.predictions, self.probabilities
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(self.test_labels, self.predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.test_labels, self.predictions, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            self.test_labels, self.predictions, average='macro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            self.test_labels, self.predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.test_labels, self.predictions)
        
        # Store results
        self.test_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix,
            'per_class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        }
        
        return self.test_results
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        
        # Get class names
        class_names = [self.reverse_label_mapping[i] for i in range(len(self.reverse_label_mapping))]
        
        # Generate report
        report = classification_report(
            self.test_labels, 
            self.predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        self.classification_report = report
        
        # Print formatted report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            self.test_labels, 
            self.predictions, 
            target_names=class_names,
            zero_division=0
        ))
        
        return report
    
    def analyze_misclassifications(self, num_examples=5):
        """Analyze misclassified examples"""
        
        # Find misclassified indices
        misclassified_idx = np.where(np.array(self.test_labels) != self.predictions)[0]
        
        print(f"\n" + "="*60)
        print(f"MISCLASSIFICATION ANALYSIS")
        print("="*60)
        print(f"Total misclassified: {len(misclassified_idx)} out of {len(self.test_labels)}")
        print(f"Misclassification rate: {len(misclassified_idx)/len(self.test_labels)*100:.2f}%")
        
        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return []
        
        # Show examples
        print(f"\nFirst {min(num_examples, len(misclassified_idx))} misclassified examples:")
        print("-" * 60)
        
        """ for i, idx in enumerate(misclassified_idx[:num_examples]):
            true_label = self.reverse_label_mapping[self.test_labels[idx]]
            pred_label = self.reverse_label_mapping[self.predictions[idx]]
            confidence = self.probabilities[idx][self.predictions[idx]]
            text = self.test_texts[idx][:100] + "..." if len(self.test_texts[idx]) > 100 else self.test_texts[idx]
            
            print(f"\nExample {i+1}:")
            print(f"Text: {text}")
            print(f"True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.3f}") """
        
        return misclassified_idx
    
    def test_individual_samples(self, samples):
        """
        Test individual text samples
        
        Args:
            samples: List of text strings to test
        """
        
        print(f"\n" + "="*60)
        print("INDIVIDUAL SAMPLE TESTING")
        print("="*60)
        
        results = []
        
        for i, text in enumerate(samples):
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)
            
            pred_label = self.reverse_label_mapping[prediction.item()]
            confidence = probabilities[0][prediction].item()
            
            print(f"\nSample {i+1}:")
            print(f"Text: {text}")
            print(f"Predicted: {pred_label}")
            print(f"Confidence: {confidence:.3f}")
            
            # Show all class probabilities
            print("All class probabilities:")
            for j, (class_idx, prob) in enumerate(zip(range(len(self.reverse_label_mapping)), probabilities[0])):
                class_name = self.reverse_label_mapping[class_idx]
                print(f"  {class_name}: {prob:.3f}")
            
            results.append({
                'text': text,
                'predicted_label': pred_label,
                'confidence': confidence,
                'all_probabilities': probabilities[0].cpu().numpy()
            })
        
        return results
    
    def save_results(self, output_dir='test_results'):
        """Save test results to files"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'test_metrics.json')
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {
            'accuracy': float(self.test_results['accuracy']),
            'precision_macro': float(self.test_results['precision_macro']),
            'recall_macro': float(self.test_results['recall_macro']),
            'f1_macro': float(self.test_results['f1_macro']),
            'precision_weighted': float(self.test_results['precision_weighted']),
            'recall_weighted': float(self.test_results['recall_weighted']),
            'f1_weighted': float(self.test_results['f1_weighted']),
            'confusion_matrix': self.test_results['confusion_matrix'].tolist(),
            'per_class_metrics': {
                'precision': self.test_results['per_class_metrics']['precision'].tolist(),
                'recall': self.test_results['per_class_metrics']['recall'].tolist(),
                'f1': self.test_results['per_class_metrics']['f1'].tolist(),
                'support': self.test_results['per_class_metrics']['support'].tolist()
            },
            'label_mapping': getattr(self, 'label_mapping', {}),
            'test_timestamp': datetime.now().isoformat()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'text': self.test_texts,
            'true_label': [self.reverse_label_mapping[label] for label in self.test_labels],
            'predicted_label': [self.reverse_label_mapping[pred] for pred in self.predictions],
            'confidence': np.max(self.probabilities, axis=1),
            'correct': np.array(self.test_labels) == self.predictions
        })
        
        results_file = os.path.join(output_dir, 'detailed_results.csv')
        results_df.to_csv(results_file, index=False)
        
        # Save classification report
        report_file = os.path.join(output_dir, 'classification_report.json')
        with open(report_file, 'w') as f:
            json.dump(self.classification_report, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"- Metrics: {metrics_file}")
        print(f"- Detailed results: {results_file}")
        print(f"- Classification report: {report_file}")
        
        return output_dir

def run_comprehensive_test(model_path, test_data_path=None, test_df=None, 
                          text_column='text', label_column='label', 
                          output_dir='test_results'):
    """
    Run comprehensive testing on a trained model
    
    Args:
        model_path: Path to the saved model
        test_data_path: Path to test CSV file
        test_df: DataFrame with test data (alternative to file path)
        text_column: Name of text column
        label_column: Name of label column
        output_dir: Directory to save results
    """
    
    print("="*60)
    print("STARTING COMPREHENSIVE MODEL TESTING")
    print("="*60)
    
    # Initialize tester
    tester = ModelTester(model_path)
    
    # Prepare test data
    tester.prepare_test_data(test_data_path, test_df, text_column, label_column)
    
    # Make predictions
    predictions, probabilities = tester.predict_batch()
    
    # Calculate metrics
    test_results = tester.calculate_metrics()
    
    # Generate classification report
    tester.generate_classification_report()
    
    # Analyze misclassifications
    tester.analyze_misclassifications()
    
    # Save results
    tester.save_results(output_dir)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return tester, test_results