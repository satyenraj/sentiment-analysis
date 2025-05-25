"""
Visualization Utilities for Model Testing
=========================================

This module provides comprehensive visualization functions for model testing results,
including confusion matrices, performance metrics, confidence analysis, and more.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TestVisualization:
    """Class for creating comprehensive test result visualizations"""
    
    def __init__(self, test_results=None, tester=None, results_file=None):
        """
        Initialize visualization class
        
        Args:
            test_results: Dictionary with test results
            tester: ModelTester instance
            results_file: Path to saved test results JSON file
        """
        if tester is not None:
            self.test_results = tester.test_results
            self.predictions = tester.predictions
            self.probabilities = tester.probabilities
            self.test_labels = tester.test_labels
            self.test_texts = tester.test_texts
            self.reverse_label_mapping = tester.reverse_label_mapping
        elif test_results is not None:
            self.test_results = test_results
        elif results_file is not None:
            self.load_results_from_file(results_file)
        else:
            raise ValueError("Either test_results, tester, or results_file must be provided")
    
    def load_results_from_file(self, results_file):
        """Load test results from saved JSON file"""
        with open(results_file, 'r') as f:
            self.test_results = json.load(f)
    
    def create_test_metrics_dashboard(self, figsize=(18, 12)):
        """Create a comprehensive dashboard for test metrics"""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Overall Performance Metrics
        overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        overall_values = [
            self.test_results['accuracy'],
            self.test_results['precision_weighted'],
            self.test_results['recall_weighted'],
            self.test_results['f1_weighted']
        ]
        
        bars1 = axes[0, 0].bar(overall_metrics, overall_values, 
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        axes[0, 0].set_title('Overall Test Performance', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, overall_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Per-Class F1 Scores
        class_names = [self.reverse_label_mapping[i] for i in range(len(self.reverse_label_mapping))]
        f1_scores = self.test_results['per_class_metrics']['f1']
        colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']
        
        bars2 = axes[0, 1].bar(class_names, f1_scores, color=colors, alpha=0.8)
        axes[0, 1].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, f1_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Precision vs Recall by Class
        precisions = self.test_results['per_class_metrics']['precision']
        recalls = self.test_results['per_class_metrics']['recall']
        
        x = np.arange(len(class_names))
        width = 0.35
        
        bars3a = axes[0, 2].bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='#4ECDC4')
        bars3b = axes[0, 2].bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='#45B7D1')
        
        axes[0, 2].set_title('Precision vs Recall by Class', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_xlabel('Classes')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(class_names)
        axes[0, 2].legend()
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # 4. Confusion Matrix
        conf_matrix = np.array(self.test_results['confusion_matrix'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[1, 0], cbar_kws={'label': 'Count'})
        axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Labels')
        axes[1, 0].set_ylabel('True Labels')
        
        # 5. Class Distribution (Support)
        supports = self.test_results['per_class_metrics']['support']
        
        wedges, texts, autotexts = axes[1, 1].pie(supports, labels=class_names, autopct='%1.1f%%',
                                                 colors=colors, startangle=90)
        axes[1, 1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        # 6. Macro vs Weighted Metrics Comparison
        metric_types = ['Precision', 'Recall', 'F1-Score']
        macro_values = [
            self.test_results['precision_macro'],
            self.test_results['recall_macro'],
            self.test_results['f1_macro']
        ]
        weighted_values = [
            self.test_results['precision_weighted'],
            self.test_results['recall_weighted'],
            self.test_results['f1_weighted']
        ]
        
        x = np.arange(len(metric_types))
        width = 0.35
        
        bars6a = axes[1, 2].bar(x - width/2, macro_values, width, label='Macro Avg', alpha=0.8, color='#FF9F43')
        bars6b = axes[1, 2].bar(x + width/2, weighted_values, width, label='Weighted Avg', alpha=0.8, color='#10AC84')
        
        axes[1, 2].set_title('Macro vs Weighted Averages', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metric_types)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Test Results Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    def create_detailed_confusion_matrix(self, figsize=(15, 6)):
        """Create a detailed confusion matrix with percentages"""
        
        conf_matrix = np.array(self.test_results['confusion_matrix'])
        class_names = [self.reverse_label_mapping[i] for i in range(len(self.reverse_label_mapping))]
        
        # Calculate percentages
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Raw counts
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Labels')
        ax1.set_ylabel('True Labels')
        
        # Percentages
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.1f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Labels')
        ax2.set_ylabel('True Labels')
        
        plt.suptitle('Detailed Confusion Matrix Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_classification_report_heatmap(self, figsize=(10, 6)):
        """Create a visual classification report"""
        
        class_names = [self.reverse_label_mapping[i] for i in range(len(self.reverse_label_mapping))]
        
        # Prepare data for heatmap
        metrics_data = []
        for i in range(len(class_names)):
            metrics_data.append([
                self.test_results['per_class_metrics']['precision'][i],
                self.test_results['per_class_metrics']['recall'][i],
                self.test_results['per_class_metrics']['f1'][i]
            ])
        
        metrics_df = pd.DataFrame(metrics_data, 
                                 columns=['Precision', 'Recall', 'F1-Score'],
                                 index=class_names)
        
        plt.figure(figsize=figsize)
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
        plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
        plt.ylabel('Classes')
        plt.xlabel('Metrics')
        plt.tight_layout()
        plt.show()
    
    def create_performance_radar_chart(self, figsize=(8, 8)):
        """Create a radar chart for overall performance"""
        
        try:
            # Performance metrics
            metrics = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']
            values = [
                self.test_results['accuracy'],
                self.test_results['precision_weighted'],
                self.test_results['recall_weighted'],
                self.test_results['f1_weighted']
            ]
            
            # Calculate angles for each metric
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True)
            
            # Add score labels
            for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
                ax.text(angle, value + 0.05, f'{value:.3f}', 
                       horizontalalignment='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Radar chart creation failed: {e}")
            print("Creating alternative bar chart...")
            
            plt.figure(figsize=figsize)
            plt.bar(metrics, values[:-1], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
            plt.title('Model Performance Overview', fontsize=16, fontweight='bold')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            for i, v in enumerate(values[:-1]):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def analyze_confidence_distribution(self, figsize=(12, 4)):
        """Analyze prediction confidence distribution"""
        
        if not hasattr(self, 'probabilities') or self.probabilities is None:
            print("Confidence analysis requires probability data from ModelTester")
            return
        
        # Get max probabilities (confidence scores)
        confidences = np.max(self.probabilities, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = np.array(self.test_labels) == self.predictions
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        print(f"\nCONFIDENCE ANALYSIS")
        print("-" * 40)
        print(f"Average confidence (correct): {np.mean(correct_confidences):.3f}")
        print(f"Average confidence (incorrect): {np.mean(incorrect_confidences):.3f}")
        print(f"Overall average confidence: {np.mean(confidences):.3f}")
        
        # Plot confidence distribution
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([correct_confidences, incorrect_confidences], 
                   labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence Score')
        plt.title('Confidence Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return confidences
    
    def create_class_performance_comparison(self, figsize=(15, 10)):
        """Create detailed class-wise performance comparison"""
        
        class_names = [self.reverse_label_mapping[i] for i in range(len(self.reverse_label_mapping))]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Precision by class
        precisions = self.test_results['per_class_metrics']['precision']
        bars1 = axes[0, 0].bar(class_names, precisions, color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_title('Precision by Class', fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars1, precisions):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Recall by class
        recalls = self.test_results['per_class_metrics']['recall']
        bars2 = axes[0, 1].bar(class_names, recalls, color='#45B7D1', alpha=0.8)
        axes[0, 1].set_title('Recall by Class', fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars2, recalls):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. F1-Score by class
        f1_scores = self.test_results['per_class_metrics']['f1']
        bars3 = axes[1, 0].bar(class_names, f1_scores, color='#96CEB4', alpha=0.8)
        axes[1, 0].set_title('F1-Score by Class', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars3, f1_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Support (number of samples) by class
        supports = self.test_results['per_class_metrics']['support']
        bars4 = axes[1, 1].bar(class_names, supports, color='#FF6B6B', alpha=0.8)
        axes[1, 1].set_title('Support by Class', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars4, supports):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 10,
                           f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Class-wise Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_all_visualizations(self):
        """Create all visualizations in sequence"""
        
        print("Creating comprehensive test result visualizations...")
        
        print("1. Main test dashboard...")
        self.create_test_metrics_dashboard()
        
        print("2. Detailed confusion matrix...")
        self.create_detailed_confusion_matrix()
        
        print("3. Classification report heatmap...")
        self.create_classification_report_heatmap()
        
        print("4. Performance radar chart...")
        self.create_performance_radar_chart()
        
        print("5. Class performance comparison...")
        self.create_class_performance_comparison()
        
        if hasattr(self, 'probabilities') and self.probabilities is not None:
            print("6. Confidence distribution analysis...")
            self.analyze_confidence_distribution()
        
        print("All visualizations completed!")

def print_test_summary(test_results, reverse_label_mapping=None):
    """Print a formatted test summary"""
    
    print("="*60)
    print("MODEL TEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nOVERALL PERFORMANCE:")
    print("-" * 30)
    print(f"Accuracy:           {test_results['accuracy']:.4f}")
    print(f"Precision (Macro):  {test_results['precision_macro']:.4f}")
    print(f"Recall (Macro):     {test_results['recall_macro']:.4f}")
    print(f"F1-Score (Macro):   {test_results['f1_macro']:.4f}")
    print(f"Precision (Weighted): {test_results['precision_weighted']:.4f}")
    print(f"Recall (Weighted):    {test_results['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted):  {test_results['f1_weighted']:.4f}")
    
    print(f"\nPER-CLASS PERFORMANCE:")
    print("-" * 30)
    
    if reverse_label_mapping is None:
        class_names = [f"Class_{i}" for i in range(len(test_results['per_class_metrics']['f1']))]
    else:
        class_names = [reverse_label_mapping[i] for i in range(len(reverse_label_mapping))]
    
    for i, cls_name in enumerate(class_names):
        precision = test_results['per_class_metrics']['precision'][i]
        recall = test_results['per_class_metrics']['recall'][i]
        f1 = test_results['per_class_metrics']['f1'][i]
        support = test_results['per_class_metrics']['support'][i]
        
        print(f"{cls_name:10} - Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1: {f1:.4f}, "
              f"Support: {int(support)}")
    
    # Calculate total samples
    total_samples = sum(test_results['per_class_metrics']['support'])
    print(f"\nTOTAL TEST SAMPLES: {int(total_samples)}")
    
    # Identify best and worst performing classes
    f1_scores = test_results['per_class_metrics']['f1']
    best_idx = np.argmax(f1_scores)
    worst_idx = np.argmin(f1_scores)
    
    print(f"\nBEST PERFORMING CLASS:  {class_names[best_idx]} (F1: {f1_scores[best_idx]:.4f})")
    print(f"WORST PERFORMING CLASS: {class_names[worst_idx]} (F1: {f1_scores[worst_idx]:.4f})")

def create_interactive_plots(test_results, reverse_label_mapping=None):
    """Create interactive plots using plotly (if available)"""
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        class_names = [reverse_label_mapping[i] for i in range(len(reverse_label_mapping))] if reverse_label_mapping else [f"Class_{i}" for i in range(len(test_results['per_class_metrics']['f1']))]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Performance', 'Per-Class F1 Scores', 
                          'Precision vs Recall', 'Confusion Matrix'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Overall performance
        overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        overall_values = [
            test_results['accuracy'],
            test_results['precision_weighted'],
            test_results['recall_weighted'],
            test_results['f1_weighted']
        ]
        
        fig.add_trace(go.Bar(
            x=overall_metrics, 
            y=overall_values,
            name='Overall Performance',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ), row=1, col=1)
        
        # Per-class F1 scores
        f1_scores = test_results['per_class_metrics']['f1']
        fig.add_trace(go.Bar(
            x=class_names,
            y=f1_scores,
            name='F1 Scores',
            marker_color=['#FF6B6B', '#FFD93D', '#6BCF7F']
        ), row=1, col=2)
        
        # Precision vs Recall scatter
        precisions = test_results['per_class_metrics']['precision']
        recalls = test_results['per_class_metrics']['recall']
        
        fig.add_trace(go.Scatter(
            x=precisions,
            y=recalls,
            mode='markers+text',
            text=class_names,
            textposition="top center",
            marker=dict(size=10, color=['#FF6B6B', '#FFD93D', '#6BCF7F']),
            name='Classes'
        ), row=2, col=1)
        
        # Confusion matrix heatmap
        conf_matrix = test_results['confusion_matrix']
        fig.add_trace(go.Heatmap(
            z=conf_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Interactive Test Results Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        fig.show()
        
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        print("Falling back to matplotlib visualizations...")