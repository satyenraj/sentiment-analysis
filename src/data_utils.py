"""
Data Utilities for Model Testing
================================

This module provides utility functions for handling test data,
including data loading, preprocessing, and sample generation.

Author: Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_test_data(file_path: str, text_column: str = 'review', label_column: str = 'label') -> pd.DataFrame:
    """
    Load test data from various file formats
    
    Args:
        file_path: Path to the test data file
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        DataFrame with test data
    """
    
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} samples from {file_path}")
    print(f"Class distribution:\n{df[label_column].value_counts()}")
    
    return df

def create_sample_test_data(num_samples: int = 100, include_language_tags: bool = True) -> pd.DataFrame:
    """
    Create sample test data for demonstration purposes
    
    Args:
        num_samples: Number of samples to generate
        include_language_tags: Whether to include language tags
        
    Returns:
        DataFrame with sample test data
    """
    
    # Sample texts for each sentiment and language
    sample_texts = {
        'positive': {
            'english': [
                "This product is amazing! Highly recommended.",
                "Excellent quality and fast delivery.",
                "Love it! Worth every penny.",
                "Outstanding service and great product.",
                "Perfect! Exactly what I needed.",
                "Fantastic experience, will buy again.",
                "Top quality, exceeded expectations.",
                "Brilliant product, very satisfied.",
                "Great value for money.",
                "Impressive quality and design."
            ],
            'nepali_romanized': [
                "Yo product ekdam ramro cha, dami lagyo.",
                "Khusi lagyo, ramro quality cha.",
                "Ekdam ramro product ho, man paryo.",
                "Sahi cha, paisa wasool cha.",
                "Dherai ramro lagyo, satisfied chu.",
                "Quality ramro cha, recommend garchu.",
                "Ekdam best product cha.",
                "Man paryo, ramro investment thiyo.",
                "Price anusar ramro cha.",
                "Khusi chu yo product le."
            ]
        },
        'negative': {
            'english': [
                "Terrible quality, waste of money.",
                "Poor service and bad product.",
                "Disappointed with the purchase.",
                "Not worth it, very poor quality.",
                "Horrible experience, don't recommend.",
                "Bad quality, money wasted.",
                "Very disappointing product.",
                "Poor packaging and damaged item.",
                "Not satisfied at all.",
                "Regret buying this product."
            ],
            'nepali_romanized': [
                "Ekdam kharab product cha, paisa barbad bhayo.",
                "Quality ramro chaina, man pardaina.",
                "Khasai ramro lagena, disappointed chu.",
                "Paisa worth chaina, kharab cha.",
                "Ekdam disappointed chu, bad experience.",
                "Quality kharab cha, regret garchu.",
                "Man pardaina, waste of money.",
                "Kharab product cha, don't buy.",
                "Satisfied chaina, poor quality.",
                "Ramro hoina, money waste."
            ]
        },
        'neutral': {
            'english': [
                "It's okay, nothing special.",
                "Average quality, decent price.",
                "Not bad, but not great either.",
                "It's fine, meets basic requirements.",
                "Acceptable quality for the price.",
                "Could be better, but it works.",
                "Standard product, nothing extraordinary.",
                "It's alright, does the job.",
                "Fair quality, reasonable price.",
                "Decent enough, no complaints."
            ],
            'nepali_romanized': [
                "Thikai cha, khasai special chaina.",
                "Average quality cha, price thikai cha.",
                "Ramro ta hoina, tara chalcha.",
                "Thikai lagyo, basic requirement pura garcha.",
                "Price anusar thikai cha.",
                "Ramro huna sakthiyo, tara chalcha.",
                "Normal product cha, kei special chaina.",
                "Chalcha, kaam garcha.",
                "Thikai quality cha, price reasonable.",
                "Decent cha, complaint chaina."
            ]
        }
    }
    
    texts = []
    labels = []
    
    # Calculate samples per category
    samples_per_sentiment = num_samples // 3
    samples_per_lang = samples_per_sentiment // 2
    
    for sentiment in ['positive', 'negative', 'neutral']:
        for lang_type in ['english', 'nepali_romanized']:
            lang_texts = sample_texts[sentiment][lang_type]
            
            # Sample with replacement if needed
            sampled_texts = np.random.choice(lang_texts, size=samples_per_lang, replace=True)
            
            for text in sampled_texts:
                if include_language_tags:
                    if lang_type == 'english':
                        text = f"<en> {text}"
                    else:
                        text = f"<ne-rom> {text}"
                
                texts.append(text)
                labels.append(sentiment)
    
    # Add remaining samples to balance
    remaining = num_samples - len(texts)
    if remaining > 0:
        for _ in range(remaining):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'])
            lang_type = np.random.choice(['english', 'nepali_romanized'])
            text = np.random.choice(sample_texts[sentiment][lang_type])
            
            if include_language_tags:
                if lang_type == 'english':
                    text = f"<en> {text}"
                else:
                    text = f"<ne-rom> {text}"
            
            texts.append(text)
            labels.append(sentiment)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Generated {len(df)} sample test records")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df

def preprocess_text(text: str, remove_language_tags: bool = False) -> str:
    """
    Preprocess text for testing
    
    Args:
        text: Input text
        remove_language_tags: Whether to remove language tags
        
    Returns:
        Preprocessed text
    """
    
    if remove_language_tags:
        # Remove language tags like <en>, <ne-rom>, etc.
        import re
        text = re.sub(r'<[^>]+>', '', text).strip()
    
    # Basic cleaning
    text = text.strip()
    
    return text

def balance_test_data(df: pd.DataFrame, label_column: str = 'label', 
                     balance_method: str = 'undersample') -> pd.DataFrame:
    """
    Balance test data across classes
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        balance_method: 'undersample', 'oversample', or 'none'
        
    Returns:
        Balanced DataFrame
    """
    
    if balance_method == 'none':
        return df
    
    class_counts = df[label_column].value_counts()
    
    if balance_method == 'undersample':
        # Undersample to the smallest class
        min_count = class_counts.min()
        balanced_dfs = []
        
        for label in class_counts.index:
            class_df = df[df[label_column] == label].sample(n=min_count, random_state=42)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif balance_method == 'oversample':
        # Oversample to the largest class
        max_count = class_counts.max()
        balanced_dfs = []
        
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            if len(class_df) < max_count:
                # Oversample with replacement
                oversampled = class_df.sample(n=max_count, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced data
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced data using {balance_method}:")
    print(f"New class distribution:\n{balanced_df[label_column].value_counts()}")
    
    return balanced_df

def split_by_language(df: pd.DataFrame, text_column: str = 'text') -> Dict[str, pd.DataFrame]:
    """
    Split test data by language based on language tags
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        
    Returns:
        Dictionary with language-specific DataFrames
    """
    
    import re
    
    language_dfs = {}
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        # Extract language tag
        lang_match = re.match(r'<([^>]+)>', text)
        if lang_match:
            lang_tag = lang_match.group(1)
        else:
            lang_tag = 'unknown'
        
        if lang_tag not in language_dfs:
            language_dfs[lang_tag] = []
        
        language_dfs[lang_tag].append(row)
    
    # Convert lists to DataFrames
    for lang in language_dfs:
        language_dfs[lang] = pd.DataFrame(language_dfs[lang])
        print(f"Language '{lang}': {len(language_dfs[lang])} samples")
    
    return language_dfs

def export_test_data(df: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
    """
    Export test data to file
    
    Args:
        df: DataFrame to export
        output_path: Output file path
        format: Export format ('csv', 'json', 'excel')
    """
    
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() == 'json':
        df.to_json(output_path, orient='records', indent=2)
    elif format.lower() == 'excel':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Test data exported to {output_path}")

def validate_test_data(df: pd.DataFrame, text_column: str = 'text', 
                      label_column: str = 'label') -> Dict[str, any]:
    """
    Validate test data and return statistics
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Dictionary with validation statistics
    """
    
    stats = {
        'total_samples': len(df),
        'missing_text': df[text_column].isna().sum(),
        'missing_labels': df[label_column].isna().sum(),
        'empty_text': (df[text_column] == '').sum(),
        'unique_labels': df[label_column].nunique(),
        'class_distribution': df[label_column].value_counts().to_dict(),
        'avg_text_length': df[text_column].str.len().mean(),
        'min_text_length': df[text_column].str.len().min(),
        'max_text_length': df[text_column].str.len().max()
    }
    
    # Check for language tags
    import re
    has_language_tags = df[text_column].str.contains(r'<[^>]+>', regex=True).any()
    stats['has_language_tags'] = has_language_tags
    
    if has_language_tags:
        # Extract language tags
        language_tags = df[text_column].str.extract(r'<([^>]+)>')[0].value_counts().to_dict()
        stats['language_distribution'] = language_tags
    
    print("DATA VALIDATION RESULTS:")
    print("-" * 40)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Missing text: {stats['missing_text']}")
    print(f"Missing labels: {stats['missing_labels']}")
    print(f"Empty text: {stats['empty_text']}")
    print(f"Unique labels: {stats['unique_labels']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")
    print(f"Has language tags: {stats['has_language_tags']}")
    
    if stats['has_language_tags']:
        print(f"Language distribution: {stats['language_distribution']}")
    
    print(f"Class distribution: {stats['class_distribution']}")
    
    return stats

# Example usage and utility functions
def create_multilingual_samples() -> List[Dict[str, str]]:
    """Create sample multilingual test cases"""
    
    samples = [
        {
            'text': '<en> This product exceeded my expectations!',
            'expected_label': 'positive',
            'language': 'english'
        },
        {
            'text': '<ne-rom> Yo product ekdam ramro cha, man paryo.',
            'expected_label': 'positive',
            'language': 'nepali_romanized'
        },
        {
            'text': '<en> Poor quality, not worth the money.',
            'expected_label': 'negative',
            'language': 'english'
        },
        {
            'text': '<ne-rom> Quality kharab cha, paisa waste bhayo.',
            'expected_label': 'negative',
            'language': 'nepali_romanized'
        },
        {
            'text': '<en> It\'s okay, nothing special but works.',
            'expected_label': 'neutral',
            'language': 'english'
        },
        {
            'text': '<ne-rom> Thikai cha, khasai ramro hoina tara chalcha.',
            'expected_label': 'neutral',
            'language': 'nepali_romanized'
        }
    ]
    
    return samples

if __name__ == "__main__":
    # Example usage
    print("Data utilities loaded successfully!")
    
    # Create sample data
    sample_df = create_sample_test_data(num_samples=50, include_language_tags=True)
    
    # Validate data
    validation_stats = validate_test_data(sample_df)
    
    # Split by language
    language_splits = split_by_language(sample_df)
    
    print("\nSample test data created and validated!")