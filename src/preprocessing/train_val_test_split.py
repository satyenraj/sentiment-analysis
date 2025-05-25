import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(df, language_column, label_column, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split dataframe into train, validation and test sets with stratification
    
    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of the column containing labels for stratification
        val_size (float): Proportion of data for validation set
        test_size (float): Proportion of data for test set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[[language_column, label_column]],
        random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    # Adjust validation size to account for the test split
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[[language_column, label_column]],
        random_state=random_state
    )
    
    return train_df, val_df, test_df