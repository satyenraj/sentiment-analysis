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


nep_aug_df = pd.read_csv('data/processed/train_eng_to_np_all_update.csv')
multi_train_df = pd.read_csv('data/processed/multi/train.csv')
multi_val_df = pd.read_csv('data/processed/multi/val.csv')
multi_test_df = pd.read_csv('data/processed/multi/test.csv')

print(len(multi_train_df))
print(len(multi_val_df))
print(len(multi_test_df))

prefix_map = {
    'english': '<en>',
    'nepali': '<ne>',
    'romanized_nepali': '<ne-rom>'
}

all_df = pd.concat([multi_train_df, multi_val_df, multi_test_df], ignore_index=True)
all_df = all_df.sample(frac=1).reset_index(drop=True)
all_df.to_csv('data/processed/multi/all_without_aug.csv', index=False)

df = pd.concat([all_df, nep_aug_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/processed/multi/all_with_aug.csv', index=False)

# Create a new column with prefixed sentences
df['review'] = df['language'].map(prefix_map) + ' ' + df['review']

# Split data with stratification 
multi_train_df, multi_val_df, multi_test_df = stratified_split(df, 'language', 'label', 0.1, 0.1)

# Create a new column with prefixed sentences
#multi_train_df['review'] = multi_train_df['language'].map(prefix_map) + ' ' + multi_train_df['review']
#multi_val_df['review'] = multi_val_df['language'].map(prefix_map) + ' ' + multi_val_df['review']

print(len(multi_train_df))
print(len(multi_val_df))
print(len(multi_test_df))

multi_train_df.to_csv('data/processed/multi_tagged/train.csv', index=False)
multi_val_df.to_csv('data/processed/multi_tagged/val.csv', index=False)
multi_test_df.to_csv('data/processed/multi_tagged/test.csv', index=False)

