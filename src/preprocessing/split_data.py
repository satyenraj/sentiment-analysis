import pandas as pd
from train_val_test_split import stratified_split

""" df = pd.read_csv('data/processed/multi/reduced_eng_all.csv')

train_df, val_df, test_df = stratified_split(df, 'language', 'sentiment')

train_df.to_csv('data/processed/multi/filtered/train.csv')
val_df.to_csv('data/processed/multi/filtered/val.csv')
test_df.to_csv('data/processed/multi/filtered/test.csv') """


df = pd.read_csv('data/processed/multi/filtered/tag/all.csv')

train_df, val_df, test_df = stratified_split(df, 'language', 'sentiment')

train_df.to_csv('data/processed/multi/filtered/tag/train.csv')
val_df.to_csv('data/processed/multi/filtered/tag/val.csv')
test_df.to_csv('data/processed/multi/filtered/tag/test.csv')
