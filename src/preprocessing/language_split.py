from train_val_test_split import stratified_split
import pandas as pd

""" rn_train_df = pd.read_csv('data/processed/roman_nepali/train.csv')
rn_val_df = pd.read_csv('data/processed/roman_nepali/val.csv')
rn_test_df = pd.read_csv('data/processed/roman_nepali/test.csv')

rn_train_df['review'] = '<ne-rom> ' + rn_train_df['review']
rn_val_df['review'] = '<ne-rom> ' + rn_val_df['review']
rn_test_df['review'] = '<ne-rom> ' + rn_test_df['review']

rn_train_df.to_csv('data/processed/roman_nepali/tagged/train.csv', index=False)
rn_train_df.to_csv('data/processed/roman_nepali/tagged/val.csv', index=False)
rn_train_df.to_csv('data/processed/roman_nepali/tagged/test.csv', index=False) """


np_train_df = pd.read_csv('data/processed/nepali/train.csv')
np_val_df = pd.read_csv('data/processed/nepali/val.csv')
np_test_df = pd.read_csv('data/processed/nepali/test.csv')

np_train_df['review'] = '<ne> ' + np_train_df['review']
np_val_df['review'] = '<ne> ' + np_val_df['review']
np_test_df['review'] = '<ne> ' + np_test_df['review']

np_train_df.to_csv('data/processed/nepali/tagged/train.csv', index=False)
np_train_df.to_csv('data/processed/nepali/tagged/val.csv', index=False)
np_train_df.to_csv('data/processed/nepali/tagged/test.csv', index=False)
