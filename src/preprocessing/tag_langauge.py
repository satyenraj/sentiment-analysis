import pandas as pd

prefix_map = {
    'english': '<en>',
    'nepali': '<ne>',
    'romanized_nepali': '<ne-rom>'
}

df = pd.read_csv('data/processed/multi/reduced_eng_all.csv')

df['review'] = df['language'].map(prefix_map) + ' ' + df['review']

df.to_csv('data/processed/multi/filtered/tag/all.csv', index = False)

