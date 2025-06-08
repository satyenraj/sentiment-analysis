import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd

# ------------------------------
# Detect device (MPS for Mac, CUDA for NVIDIA, else CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------
# Load Zero-Shot Classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if device != "cpu" else -1
)

# ------------------------------
# Load your dataset
# Assume CSV with column 'review'
df = pd.read_csv("/Users/satyendrarajshakya/My Documents/Learning/MSc_DS/research/code/sentiment/data/reviews_with_language_1.csv")
df = df[df["review"].notna()].copy()
#reviews = df_non_null["review"].astype(str).tolist()

# ------------------------------
# Candidate sentiment labels
candidate_labels = ["positive", "neutral", "negative"]

# Open a CSV file to append the results
output_file = "/Users/satyendrarajshakya/My Documents/Learning/MSc_DS/research/code/sentiment/data/reviews_with_sentiment_1.csv"
first_batch = True  # To write header only once

# ------------------------------
# Batch processing
batch_size = 32
predictions = []

# Split into batches
for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_reviews = df.iloc[start_idx:end_idx]["review"].astype(str).tolist()

    # Predict
    batch_predictions = classifier(batch_reviews, candidate_labels, multi_label=False)

    # Get the highest scoring label
    sentiments = [pred["labels"][0] for pred in batch_predictions]

    # Add predicted sentiment to this batch
    df_batch = df.iloc[start_idx:end_idx].copy()
    df_batch["predicted_sentiment"] = sentiments

    # Append to CSV
    if first_batch:
        df_batch.to_csv(output_file, index=False, mode="w", encoding="utf-8-sig")
        first_batch = False
    else:
        df_batch.to_csv(output_file, index=False, mode="a", header=False, encoding="utf-8-sig")

print(f"\n✅ Done! Saved predictions to {output_file}")





""" from transformers import pipeline
import pandas as pd

# Load dataset (no labels yet)
input_path = "/Users/satyendrarajshakya/My Documents/Learning/MSc_DS/research/code/sentiment/data/daraz_product_review_with_language.csv"
df = pd.read_csv(input_path)  # contains a 'text' column

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["positive", "neutral", "negative"]

def auto_label(text):
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # top predicted label

# Apply weak supervision to generate pseudo-labels
df['label'] = df['review'].astype(str).apply(auto_label)

# Map string labels to numeric for training
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['label'].map(label_map)

# Save to CSV
df[['product_id','rating','review','date','label']].to_csv("/Users/satyendrarajshakya/My Documents/Learning/MSc_DS/research/code/sentiment/data/pseudo_labeled_dataset.csv", index=False) """