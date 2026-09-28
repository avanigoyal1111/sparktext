import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords list (only needs to run once, but safe to leave in)
nltk.download('stopwords')

# ---- Load the cleaned data from the previous step ----
df = pd.read_csv("data/processed/clean_abstracts.csv")
print("Loaded rows:", len(df))

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()                        # Step: case normalization
    text = re.sub(r'[^a-z\s]', ' ', text)            # Step: strip punctuation/numbers/special symbols
    tokens = text.split()                             # Step: tokenize into words
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # Step: remove stopwords + very short tokens
    tokens = [stemmer.stem(t) for t in tokens]        # Step: Porter stemming
    return " ".join(tokens)

# Apply preprocessing to every abstract
df["clean_text"] = df["text"].apply(preprocess_text)

# ---- Save a BEFORE/AFTER sample for demonstration ----
sample = df[["label", "text", "clean_text"]].head(10)
sample.to_csv("results/preprocessing_before_after_sample.csv", index=False)
print("\nSaved 10-row before/after sample to results/preprocessing_before_after_sample.csv")

# ---- Save the fully preprocessed dataset ----
df.to_csv("data/processed/preprocessed_abstracts.csv", index=False)
print("Saved full preprocessed dataset to data/processed/preprocessed_abstracts.csv")

# ---- Print a couple of examples directly to console too ----
print("\n--- Example 1 (BEFORE) ---")
print(df["text"].iloc[0][:200])
print("\n--- Example 1 (AFTER) ---")
print(df["clean_text"].iloc[0][:200])