import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import pickle

df = pd.read_csv("data/processed/preprocessed_abstracts.csv")
df = df.dropna(subset=["clean_text"])
print("Rows used for feature extraction:", len(df))

# TF-IDF with unigrams + bigrams, matching the paper's 2-gram strategy
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams AND bigrams, as in the paper's Fig 3
    max_features=20000,   # cap vocabulary size to keep this manageable on a laptop
    min_df=2              # ignore terms that appear in fewer than 2 documents (removes rare noise)
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("TF-IDF matrix shape (documents x features):", X.shape)
print("\nSample of feature names (first 20):")
print(vectorizer.get_feature_names_out()[:20])

# ---- Save everything needed for the modeling step ----
sp.save_npz("data/processed/tfidf_matrix.npz", X)
y.to_csv("data/processed/labels.csv", index=False)

with open("data/processed/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ---- Save a small bag-of-words style table for demonstration (like paper's Table 1) ----
sample_df = pd.DataFrame(
    X[:10].toarray(),
    columns=vectorizer.get_feature_names_out()
)
# Only keep a handful of non-zero, interesting columns for readability
nonzero_cols = sample_df.columns[(sample_df != 0).any(axis=0)][:15]
demo_table = sample_df[nonzero_cols].copy()
demo_table.insert(0, "label", df["label"].iloc[:10].values)
demo_table.to_csv("results/bag_of_words_sample_table.csv", index=False)
print("\nSaved sample bag-of-words table to results/bag_of_words_sample_table.csv")