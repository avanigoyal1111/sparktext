import time
import re
import pandas as pd
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

total_start = time.time()

# ---- Stage 1: Load raw data ----
stage_start = time.time()
labels, texts = [], []
import csv
with open("data/raw/SparkText_SampleDataset_19681Abstracts.csv", "r", encoding="latin1") as f:
    reader = csv.reader(f, delimiter=" ", quotechar='"')
    next(reader)
    for row in reader:
        if len(row) >= 2 and row[0].strip() and row[1].strip():
            labels.append(row[0].strip())
            texts.append(row[1].strip())
df = pd.DataFrame({"label": labels, "text": texts})
load_time = time.time() - stage_start
print(f"Stage 1 - Load raw data: {load_time:.2f} seconds")

# ---- Stage 2: Text preprocessing ----
stage_start = time.time()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)
preprocess_time = time.time() - stage_start
print(f"Stage 2 - Text preprocessing: {preprocess_time:.2f} seconds")

# ---- Stage 3: TF-IDF feature extraction ----
stage_start = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]
feature_time = time.time() - stage_start
print(f"Stage 3 - Feature extraction: {feature_time:.2f} seconds")

# ---- Stage 4: Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Stage 5: Train all 3 classifiers (single run, not 5-fold CV) ----
stage_start = time.time()
LinearSVC(max_iter=5000).fit(X_train, y_train)
svm_time = time.time() - stage_start
print(f"Stage 5a - Train SVM: {svm_time:.2f} seconds")

stage_start = time.time()
LogisticRegression(max_iter=1000).fit(X_train, y_train)
lr_time = time.time() - stage_start
print(f"Stage 5b - Train Logistic Regression: {lr_time:.2f} seconds")

stage_start = time.time()
MultinomialNB().fit(X_train, y_train)
nb_time = time.time() - stage_start
print(f"Stage 5c - Train Naive Bayes: {nb_time:.2f} seconds")

total_time = time.time() - total_start
total_minutes = total_time / 60
print(f"\nTOTAL pipeline time: {total_time:.2f} seconds ({total_minutes:.2f} minutes)")

# ---- Build Table 5 comparison: your measured time vs. paper's published times ----
table5 = pd.DataFrame({
    "Tool": ["Weka Library (paper-reported)", "TagHelper Tools (paper-reported)",
             "SparkText / Spark cluster (paper-reported)", "Your Python/scikit-learn Framework (measured)"],
    "Dataset": ["Abstracts"] * 4,
    "Running Time (minutes)": [138, 201, 3, round(total_minutes, 2)]
})

table5.to_csv("results/table5_runtime_comparison.csv", index=False)
print("\nSaved to results/table5_runtime_comparison.csv")
print(table5)