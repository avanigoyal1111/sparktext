import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---- Load the saved features and labels ----
X = sp.load_npz("data/processed/tfidf_matrix.npz")
y = pd.read_csv("data/processed/labels.csv")["label"]

print("Feature matrix shape:", X.shape)
print("Label distribution:\n", y.value_counts())

# ---- Define the three classifiers, matching the paper's defaults ----
classifiers = {
    "SVM": LinearSVC(penalty="l2", max_iter=5000),                  # paper: SVM default = L2 regularization
    "Logistic Regression": LogisticRegression(penalty="l2", max_iter=1000),  # paper: LR default = L2
    "Naive Bayes": MultinomialNB()                                  # paper: independence assumption between features
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, clf in classifiers.items():
    accs, precs, recs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accs.append(accuracy_score(y_test, preds))
        precs.append(precision_score(y_test, preds, average="macro"))
        recs.append(recall_score(y_test, preds, average="macro"))

        print(f"{name} - Fold {fold+1}: Accuracy={accs[-1]:.4f}")

    avg_acc = np.mean(accs)
    avg_prec = np.mean(precs)
    avg_rec = np.mean(recs)

    results.append({
        "Classifier": name,
        "Accuracy": f"{avg_acc*100:.2f}%",
        "Precision": f"{avg_prec*100:.2f}%",
        "Recall": f"{avg_rec*100:.2f}%"
    })

    print(f"\n{name} AVERAGE across 5 folds: Acc={avg_acc*100:.2f}%, Prec={avg_prec*100:.2f}%, Rec={avg_rec*100:.2f}%\n")

# ---- Save results table (this reproduces the paper's Table 3 for Abstracts) ----
results_df = pd.DataFrame(results)
results_df.to_csv("results/table3_accuracy_precision_recall.csv", index=False)
print("\nSaved final results table to results/table3_accuracy_precision_recall.csv")
print(results_df)