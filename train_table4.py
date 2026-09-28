import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score

X = sp.load_npz("data/processed/tfidf_matrix.npz")
y = pd.read_csv("data/processed/labels.csv")["label"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(clf):
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accs.append(accuracy_score(y_test, preds))
    return np.mean(accs)

# ---- Define every (classifier, regularization) combination from the paper's Table 4 ----
configs = [
    ("SVM", "L2 (Default)", LinearSVC(penalty="l2", max_iter=5000)),
    ("SVM", "L1",            LinearSVC(penalty="l1", dual=False, max_iter=5000)),
    ("SVM", "None",          SGDClassifier(loss="hinge", penalty=None, max_iter=2000, random_state=42)),

    ("Logistic Regression", "L2 (Default)", LogisticRegression(penalty="l2", max_iter=1000)),
    ("Logistic Regression", "L1",            LogisticRegression(penalty="l1", solver="saga", max_iter=3000)),
    ("Logistic Regression", "None",          LogisticRegression(penalty=None, max_iter=1000)),
]

results = []
for classifier_name, reg_name, clf in configs:
    print(f"Running {classifier_name} with {reg_name} regularization...")
    acc = run_cv(clf)
    print(f"  -> Average accuracy: {acc*100:.2f}%\n")
    results.append({
        "Classifier": classifier_name,
        "Regularization Parameter": reg_name,
        "Accuracy": f"{acc*100:.2f}%"
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/table4_regularization.csv", index=False)
print("\nSaved to results/table4_regularization.csv\n")
print(results_df)