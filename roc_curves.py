import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc

X = sp.load_npz("data/processed/tfidf_matrix.npz")
y = pd.read_csv("data/processed/labels.csv")["label"]

classes = sorted(y.unique())
print("Classes:", classes)

y_bin = label_binarize(y, classes=classes)  # one-hot encode labels for multiclass ROC

X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=42, stratify=y
)

classifiers = {
    "SVM": OneVsRestClassifier(LinearSVC(max_iter=5000)),
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    "Naive Bayes": OneVsRestClassifier(MultinomialNB())
}

plt.figure(figsize=(8, 6))

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)

    # Get ranking scores: decision_function for SVM, predict_proba for the others
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)

    # Micro-average ROC: pool all classes together into one curve
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    print(f"{name}: AUC = {roc_auc:.3f}")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC Curves - Abstracts Dataset (Micro-averaged, 3-class One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.savefig("results/fig4_roc_curves.png", dpi=150, bbox_inches="tight")
print("\nSaved ROC curve plot to results/fig4_roc_curves.png")
plt.show()