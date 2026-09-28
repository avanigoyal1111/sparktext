# SparkText Replication: Cancer Type Classification from PubMed Abstracts

A from-scratch replication of the machine learning experiments in:

> Ye Z, Tafti AP, He KY, Wang K, He MM (2016). *SparkText: Biomedical Text Mining on Big Data Framework.* PLOS ONE 11(9): e0162721. https://doi.org/10.1371/journal.pone.0162721

The paper classifies PubMed abstracts into **breast, lung, or prostate cancer** using TF-IDF features and three classifiers (Naive Bayes, SVM, Logistic Regression). The original used Java with Apache Spark on a 20-node cluster. This project reproduces the same pipeline in **Python with scikit-learn on a single laptop**.

## Dataset

The 19,681 PubMed abstracts published by the paper's authors on Figshare (DOI `10.6084/m9.figshare.3796290`). Class counts match the paper's Table 2 exactly:

| Class | Abstracts |
|---|---|
| Breast cancer | 6,137 |
| Lung cancer | 6,680 |
| Prostate cancer | 6,864 |
| **Total** | **19,681** |

The raw file is space-delimited with quoted text and Latin-1 encoding (not a standard comma-separated CSV), so `load_data.py` uses a custom parser.

## Pipeline

1. **Load and clean** the raw file into a two-column table (`label`, `text`).
2. **Preprocess**: lowercase, strip punctuation and digits, remove stopwords, Porter stemming.
3. **Features**: TF-IDF with unigrams and bigrams (20,000 features, `min_df=2`).
4. **Models**: Naive Bayes, linear SVM, Logistic Regression.
5. **Evaluation**: 5-fold stratified cross-validation, accuracy / precision / recall, regularization sweep, ROC curves, runtime.

## Project structure

```
data/raw/          Original dataset file from Figshare
data/processed/    Cleaned data, TF-IDF matrix, labels
results/           Output tables (CSV) and figures (PNG)
load_data.py           Parse raw file into a clean CSV
preprocess.py          Text cleaning and stemming
feature_extraction.py  TF-IDF (unigrams + bigrams)
train_models.py        Table 3: accuracy / precision / recall
train_table4.py        Table 4: L2 / L1 / no regularization
roc_curves.py          Fig 4: ROC curves
fig5_bar_chart.py      Fig 5: bar chart comparison
table5_runtime.py      Table 5: runtime measurement
inspect_raw.py         Helper used to inspect the raw file format
```

## How to run

```bash
pip install pandas scikit-learn numpy scipy matplotlib nltk
python load_data.py
python preprocess.py
python feature_extraction.py
python train_models.py
python train_table4.py
python roc_curves.py
python fig5_bar_chart.py
python table5_runtime.py
```

Run them in this order. Outputs are written to `data/processed/` and `results/`.

## Results (Abstracts dataset)

### Accuracy, precision, recall (paper's Table 3)

| Classifier | Paper accuracy | This project accuracy | Precision | Recall |
|---|---|---|---|---|
| SVM | 94.63% | 91.24% | 91.21% | 91.24% |
| Logistic Regression | 92.19% | 92.45% | 92.42% | 92.46% |
| Naive Bayes | 89.38% | 87.21% | 87.20% | 87.21% |

### Regularization (paper's Table 4, accuracy)

| Classifier | L2 (default) | L1 | None |
|---|---|---|---|
| SVM | 91.24% | 93.07% | 91.14% |
| Logistic Regression | 92.45% | 93.82% | 90.39% |

### ROC curves (paper's Fig 4)

Micro-averaged one-vs-rest AUC: SVM 0.980, Logistic Regression 0.985, Naive Bayes 0.971.

![ROC curves](results/fig4_roc_curves.png)

### Metrics by classifier (paper's Fig 5)

![Accuracy, precision, recall](results/fig5_bar_chart.png)

### Runtime (paper's Table 5)

| Tool | Time (minutes) |
|---|---|
| Weka Library (paper-reported) | 138 |
| TagHelper Tools (paper-reported) | 201 |
| SparkText on a 20-node cluster (paper-reported) | 3 |
| This project, Python / scikit-learn (measured on one laptop) | 0.92 |

## Discussion

The main conclusions of the paper reproduce: SVM and Logistic Regression clearly outperform Naive Bayes, all models exceed 87% accuracy, and regularization affects results.

Some details differ from the paper, for reasons that are expected when swapping libraries:

- **Different implementations.** The paper used Spark MLlib (SGD-based SVM and Logistic Regression). scikit-learn uses different optimizers, so exact numbers and even the SVM vs. Logistic Regression ordering can shift by a few points.
- **Regularization ranking.** The paper found L2 best; here L1 performed best for both classifiers.
- **Feature cap.** TF-IDF was limited to 20,000 features for laptop-scale processing.
- **Citation headers not stripped.** Each abstract begins with journal/date/DOI metadata that was left in, which may act as a weak extra signal.
- **Runtime is not a like-for-like comparison.** At about 20,000 short documents, the overhead of a distributed cluster outweighs its benefits, so a single laptop finishes faster. Spark's advantage appears at much larger scale, which this project does not test. Weka and TagHelper times are quoted from the paper, not re-measured.

## Scope and limitations

- Only the **Abstracts** dataset was replicated. The paper's two full-text datasets were not run.
- Fig 4's ROC curve in the paper uses the Full-text Articles II dataset; here it is computed on Abstracts.
- Fig 5's Weka and TagHelper bars are not reproduced.
- The Big Data infrastructure claims (Spark, Hadoop, Cassandra) are not tested.

## Credits

Dataset and methodology by the authors of the SparkText paper. All code in this repository was written for this replication.
