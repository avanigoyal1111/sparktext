# SparkText : Cancer Type Classification from PubMed Abstracts

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)
![Status](https://img.shields.io/badge/status-replication%20complete-brightgreen)

A scikit-learn reproduction of the text-classification experiments in:

> Ye Z, Tafti AP, He KY, Wang K, He MM (2016). *SparkText: Biomedical Text Mining on Big Data Framework.* PLOS ONE 11(9): e0162721. https://doi.org/10.1371/journal.pone.0162721

The original work used Java and Apache Spark on a 20-node cluster. This project asks: **how much of the paper's results survive when the same pipeline is rebuilt with scikit-learn on one machine?**

## Contents

- [TL;DR](#tldr)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Results](#results)
- [Discussion](#discussion)
- [Scope and limitations](#scope-and-limitations)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Credits](#credits)

## TL;DR

| Question | Answer |
|---|---|
| Task | Classify PubMed abstracts as **breast**, **lung**, or **prostate** cancer |
| Features | TF-IDF, unigrams + bigrams, 20,000 features |
| Models | Naive Bayes, linear SVM, Logistic Regression |
| Best accuracy here | **Logistic Regression, 92.45%** (93.82% with L1) |
| Main paper conclusion reproduced? | **Yes.** SVM and LR clearly beat Naive Bayes; all models exceed 87% |
| Where results differ | SVM vs. LR ordering, and best regularizer (L1 here, L2 in the paper) |
| Full pipeline runtime | About 0.92 min on one laptop |

## Dataset

The 19,681 PubMed abstracts published by the paper's authors on Figshare (DOI `10.6084/m9.figshare.3796290`). Class counts match the paper's Table 2 exactly:

| Class | Abstracts |
|---|---|
| Breast cancer | 6,137 |
| Lung cancer | 6,680 |
| Prostate cancer | 6,864 |
| **Total** | **19,681** |

The raw file is space-delimited with quoted text and Latin-1 encoding (not a standard CSV), so `load_data.py` uses a custom parser.

Download the file from Figshare and place it in `data/raw/` before running the pipeline.

## Pipeline

```
raw file ─▶ load_data ─▶ preprocess ─▶ TF-IDF ─▶ models ─▶ tables & figures
```

1. **Load and clean** the raw file into a two-column table (`label`, `text`).
2. **Preprocess**: lowercase, strip punctuation and digits, remove stopwords, Porter stemming.
3. **Features**: TF-IDF with unigrams and bigrams (20,000 features, `min_df=2`).
4. **Models**: Naive Bayes, linear SVM, Logistic Regression.
5. **Evaluation**: 5-fold stratified cross-validation, accuracy / precision / recall, regularization sweep, ROC curves, runtime.

## Quick start

**Requirements:** Python 3.9+ and the packages below.

```bash
# optional: isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pandas scikit-learn numpy scipy matplotlib nltk
```

The preprocessing step needs NLTK's stopword list. If it is not already installed:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Run the scripts **in this order** (each one depends on the previous outputs):

```bash
python load_data.py          # parse raw file -> clean CSV
python preprocess.py         # clean + stem text
python feature_extraction.py # TF-IDF matrix
python train_models.py       # Table 3
python train_table4.py       # Table 4
python roc_curves.py         # Fig 4
python fig5_bar_chart.py     # Fig 5
python table5_runtime.py     # Table 5
```

Outputs are written to `data/processed/` and `results/`.

## Project structure

```
data/
  raw/                     Original dataset file from Figshare
  processed/               Cleaned data, TF-IDF matrix, labels
results/                   Output tables (CSV) and figures (PNG)

load_data.py               Parse raw file into a clean CSV
preprocess.py              Text cleaning and stemming
feature_extraction.py      TF-IDF (unigrams + bigrams)
train_models.py            Table 3: accuracy / precision / recall
train_table4.py            Table 4: L2 / L1 / no regularization
roc_curves.py              Fig 4: ROC curves
fig5_bar_chart.py          Fig 5: bar chart comparison
table5_runtime.py          Table 5: runtime measurement
inspect_raw.py             Helper used to inspect the raw file format
```

## Results

All numbers are from the **Abstracts** dataset with 5-fold stratified cross-validation.

### Accuracy, precision, recall (paper's Table 3)

| Classifier | Paper accuracy | This project accuracy | Precision | Recall | Δ vs. paper |
|---|---|---|---|---|---|
| SVM | 94.63% | 91.24% | 91.21% | 91.24% | −3.39 |
| Logistic Regression | 92.19% | 92.45% | 92.42% | 92.46% | +0.26 |
| Naive Bayes | 89.38% | 87.21% | 87.20% | 87.21% | −2.17 |

### Regularization (paper's Table 4, accuracy)

| Classifier | L2 (default) | L1 | None |
|---|---|---|---|
| SVM | 91.24% | 93.07% | 91.14% |
| Logistic Regression | 92.45% | 93.82% | 90.39% |

### ROC curves (paper's Fig 4)

Micro-averaged one-vs-rest AUC:

| Classifier | AUC |
|---|---|
| Logistic Regression | 0.985 |
| SVM | 0.980 |
| Naive Bayes | 0.971 |

![ROC curves](results/fig4_roc_curves.png)

### Metrics by classifier (paper's Fig 5)

![Accuracy, precision, recall](results/fig5_bar_chart.png)

### Runtime (paper's Table 5)

| Tool | Time (minutes) |
|---|---|
| Weka Library (paper-reported) | 138 |
| TagHelper Tools (paper-reported) | 201 |
| SparkText on a 20-node cluster (paper-reported) | 3 |
| **This project**, Python / scikit-learn (measured on one laptop) | **0.92** |

> Runtime is not a like-for-like comparison. See the discussion below.

## Discussion

The main conclusions of the paper reproduce: SVM and Logistic Regression clearly outperform Naive Bayes, all models exceed 87% accuracy, and regularization affects results.

Some details differ from the paper, for reasons that are expected when swapping libraries:

- **Different implementations.** The paper used Spark MLlib (SGD-based SVM and Logistic Regression). scikit-learn uses different optimizers, so exact numbers, and even the SVM vs. Logistic Regression ordering, can shift by a few points.
- **Regularization ranking.** The paper found L2 best; here L1 performed best for both classifiers.
- **Feature cap.** TF-IDF was limited to 20,000 features for laptop-scale processing.
- **Citation headers not stripped.** Each abstract begins with journal/date/DOI metadata that was left in, which may act as a weak extra signal.
- **Runtime is not a like-for-like comparison.** At about 20,000 short documents, the overhead of a distributed cluster outweighs its benefits, so a single laptop finishes faster. Spark's advantage appears at much larger scale, which this project does not test. Weka and TagHelper times are quoted from the paper, not re-measured.

## Scope and limitations

- Only the **Abstracts** dataset was replicated. The paper's two full-text datasets were not run.
- Fig 4's ROC curve in the paper uses the Full-text Articles II dataset; here it is computed on Abstracts.
- Fig 5's Weka and TagHelper bars are not reproduced.
- The Big Data infrastructure claims (Spark, Hadoop, Cassandra) are not tested.

## Reproducibility notes

- Use the same script order as in [Quick start](#quick-start); later scripts read files produced by earlier ones.
- Results depend on scikit-learn version and solver defaults. Record your versions with `pip freeze > requirements.txt` when you rerun.
- Cross-validation is stratified, so class proportions are preserved in every fold. Fix `random_state` in the scripts if you need identical folds across runs.
- Runtime in `table5_runtime.py` depends on your hardware; expect different numbers on different machines.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `UnicodeDecodeError` when loading the data | The raw file is Latin-1, not UTF-8. Use `load_data.py` rather than `pd.read_csv` directly |
| `LookupError: Resource stopwords not found` | Run `nltk.download('stopwords')` |
| `FileNotFoundError` in a later script | An earlier step was skipped; rerun the scripts in order |
| Slow preprocessing | Porter stemming over ~20k abstracts is the bottleneck; run it once and reuse `data/processed/` |

## Citation

If you use this replication, please cite the original paper:

```bibtex
@article{ye2016sparktext,
  title   = {SparkText: Biomedical Text Mining on Big Data Framework},
  author  = {Ye, Zhan and Tafti, Ahmad P. and He, Karen Y. and Wang, Kai and He, Max M.},
  journal = {PLOS ONE},
  volume  = {11},
  number  = {9},
  pages   = {e0162721},
  year    = {2016},
  doi     = {10.1371/journal.pone.0162721}
}
```

## Credits

Dataset and methodology by the authors of the SparkText paper. All code in this repository was written for this replication.
