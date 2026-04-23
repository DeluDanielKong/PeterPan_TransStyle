# PeterPan_TransStyle

## About This Repository

This repository contains the data, scripts, and supplementary materials for the paper:

> **Unveiling the Stylometric Signatures in Translated Children's Literature: Peter Pan in Chinese**
> He Ting, Kong Delu\*, Li Baohu
> *Humanities and Social Sciences Communications* (in press)
> \* Corresponding author

The study applies stylometric analysis and machine learning to distinguish the translation styles of three Chinese translators of *Peter Pan* (J. M. Barrie), offering a computational perspective on translator stylistics in children's literature.

---

## Corpus

The corpus comprises three complete Chinese translations of *Peter Pan*. The corpus files contain the original corpus data used for training and testing the model. Due to copyright restrictions, only a representative sample (5 chapters per translator) is included in this repository; the full corpus is available upon request.

| Folder | Translator | Year | Abbrev. |
|---|---|---|---|
| `corpus/Liang1929/` | Liang Shiqiu (梁实秋) | 1929 | Liang1929 |
| `corpus/Y&G1991/` | Yang Jingyuan & Gu Xianghua (杨静远 & 顾翔华) | 1991 | Yang&Gu |
| `corpus/Ren2006/` | Ren Rongrong (任溶溶) | 2006 | Ren2006 |

Each translation was segmented into chapter-level samples. The full dataset contains **159 samples** across the three translations.

> **Note:** `Feature_set.xlsx` contains the full feature set, feature selection data, and the list of salient features selected for the final model.

---

## Feature Engineering

A total of **335 features** were extracted across four categories:

| Level | Category | Description |
|---|---|---|
| 1 | Lexical features | TTR, MATTR, hapax legomena ratio, stopword ratio, Simpson's diversity index, POS ratios, conjunction subtypes, aspect markers, etc. |
| 2 | Syntactical features | Dependency relation tag distributions, special punctuation markers (SpMark), radicals (RAD) |
| 3 | Readability features | Rhyme density, semantic accessibility (SemAcc), idiom density |
| 4 | N-word / POS-gram | High-frequency unigrams, bigrams, and POS trigrams |

Feature selection was performed using the **Chi-square test** with a 10% threshold (top 33 salient features), which captures **52.5%** of the total discriminative power. Threshold robustness is validated through sensitivity analysis across 13 thresholds (3%–100%); see `data/Sensitivity_Analysis_Results.csv` and `data/Feature_Selection_Justification_Report.txt`.

---

## Classification

Three classifiers were used with **10-fold stratified cross-validation**, and their predicted probabilities were averaged into an ensemble:

- Linear **SVM**
- **Logistic Regression**
- **Naive Bayes**

The ensemble model achieves a classification accuracy of **96.2%** (weighted F1 = 0.962) using the top-10% feature set, demonstrating that the three translations exhibit highly distinctive and learnable stylometric signatures.

---

## Repository Structure

```
PeterPan_TransStyle/
├── corpus/
│   ├── Liang1929/          # Sample chapters — Liang Shiqiu (1929)
│   ├── Ren2006/            # Sample chapters — Ren Rongrong (2006)
│   └── Y&G1991/            # Sample chapters — Yang & Gu (1991)
├── data/
│   ├── Ensemble_Classification_Results.csv          # Per-sample predicted labels & probabilities
│   ├── Feature_Selection_Justification_Report.txt   # Full threshold justification report
│   ├── Sensitivity_Analysis_Results.csv             # Accuracy metrics across 13 thresholds
│   └── vectorized data/
│       └── merged_vectorized_data.csv               # Merged feature matrix (all 335 features)
├── figures/
│   └── 3_Feature_Importance.html                   # Interactive feature importance chart (Plotly)
├── scripts/
│   ├── lexical_features.py                          # Lexical feature extraction (example)
│   ├── Peterpan_feature_seletion.py                 # Chi-square feature selection
│   ├── Peterpan_feature_threshold_justification.py  # Threshold sensitivity & elbow analysis
│   ├── peterpan_kmeans.py                           # K-means clustering + PCA visualization
│   ├── Peterpan_SVM_NB_simpleRegression.py          # SVM / LR / NB + ensemble classification
│   └── Feature_Impo_Plotly.py                       # Feature importance bar chart (Plotly)
├── Feature_set.xlsx                                 # Full feature set & salient feature list
└── README.md
```

---

## Scripts

The scripts directory provides code for data preprocessing, feature extraction, feature selection, and model training and evaluation, organized into separate files for each step. Note that `lexical_features.py` is presented as an example implementation only, as the full code package is under development for a forthcoming GUI software.

| Script | Purpose |
|---|---|
| `lexical_features.py` | Example implementation of lexical and POS-based feature extraction. Part of a GUI software package under development. |
| `Peterpan_feature_seletion.py` | Loads the four feature CSV files, merges them, runs Chi-square (`SelectKBest`) across all features, and exports a ranked contribution table. |
| `Peterpan_feature_threshold_justification.py` | Evaluates 13 feature thresholds (3%–100%) via 10-fold CV on SVM, LR, and NB; detects elbow points; runs paired *t*-tests against the 10% baseline; exports 7 publication-quality figures and summary CSVs. |
| `peterpan_kmeans.py` | Applies K-means clustering (*k* = 3) on the selected features, reduces dimensionality with PCA, and plots 95% confidence ellipses. |
| `Peterpan_SVM_NB_simpleRegression.py` | Runs SVM, LR, and NB with 10-fold CV, averages predicted probabilities into an ensemble, outputs a confusion matrix and per-sample classification table. |
| `Feature_Impo_Plotly.py` | Renders an interactive horizontal bar chart of the top-33 salient features with error bars, grouped by feature category. |

### Configuration

All scripts that read external data files use a `DATA_DIR` / `OUTPUT_DIR` variable at the top of the file. Set these to the appropriate local paths before running:

```python
DATA_DIR   = r"D:\path\to\your\data"   # directory containing the feature CSVs
OUTPUT_DIR = r"D:\path\to\your\output" # directory for saving results and figures
```

### Dependencies

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
plotly
kaleido        # for static PNG export from Plotly
pypinyin       # for rhyme-related lexical features
nltk
```

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn plotly kaleido pypinyin nltk
```

---

## Contact

For questions about the paper or requests for the full corpus, please contact the corresponding author:

**Kong Delu** — [GitHub: DeluDanielKong](https://github.com/DeluDanielKong)
