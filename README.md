# 🔍 OpenSourceAI

### Intelligent Open Source Maintainer & Contributor Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)]()

An ML-powered tool that automates GitHub issue triage using NLP and machine learning. Built to solve a real problem I personally experienced as an open source contributor.

> 🚀 **Live Demo:** https://opensourceaigit-wzlncnldkmhdvwjyectdmb.streamlit.app/

---

## 🎯 Problem

Open source maintainers spend significant time on repetitive triage tasks:

- Manually categorizing every incoming issue
- Identifying duplicate issues that have already been reported
- Figuring out which contributor is best suited for each issue

This scales poorly for large repositories receiving dozens of issues daily.

---

## 💡 Solution

OpenSourceAI automates three core maintainer workflows:

| Feature | How it works |
|---|---|
| **Issue Classification** | TF-IDF vectorization + Logistic Regression classifies issues as Bug / Feature / Documentation |
| **Duplicate Detection** | Cosine similarity on TF-IDF vectors surfaces semantically similar existing issues |
| **Contributor Recommendation** | Connects to the GitHub API to fetch real contributors and matches issue content to their skill profiles using cosine similarity |

---

## 🧠 ML Concepts Used

- **Text Vectorization** — TF-IDF with unigrams, bigrams, and trigrams; sublinear TF scaling
- **Multi-class Classification** — Logistic Regression with balanced class weights; evaluated using precision / recall / F1
- **Cosine Similarity** — Measures semantic closeness between issue texts and contributor profiles
- **Cross-Validation** — 5-fold CV for honest accuracy estimate on relatively small datasets
- **Feature Engineering** — Title + body concatenation, stop word removal, n-gram features

---

## 📁 Project Structure

```
OpenSourceAI/
│
├── data/
│   ├── github_issues.csv          # Training dataset (300 labeled issues)
│   └── generate_dataset.py        # Script to regenerate / expand dataset
│
├── models/
│   ├── classifier.pkl             # Trained Logistic Regression model
│   ├── vectorizer.pkl             # Fitted TF-IDF vectorizer
│   └── label_map.pkl              # Integer to label mapping (from classifier.classes_)
│
├── src/
│   ├── train_classifier.py        # Feature 1: trains and evaluates classifier
│   ├── duplicate_detector.py      # Feature 2: finds similar issues
│   └── contributor_recommender.py # Feature 3: recommends contributors
│
├── .streamlit/
│   └── config.toml                # Dark theme configuration
│
├── .devcontainer/
│   └── devcontainer.json          # One-click GitHub Codespaces setup
│
├── app.py                         # Streamlit dashboard (all 3 features)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/kanakanjali/OpenSourceAI.git
cd OpenSourceAI
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Generate data + train models** *(skip if pre-trained `.pkl` files are already in `models/`)*

```bash
python data/generate_dataset.py   # creates data/github_issues.csv
python src/train_classifier.py    # creates models/*.pkl
```

**4. Run the app**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🐙 GitHub Integration

The sidebar lets you connect any public GitHub repository to fetch **real contributors** instead of demo profiles.

1. Enter a repo in `owner/repo` format — e.g. `facebook/react`
2. Click **Load Contributors**
3. The app fetches real contributor bios, languages, and commit counts via the GitHub REST API
4. Issue recommendations now match against actual people with clickable GitHub profile links

> **Optional:** Add a Personal Access Token to increase the API rate limit from 60 → 5,000 requests/hour. Create one at [github.com/settings/tokens](https://github.com/settings/tokens) — no scopes needed for public repos.

---

## 📊 Model Performance

| Metric | v1 (120 samples) | v2 (300 samples) |
|---|---|---|
| Hold-out Accuracy | — | **87.5%** |
| Bug F1 | 0.80 | **0.87** |
| Feature F1 | 0.82 | **0.90** |
| Documentation F1 | 1.00 | **0.97** |
| CV Folds | — | 5-fold |

Improvements in v2: expanded dataset (300 vs 120 samples), trigrams, balanced class weights, 5-fold cross-validation.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn — TF-IDF, Logistic Regression, cosine similarity |
| Data | Pandas, NumPy |
| Frontend | Streamlit |
| API | GitHub REST API v3 |

---

## 💬 Motivation

As a contributor to open source projects through GSSOC, I noticed maintainers manually triaging the same types of issues repeatedly. This project automates those workflows using core NLP and ML techniques.

---

## 📌 Future Work

- [ ] Replace TF-IDF with Sentence Transformers for better semantic understanding
- [ ] Add FAISS vector index for faster similarity search at scale
- [ ] Add confusion matrix and EDA visualizations inside the app
- [ ] Support private repositories via OAuth

---

*Automating open source maintainer workflows with NLP and machine learning.*
