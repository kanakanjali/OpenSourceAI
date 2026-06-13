"""
train_classifier.py
Trains a GitHub issue classifier (bug / feature / documentation)
using TF-IDF + Logistic Regression.

Run: python src/train_classifier.py
"""

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── 1. Load Data ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/github_issues.csv")
print(f"Loaded {len(df)} issues")
print("Label distribution:")
print(df["label"].value_counts())
print()

# ── 2. Preprocess ─────────────────────────────────────────────────────────────

# Combine title + body — title carries strong signal, body adds context
df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

# Map labels to integers
label_map     = {"bug": 0, "feature": 1, "documentation": 2}
label_reverse = {0: "bug", 1: "feature", 2: "documentation"}
df["label_enc"] = df["label"].map(label_map)
df = df.dropna(subset=["label_enc"])

# ── 3. Train / Test Split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label_enc"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_enc"],      # keep class balance in both splits
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print()

# ── 4. TF-IDF Vectorization ───────────────────────────────────────────────────

vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1, 2),            # unigrams + bigrams
    sublinear_tf=True,             # apply log normalization to TF
    min_df=1,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ── 5. Train Model ────────────────────────────────────────────────────────────

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    random_state=42,
)

model.fit(X_train_vec, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["bug", "feature", "documentation"]
))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
print(f"{'':15} bug  feature  docs")
for i, row in enumerate(cm):
    label = ["bug", "feature", "docs"][i]
    print(f"{label:15} {row[0]:3}  {row[1]:7}  {row[2]:4}")
print()

# ── 7. Test with Sample Predictions ──────────────────────────────────────────

samples = [
    "App crashes when clicking submit button",
    "Add support for dark mode in settings",
    "Update API authentication examples in documentation",
    "TypeError thrown when input is null",
    "Allow users to export data as PDF",
    "Fix broken links in the getting started guide",
]

print("Sample Predictions:")
print("-" * 60)
for text in samples:
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    conf = max(prob) * 100
    print(f"  Input : {text[:55]}")
    print(f"  Label : {label_reverse[pred].upper()} ({conf:.1f}% confidence)")
    print()

# ── 8. Save Models ────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

with open("models/classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/label_map.pkl", "wb") as f:
    pickle.dump(label_reverse, f)

print("Models saved to models/")
print("  classifier.pkl")
print("  vectorizer.pkl")
print("  label_map.pkl")
