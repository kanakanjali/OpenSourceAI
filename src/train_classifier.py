"""
src/train_classifier.py — Train and evaluate the issue classifier.

Usage:
    python src/train_classifier.py

Outputs to models/:
    classifier.pkl   — trained Logistic Regression model
    vectorizer.pkl   — fitted TF-IDF vectorizer
    label_map.pkl    — {int: label_name} mapping derived from classifier.classes_

Improvements over v1:
  - 5-fold cross-validation for honest accuracy estimate on small dataset
  - class_weight='balanced' to handle any minor class imbalance
  - Larger TF-IDF vocab (max_features=5000) and trigrams
  - Prints per-class F1 breakdown
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "github_issues.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")

def train():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    X = df["text"].tolist()
    y = df["label"].tolist()

    print(f"Dataset: {len(df)} samples  |  Classes: {sorted(set(y))}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # 2. Vectorize — unigrams + bigrams + trigrams, sublinear TF scaling
    vectorizer = TfidfVectorizer(
        ngram_range   = (1, 3),
        sublinear_tf  = True,
        max_features  = 5000,
        stop_words    = "english",
        min_df        = 1,
    )
    X_vec = vectorizer.fit_transform(X)

    # 3. Classifier — balanced weights handle unequal class sizes
    clf = LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        class_weight = "balanced",
        solver       = "lbfgs",
    )

    # 4. 5-fold cross-validation (honest estimate on small datasets)
    cv_scores = cross_val_score(clf, X_vec, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Per-fold scores:     {[round(s, 4) for s in cv_scores]}\n")

    # 5. Final fit on all data + held-out test for the report
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Hold-out test set results:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))

    # 6. Refit on ALL data for deployment
    clf.fit(X_vec, y)

    # 7. Build label_map from classifier.classes_ (sorted alphabetically by sklearn)
    #    This is the ONLY correct source of truth — never hardcode label indices.
    label_map = {i: label for i, label in enumerate(clf.classes_)}
    print(f"Label map (from classifier.classes_): {label_map}")

    # 8. Save artefacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "label_map.pkl"), "wb") as f:
        pickle.dump(label_map, f)

    print("\n✅  Models saved to models/")
    print("    classifier.pkl  vectorizer.pkl  label_map.pkl")

if __name__ == "__main__":
    train()
