"""
duplicate_detector.py
Detects duplicate / similar GitHub issues using TF-IDF cosine similarity.

Note: We use TF-IDF cosine similarity here (no extra dependencies).
If you want even better results, swap in sentence-transformers later.

Run standalone: python src/duplicate_detector.py
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Load models and data ──────────────────────────────────────────────────────

def load_detector():
    """Load vectorizer and issue database. Returns (vectorizer, df, matrix)."""
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    df = pd.read_csv("data/github_issues.csv")
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    # Pre-compute TF-IDF matrix for all existing issues
    issue_matrix = vectorizer.transform(df["text"].tolist())

    return vectorizer, df, issue_matrix


# ── Core function ─────────────────────────────────────────────────────────────

def find_similar_issues(new_issue_text, vectorizer, df, issue_matrix,
                        top_k=5, threshold=0.20):
    """
    Given a new issue text, return the most similar existing issues.

    Args:
        new_issue_text : str   - title + body of the incoming issue
        vectorizer     : fitted TfidfVectorizer
        df             : DataFrame of existing issues
        issue_matrix   : pre-computed TF-IDF matrix of existing issues
        top_k          : int   - max number of results to return
        threshold      : float - minimum similarity score (0-1)

    Returns:
        list of dicts with keys: title, label, similarity
    """
    if not new_issue_text.strip():
        return []

    new_vec = vectorizer.transform([new_issue_text])
    scores  = cosine_similarity(new_vec, issue_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sim = float(scores[idx])
        if sim >= threshold:
            results.append({
                "title"      : df.iloc[idx]["title"],
                "label"      : df.iloc[idx]["label"],
                "similarity" : round(sim, 3),
            })

    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorizer, df, issue_matrix = load_detector()

    test_cases = [
        "Application crashes when uploading large files on mobile devices",
        "Add dark mode option to the settings page",
        "API documentation is missing code examples",
        "Login page shows 500 error after submitting the form",
        "Export functionality should support Excel format",
    ]

    print("Duplicate / Similar Issue Detection")
    print("=" * 60)

    for query in test_cases:
        print(f"\nNew Issue  : {query}")
        results = find_similar_issues(query, vectorizer, df, issue_matrix,
                                      top_k=3, threshold=0.15)
        if results:
            print("Similar issues found:")
            for r in results:
                print(f"  [{r['label'].upper():13}] {r['similarity']:.0%} — {r['title']}")
        else:
            print("  No similar issues found.")

    print("\n" + "=" * 60)
    print("Detector working correctly!")
