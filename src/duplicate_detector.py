"""
src/duplicate_detector.py — Standalone duplicate/similarity detection.

Usage:
    from src.duplicate_detector import DuplicateDetector
    detector = DuplicateDetector("data/github_issues.csv")
    results  = detector.find_similar("App crashes on login", top_k=5, threshold=0.15)
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateDetector:
    def __init__(self, csv_path: str, vectorizer_path: str = "models/vectorizer.pkl"):
        self.df = pd.read_csv(csv_path)
        self.df["text"] = self.df["title"].fillna("") + " " + self.df["body"].fillna("")

        # Reuse the same vectorizer fitted during training for consistency
        try:
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            # Fallback: fit a fresh vectorizer on the dataset
            self.vectorizer = TfidfVectorizer(
                ngram_range=  (1, 3),
                sublinear_tf= True,
                max_features= 5000,
                stop_words=   "english",
            )
            self.vectorizer.fit(self.df["text"].tolist())

        self.issue_matrix = self.vectorizer.transform(self.df["text"].tolist())

    def find_similar(self, query: str, top_k: int = 5, threshold: float = 0.15):
        """
        Return up to top_k issues whose cosine similarity to `query` is >= threshold.

        Returns a list of dicts with keys: title, label, similarity.
        """
        vec    = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.issue_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            sim = float(scores[idx])
            if sim >= threshold:
                row = self.df.iloc[idx]
                results.append({
                    "title":      row["title"],
                    "label":      row["label"],
                    "similarity": round(sim, 4),
                })
        return results


if __name__ == "__main__":
    detector = DuplicateDetector("data/github_issues.csv")
    query    = "App crashes when uploading large files"
    results  = detector.find_similar(query)
    print(f"Issues similar to: '{query}'\n")
    for r in results:
        print(f"  [{r['label']:13}] {r['similarity']:.0%}  —  {r['title']}")
