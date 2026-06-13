"""
src/contributor_recommender.py — Standalone contributor recommendation.

Usage:
    from src.contributor_recommender import ContributorRecommender
    rec     = ContributorRecommender(contributors)
    results = rec.recommend("App crashes on login", top_k=3)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ContributorRecommender:
    def __init__(self, contributors: list):
        """
        contributors: list of dicts with at least 'username' and 'skills' keys.
        """
        self.contributors = contributors
        self.vectorizer   = TfidfVectorizer(stop_words="english")
        skills            = [c["skills"] for c in contributors]
        self.matrix       = self.vectorizer.fit_transform(skills)

    def recommend(self, issue_text: str, top_k: int = 3, threshold: float = 0.05):
        """
        Return up to top_k contributors ranked by cosine similarity to issue_text.
        """
        vec    = self.vectorizer.transform([issue_text])
        scores = cosine_similarity(vec, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score >= threshold:
                c = self.contributors[idx]
                results.append({
                    "username":      c["username"],
                    "match_score":   round(score, 4),
                    "bio":           c.get("bio", ""),
                    "languages":     c.get("languages", []),
                    "contributions": c.get("contributions", 0),
                    "profile_url":   c.get("profile_url", ""),
                })
        return results


if __name__ == "__main__":
    # Inline demo profiles so this script runs without importing Streamlit via app.py
    _DEMO_CONTRIBUTORS = [
        {"username": "Ishan_dev",       "skills": "python backend api authentication login security jwt oauth flask django rest",      "languages": ["Python", "SQL"],               "bio": "Backend developer. Python, Flask, Django.",    "contributions": 0, "profile_url": ""},
        {"username": "Pranay_frontend", "skills": "javascript react css html ui frontend design dark mode mobile responsive",          "languages": ["JavaScript", "TypeScript"],     "bio": "Frontend engineer. React, CSS, UI/UX.",        "contributions": 0, "profile_url": ""},
        {"username": "Dhruv_ml",        "skills": "machine learning nlp python scikit-learn tensorflow model training classification", "languages": ["Python", "R"],                  "bio": "ML engineer. NLP, classification.",            "contributions": 0, "profile_url": ""},
        {"username": "Prince_devops",   "skills": "docker kubernetes ci cd deployment linux bash server infrastructure performance",   "languages": ["Bash", "YAML", "Python"],       "bio": "DevOps engineer. Docker, Kubernetes, CI/CD.", "contributions": 0, "profile_url": ""},
    ]
    rec     = ContributorRecommender(_DEMO_CONTRIBUTORS)
    query   = "App crashes when uploading large files on mobile"
    results = rec.recommend(query)
    print(f"Top contributors for: '{query}'\n")
    for r in results:
        print(f"  @{r['username']:20}  match={r['match_score']:.0%}  —  {r['bio']}")
