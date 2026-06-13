"""
contributor_recommender.py
Recommends contributors for a GitHub issue based on their skill profile
and the issue content using TF-IDF cosine similarity.

Run standalone: python src/contributor_recommender.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Mock contributor profiles ─────────────────────────────────────────────────
# In a real project, fetch these from GitHub API:
#   GET /repos/{owner}/{repo}/contributors
#   GET /users/{username} for bio and repos

CONTRIBUTORS = [
    {
        "username"  : "Ishan_dev",
        "skills"    : "python backend api authentication login security jwt oauth flask django rest",
        "languages" : ["Python", "SQL"],
        "bio"       : "Backend developer. Python, Flask, Django. API design and authentication.",
    },
    {
        "username"  : "Pranay_frontend",
        "skills"    : "javascript react css html ui frontend design dark mode mobile responsive",
        "languages" : ["JavaScript", "TypeScript", "CSS"],
        "bio"       : "Frontend engineer. React, CSS, UI/UX. Mobile and responsive design.",
    },
    {
        "username"  : "Dhruv_ml",
        "skills"    : "machine learning nlp python scikit-learn tensorflow model training classification",
        "languages" : ["Python", "R"],
        "bio"       : "ML engineer. NLP, classification, recommendation systems.",
    },
    {
        "username"  : "Prince_devops",
        "skills"    : "docker kubernetes ci cd deployment linux bash server infrastructure performance",
        "languages" : ["Bash", "YAML", "Python"],
        "bio"       : "DevOps engineer. Docker, Kubernetes, CI/CD pipelines.",
    },
    {
        "username"  : "eve_writer",
        "skills"    : "documentation writing api docs markdown tutorial guide readme technical writing",
        "languages" : ["Markdown"],
        "bio"       : "Technical writer. API docs, tutorials, contributor guides.",
    },
    {
        "username"  : "frank_db",
        "skills"    : "database sql postgresql mongodb redis query optimization schema migration",
        "languages" : ["SQL", "Python"],
        "bio"       : "Database engineer. PostgreSQL, MongoDB, query optimization.",
    },
    {
        "username"  : "grace_mobile",
        "skills"    : "ios android mobile react native swift kotlin push notifications",
        "languages" : ["Swift", "Kotlin", "JavaScript"],
        "bio"       : "Mobile developer. iOS, Android, React Native.",
    },
    {
        "username"  : "henry_security",
        "skills"    : "security vulnerability xss csrf injection penetration testing audit permissions",
        "languages" : ["Python", "Bash"],
        "bio"       : "Security engineer. Vulnerability assessment, secure code review.",
    },
]


# ── Core recommendation logic ─────────────────────────────────────────────────

def build_contributor_index(contributors):
    """Fit a TF-IDF vectorizer over all contributor skill profiles."""
    skill_docs = [c["skills"] for c in contributors]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(skill_docs)
    return vec, matrix


def recommend_contributors(issue_text, contributors, vectorizer, matrix,
                            top_k=3, threshold=0.05):
    """
    Match an issue to the best-fit contributors.

    Args:
        issue_text   : str   - combined issue title + body
        contributors : list  - list of contributor dicts
        vectorizer   : fitted TfidfVectorizer
        matrix       : TF-IDF matrix of contributor skill docs
        top_k        : int   - number of recommendations
        threshold    : float - minimum match score

    Returns:
        list of dicts: username, score, bio, languages
    """
    issue_vec = vectorizer.transform([issue_text])
    scores    = cosine_similarity(issue_vec, matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= threshold:
            c = contributors[idx]
            results.append({
                "username"  : c["username"],
                "score"     : round(score, 3),
                "bio"       : c["bio"],
                "languages" : c["languages"],
            })

    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorizer, matrix = build_contributor_index(CONTRIBUTORS)

    test_issues = [
        ("Login fails with correct credentials on Safari",
         "Users on Safari browser cannot log in. The form submits but redirects back to login page with no error."),
        ("Add dark mode support",
         "It would be great to have a dark mode option that respects system preference and saves the setting."),
        ("Update API authentication examples in documentation",
         "The API docs are missing examples for Bearer token auth in Python and JavaScript."),
        ("Memory leak in background sync service",
         "The background sync service gradually consumes more memory over 6 hours causing the process to be killed."),
    ]

    print("Contributor Recommendation Engine")
    print("=" * 60)

    for title, body in test_issues:
        issue_text = title + " " + body
        print(f"\nIssue: {title}")

        recs = recommend_contributors(issue_text, CONTRIBUTORS, vectorizer, matrix,
                                      top_k=3, threshold=0.05)

        if recs:
            print("Recommended Contributors:")
            for i, r in enumerate(recs, 1):
                langs = ", ".join(r["languages"])
                print(f"  {i}. @{r['username']} (score: {r['score']:.2f})")
                print(f"     {r['bio']}")
                print(f"     Languages: {langs}")
        else:
            print("  No matching contributors found.")

    print("\n" + "=" * 60)
    print("Recommender working correctly!")
