"""
app.py — OpenSourceAI Streamlit Dashboard
Combines all 3 ML features into one clean web interface.
Run: streamlit run app.py
"""

import time
import requests
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Mock contributor profiles (fallback when no GitHub repo connected) ─────────
MOCK_CONTRIBUTORS = [
    {"username": "Ishan_dev",       "skills": "python backend api authentication login security jwt oauth flask django rest",           "languages": ["Python", "SQL"],                        "bio": "Backend developer. Python, Flask, Django. API design and authentication.",  "contributions": 0, "profile_url": ""},
    {"username": "Pranay_frontend", "skills": "javascript react css html ui frontend design dark mode mobile responsive",               "languages": ["JavaScript", "TypeScript", "CSS"],      "bio": "Frontend engineer. React, CSS, UI/UX. Mobile and responsive design.",       "contributions": 0, "profile_url": ""},
    {"username": "Dhruv_ml",        "skills": "machine learning nlp python scikit-learn tensorflow model training classification",      "languages": ["Python", "R"],                          "bio": "ML engineer. NLP, classification, recommendation systems.",                 "contributions": 0, "profile_url": ""},
    {"username": "Prince_devops",   "skills": "docker kubernetes ci cd deployment linux bash server infrastructure performance",       "languages": ["Bash", "YAML", "Python"],               "bio": "DevOps engineer. Docker, Kubernetes, CI/CD pipelines.",                    "contributions": 0, "profile_url": ""},
    {"username": "eve_writer",      "skills": "documentation writing api docs markdown tutorial guide readme technical writing",        "languages": ["Markdown"],                             "bio": "Technical writer. API docs, tutorials, contributor guides.",               "contributions": 0, "profile_url": ""},
    {"username": "frank_db",        "skills": "database sql postgresql mongodb redis query optimization schema migration",              "languages": ["SQL", "Python"],                        "bio": "Database engineer. PostgreSQL, MongoDB, query optimization.",              "contributions": 0, "profile_url": ""},
    {"username": "grace_mobile",    "skills": "ios android mobile react native swift kotlin push notifications",                       "languages": ["Swift", "Kotlin", "JavaScript"],        "bio": "Mobile developer. iOS, Android, React Native.",                           "contributions": 0, "profile_url": ""},
    {"username": "henry_security",  "skills": "security vulnerability xss csrf injection penetration testing audit permissions",       "languages": ["Python", "Bash"],                       "bio": "Security engineer. Vulnerability assessment, secure code review.",        "contributions": 0, "profile_url": ""},
]

LABEL_EMOJI  = {"bug": "🐛", "feature": "✨", "documentation": "📄"}
LABEL_COLOR  = {"bug": "#ef4444", "feature": "#8b5cf6", "documentation": "#10b981"}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OpenSourceAI", page_icon="🔍", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base */
.stApp {
    background: #080e1c;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
h1, h2, h3, h4, h5, h6 { color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important; }

/* Hero header */
.hero-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e2d4f; border-radius: 16px;
    padding: 2rem 2.4rem 1.6rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-card::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(99,102,241,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.main-title {
    font-family: 'Inter', sans-serif; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #c7d2fe 0%, #a78bfa 55%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.35rem 0;
    line-height: 1.2; letter-spacing: -0.02em;
}
.subtitle { color: #64748b; font-size: 0.95rem; margin: 0; font-weight: 400; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d4f !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1; }
[data-testid="stSidebar"] h2 { color: #e2e8f0 !important; font-size: 1.1rem !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #111827; border: 1px solid #1e2d4f; border-radius: 12px; padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }

/* Dividers */
hr { border: none !important; border-top: 1px solid #1a2540 !important; margin: 1.5rem 0 !important; }

/* Inputs */
.stTextInput label, .stTextArea label, .stSelectbox label {
    color: #94a3b8 !important; font-weight: 500 !important; font-size: 0.84rem !important;
}
.stTextInput > div > div > input {
    background: #111827 !important; color: #e2e8f0 !important;
    border: 1px solid #1e2d4f !important; border-radius: 8px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
}
.stTextArea > div > div > textarea {
    background: #111827 !important; color: #e2e8f0 !important;
    border: 1px solid #1e2d4f !important; border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: #111827 !important; color: #e2e8f0 !important; border: 1px solid #1e2d4f !important; border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 1rem !important; height: 3rem;
    box-shadow: 0 4px 20px rgba(99,102,241,0.28) !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.45) !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
}

/* Progress bars */
.stProgress > div > div > div { background: #1e2740 !important; border-radius: 6px !important; height: 8px !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #6366f1, #a78bfa) !important; border-radius: 6px !important; }

/* Code chips */
code {
    background: #1a2035 !important; color: #7dd3fc !important;
    border: 1px solid #2d3d6e !important; border-radius: 5px !important;
    padding: 2px 7px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82em !important;
}

/* Label badge */
.label-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 7px 20px;
    border-radius: 100px; font-weight: 700; font-size: 0.92rem; color: white;
    margin-bottom: 1rem; letter-spacing: 0.07em; text-transform: uppercase;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45);
}

/* Contributor cards */
.contributor-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
    border: 1px solid #1e2d4f; border-left: 3px solid #6366f1;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.contributor-card:hover { transform: translateX(3px); box-shadow: 0 4px 20px rgba(99,102,241,0.15); }
.contributor-card b { color: #c7d2fe; }
.contributor-card small { color: #64748b; }

/* Similar issue cards */
.similar-card {
    background: linear-gradient(135deg, #18140a 0%, #140f07 100%);
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
    border: 1px solid #38290f; border-left: 3px solid #f59e0b;
    transition: transform 0.15s ease;
}
.similar-card:hover { transform: translateX(3px); }
.similar-card b { color: #fde68a; }
.similar-card small { color: #a16207; }

/* Footer */
.footer { text-align: center; color: #334155; font-size: 0.75rem; letter-spacing: 0.05em; padding: 0.5rem 0; }
.footer .dot { color: #6366f1; margin: 0 6px; }
</style>
""", unsafe_allow_html=True)

# ── Load ML models ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_all():
    try:
        with open("models/classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open("models/vectorizer.pkl", "rb") as f:
            issue_vectorizer = pickle.load(f)
        with open("models/label_map.pkl", "rb") as f:
            label_map = pickle.load(f)
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found. Please run the training script first:\n\n"
            "```bash\npython src/train_classifier.py\n```"
        )
        st.stop()

    df = pd.read_csv("data/github_issues.csv")
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    issue_matrix = issue_vectorizer.transform(df["text"].tolist())
    return classifier, issue_vectorizer, label_map, df, issue_matrix

(classifier, issue_vectorizer, label_map, issues_df, issue_matrix) = load_all()

# ── GitHub API — fetch real contributors ───────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_github_contributors(repo: str, token: str = ""):
    """
    Fetch real contributors from a GitHub repo via the public API.
    Returns (list_of_contributors, error_message).
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/contributors",
            headers=headers, params={"per_page": 10}, timeout=10,
        )
    except requests.RequestException as e:
        return None, f"Network error: {e}"

    if resp.status_code == 404:
        return None, "Repository not found. Use the format `owner/repo`."
    if resp.status_code == 403:
        return None, "API rate limit reached. Add a GitHub token in the sidebar for higher limits."
    if resp.status_code != 200:
        error_msg = resp.json().get("message", "Unknown error")
        return None, f"GitHub API error {resp.status_code}: {error_msg}"

    raw = [c for c in resp.json() if c.get("type") != "Bot"]
    if not raw:
        return None, "No human contributors found in this repository."

    contributors = []
    for c in raw[:8]:
        username = c.get("login", "unknown")
        bio = ""

        # Get bio from user profile
        try:
            ur = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=8)
            if ur.status_code == 200:
                bio = ur.json().get("bio") or ""
        except requests.RequestException:
            pass

        # Get repos to infer languages + skills
        skills_parts = [
            username.lower().replace("-", " ").replace("_", " "),
            bio.lower(),
        ]
        languages = []
        try:
            rr = requests.get(
                f"https://api.github.com/users/{username}/repos",
                headers=headers, params={"per_page": 10, "sort": "updated"}, timeout=8,
            )
            if rr.status_code == 200:
                lang_set = set()
                for repo_item in rr.json():
                    if repo_item.get("language"):
                        lang_set.add(repo_item["language"])
                    if repo_item.get("description"):
                        skills_parts.append((repo_item["description"] or "").lower())
                    if repo_item.get("name"):
                        skills_parts.append(repo_item["name"].lower().replace("-", " ").replace("_", " "))
                languages = list(lang_set)[:5]
        except requests.RequestException:
            pass

        contributors.append({
            "username":      username,
            "skills":        " ".join(filter(None, skills_parts)) or username,
            "languages":     languages or ["Unknown"],
            "bio":           bio or f"Contributor with {c.get('contributions', 0)} commits",
            "contributions": c.get("contributions", 0),
            "profile_url":   f"https://github.com/{username}",
        })
        time.sleep(0.05)

    return (contributors, None) if contributors else (None, "Could not load any contributor profiles.")

# ── Dynamic contributor TF-IDF index ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_contrib_index(skills_tuple: tuple):
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(list(skills_tuple))
    return vec, mat

# ── Session state ──────────────────────────────────────────────────────────────
if "active_contributors" not in st.session_state: st.session_state.active_contributors = MOCK_CONTRIBUTORS
if "using_real_github"   not in st.session_state: st.session_state.using_real_github   = False
if "connected_repo"      not in st.session_state: st.session_state.connected_repo      = ""

# ── Sidebar — GitHub Integration ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐙 GitHub Integration")
    st.markdown("Connect any public GitHub repo to match issues against **real contributors**.")
    st.markdown("---")

    github_repo  = st.text_input("Repository",  placeholder="e.g. facebook/react",
                                  help="Format: owner/repo")
    github_token = st.text_input("Personal Access Token (optional)", type="password",
                                  help="Raises rate limit from 60 → 5,000 req/hr.\nCreate one at github.com/settings/tokens (no scopes needed for public repos).")
    load_btn = st.button("🔄 Load Contributors", use_container_width=True)

    if load_btn:
        if not github_repo.strip():
            st.warning("Enter a repository name first.")
        else:
            with st.spinner(f"Fetching from {github_repo.strip()} …"):
                contribs, err = fetch_github_contributors(github_repo.strip(), github_token.strip())
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.active_contributors = contribs
                st.session_state.using_real_github   = True
                st.session_state.connected_repo      = github_repo.strip()
                st.success(f"✅ Loaded **{len(contribs)}** contributors from `{github_repo.strip()}`")

    if st.session_state.using_real_github:
        st.markdown("---")
        st.markdown(f"**Connected:** `{st.session_state.connected_repo}`")
        if st.button("🔌 Disconnect / Use Demo", use_container_width=True):
            st.session_state.active_contributors = MOCK_CONTRIBUTORS
            st.session_state.using_real_github   = False
            st.session_state.connected_repo      = ""
            st.rerun()

    st.markdown("---")
    st.markdown("**API rate limits**")
    st.markdown("• No token → 60 req / hour")
    st.markdown("• With token → 5,000 req / hour")

# ── Active contributors & index ────────────────────────────────────────────────
active_contribs  = st.session_state.active_contributors
skills_tuple     = tuple(c["skills"] for c in active_contribs)
contrib_vectorizer, contrib_matrix = build_contrib_index(skills_tuple)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="hero-card">
    <p class="main-title">🔍 OpenSourceAI</p>
    <p class="subtitle">Intelligent Open Source Maintainer &amp; Contributor Assistant</p>
</div>
''', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Issues in Database", len(issues_df))
if st.session_state.using_real_github:
    col_b.metric("Contributors Indexed", len(active_contribs), delta="Live from GitHub")
else:
    col_b.metric("Contributors Indexed", len(active_contribs))
col_c.metric("Model Accuracy", "87.5%")
st.markdown("---")

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("### 📝 Paste a GitHub Issue")
col1, col2 = st.columns([1, 1])
with col1:
    issue_title      = st.text_input("Issue Title *", placeholder="e.g. App crashes when uploading large files")
with col2:
    issue_label_hint = st.selectbox("Actual label (optional — for comparison)", ["Unknown", "bug", "feature", "documentation"])
issue_body = st.text_area("Issue Body (optional but improves accuracy)",
                           placeholder="Describe the issue in detail...", height=120)
analyze_btn = st.button("🔍 Analyze Issue", type="primary", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_btn:
    if not issue_title.strip():
        st.warning("⚠️ Please enter an issue title to analyze.")
        st.stop()

    combined_text = issue_title.strip() + " " + issue_body.strip()
    st.markdown("---")

    # Row 1: Classification + Keywords
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🏷️ Issue Classification")
        vec        = issue_vectorizer.transform([combined_text])
        pred_label = str(classifier.predict(vec)[0])
        proba      = classifier.predict_proba(vec)[0]

        # classifier.classes_ is the authoritative order for proba indices
        pred_idx   = list(classifier.classes_).index(pred_label)
        confidence = float(proba[pred_idx]) * 100
        color      = LABEL_COLOR[pred_label]
        emoji      = LABEL_EMOJI[pred_label]

        st.markdown(
            f'<div class="label-badge" style="background:{color}">'
            f'{emoji} {pred_label.upper()} — {confidence:.1f}% confident</div>',
            unsafe_allow_html=True,
        )

        if issue_label_hint != "Unknown":
            if issue_label_hint == pred_label:
                st.success("✅ Matches your hint!")
            else:
                st.info(f"ℹ️ You selected '{issue_label_hint}', model predicted '{pred_label}'.")

        st.markdown("**Confidence breakdown:**")

        # classifier.classes_ is the authoritative label order (sorted alphabetically
        # by sklearn). proba[idx] lines up exactly with classes_[idx].
        for idx, lname in enumerate(classifier.classes_):
            p = float(proba[idx])
            st.markdown(f"{LABEL_EMOJI[lname]} **{lname}**")
            st.progress(p, text=f"{p * 100:.1f}%")

    with c2:
        st.markdown("#### 📊 What the model used")
        feature_names = issue_vectorizer.get_feature_names_out()
        issue_tfidf   = vec.toarray()[0]
        top_feat_idx  = np.argsort(issue_tfidf)[::-1][:10]
        top_features  = [(feature_names[i], round(issue_tfidf[i], 3))
                         for i in top_feat_idx if issue_tfidf[i] > 0]
        if top_features:
            st.markdown("**Top keywords detected:**")
            for word, score in top_features[:8]:
                st.markdown(f"`{word}` — weight: {score:.3f}")
        else:
            st.info("No strong keywords found. Try adding more detail.")

    st.markdown("---")

    # Row 2: Duplicates + Contributors
    c3, c4 = st.columns([1, 1])

    with c3:
        st.markdown("#### 🔁 Similar / Duplicate Issues")
        scores  = cosine_similarity(vec, issue_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:5]
        shown   = 0
        for idx in top_idx:
            sim = float(scores[idx])
            if sim >= 0.15:
                row   = issues_df.iloc[idx]
                lname = row["label"]
                st.markdown(
                    f'<div class="similar-card">'
                    f'<b>{LABEL_EMOJI[lname]} {row["title"]}</b><br>'
                    f'<small>Label: <b>{lname}</b> &nbsp;|&nbsp; Similarity: <b>{sim:.0%}</b></small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                shown += 1
        if shown == 0:
            st.success("✅ No similar issues found. This looks like a new issue!")

    with c4:
        st.markdown("#### 👥 Recommended Contributors")
        if st.session_state.using_real_github:
            st.caption(f"🐙 Real contributors from `{st.session_state.connected_repo}`")
        else:
            st.caption("💡 Demo profiles — connect a GitHub repo in the sidebar for real data.")

        issue_cv      = contrib_vectorizer.transform([combined_text])
        contrib_scores = cosine_similarity(issue_cv, contrib_matrix)[0]
        top_cidx      = np.argsort(contrib_scores)[::-1][:3]
        shown_c       = 0

        for idx in top_cidx:
            score = float(contrib_scores[idx])
            if score >= 0.05:
                c        = active_contribs[idx]
                langs    = " &nbsp;".join([f"`{l}`" for l in c["languages"]])
                profile_url = c.get("profile_url", "")
                commits     = c.get("contributions", 0)
                extra       = f" &nbsp;·&nbsp; {commits} commits" if commits else ""
                name_html   = (
                    f'<a href="{profile_url}" target="_blank" '
                    f'style="color:#c7d2fe;text-decoration:none;">@{c["username"]}</a>'
                    if profile_url else f'@{c["username"]}'
                )
                st.markdown(
                    f'<div class="contributor-card">'
                    f'<b>{name_html}</b> &nbsp; Match: <b>{score:.0%}</b>{extra}<br>'
                    f'<small>{c["bio"]}</small><br>'
                    f'<small>{langs}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                shown_c += 1

        if shown_c == 0:
            st.info("No strong contributor match found.")

    st.markdown("---")

    # Summary
    st.markdown("#### 📋 Summary")

    # Reuse contrib_scores already computed above (avoids a redundant cosine_similarity call)
    best_idx        = int(np.argmax(contrib_scores))
    top_contributor = (
        f"@{active_contribs[best_idx]['username']}"
        if float(contrib_scores[best_idx]) >= 0.05 else "No match"
    )

    st.success(
        f"**Label:** {LABEL_EMOJI[pred_label]} {pred_label.upper()} \n"
        f"**Confidence:** {confidence:.1f}% \n"
        f"**Duplicates found:** {'Yes' if shown > 0 else 'No'} \n"
        f"**Top contributor:** {top_contributor}"
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">'
    'OpenSourceAI'
    '<span class="dot">·</span>NLP'
    '<span class="dot">·</span>Classification'
    '<span class="dot">·</span>Similarity Search'
    '<span class="dot">·</span>Contributor Recommendation'
    '</div>',
    unsafe_allow_html=True,
)
