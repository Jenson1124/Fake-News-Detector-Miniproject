"""
app.py — Fake News Detector | Streamlit Web Application
"""

import json, os, re, string, warnings, datetime
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

print("APP STARTED")

# ── Auto-train on first run ───────────────────────────────────────────────────
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.error(
        "Model files not found. Please upload model.pkl and vectorizer.pkl to GitHub."
    )
    st.stop()

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        STOP_WORDS = set()

STOP_WORDS.update({
    "reuters", "ap", "afp", "said", "would",
    "also", "one", "two", "three", "new",
    "year", "say", "told", "added",
    "according", "pti", "ani", "ians"
})

try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except Exception:
    NEWS_API_KEY = ""


SOURCE_META = {
    "the-hindu":           ("🇮🇳", "The Hindu"),
    "ndtv":                ("🇮🇳", "NDTV"),
    "the-times-of-india":  ("🇮🇳", "Times of India"),
    "india-today":         ("🇮🇳", "India Today"),
    "hindustan-times":     ("🇮🇳", "Hindustan Times"),
    "the-indian-express":  ("🇮🇳", "Indian Express"),
    "reuters":             ("🇺🇸", "Reuters"),
    "associated-press":    ("🇺🇸", "AP"),
    "the-washington-post": ("🇺🇸", "Washington Post"),
    "the-new-york-times":  ("🇺🇸", "NY Times"),
    "bbc-news":            ("🇬🇧", "BBC News"),
    "cnn":                 ("🇺🇸", "CNN"),
    "al-jazeera-english":  ("🌍",  "Al Jazeera"),
}

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }

.app-header { text-align:center; padding:2.2rem 1rem 1.6rem; border-bottom:1.5px solid #e8ecf0; margin-bottom:2rem; }
.app-header h1 { font-size:2.4rem; font-weight:700; color:#0f1923; letter-spacing:-0.5px; margin:0 0 0.4rem; }
.app-header p  { font-size:1rem; color:#64748b; margin:0; font-weight:400; }
.header-badge  { display:inline-block; background:#f0f4ff; color:#3b5bdb; font-size:0.72rem;
                 font-weight:600; letter-spacing:0.8px; text-transform:uppercase; padding:3px 10px;
                 border-radius:20px; margin-bottom:0.8rem; font-family:'JetBrains Mono',monospace; }

.result-real { background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
               border:1.5px solid #86efac; border-left:5px solid #16a34a;
               border-radius:12px; padding:1.4rem 1.8rem; margin:1rem 0; }
.result-fake { background:linear-gradient(135deg,#fff7f7 0%,#fee2e2 100%);
               border:1.5px solid #fca5a5; border-left:5px solid #dc2626;
               border-radius:12px; padding:1.4rem 1.8rem; margin:1rem 0; }
.result-label { font-size:1.6rem; font-weight:700; margin:0 0 0.3rem; letter-spacing:-0.3px; }
.result-real .result-label { color:#15803d; }
.result-fake .result-label { color:#b91c1c; }
.result-sub { font-size:0.9rem; color:#64748b; margin:0; font-weight:400; }

.conf-wrap  { margin:1.2rem 0 0.6rem; }
.conf-label { font-size:0.78rem; font-weight:600; color:#94a3b8; text-transform:uppercase;
              letter-spacing:0.8px; margin-bottom:6px; font-family:'JetBrains Mono',monospace; }
.conf-bar-bg       { background:#e2e8f0; border-radius:8px; height:10px; width:100%; overflow:hidden; }
.conf-bar-fill-real { background:linear-gradient(90deg,#16a34a,#4ade80); height:10px; border-radius:8px; }
.conf-bar-fill-fake { background:linear-gradient(90deg,#dc2626,#f87171); height:10px; border-radius:8px; }
.conf-pct      { font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin-top:0.3rem; }
.conf-pct-real { color:#16a34a; }
.conf-pct-fake { color:#dc2626; }

.prob-row  { display:flex; gap:12px; margin-top:1rem; }
.prob-pill { flex:1; text-align:center; padding:0.6rem 1rem; border-radius:8px;
             border:1.5px solid #e2e8f0; background:#f8fafc; }
.prob-pill-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.8px;
                   font-weight:600; color:#94a3b8; font-family:'JetBrains Mono',monospace; }
.prob-pill-val   { font-size:1.3rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin-top:2px; }
.prob-fake-val   { color:#dc2626; }
.prob-real-val   { color:#16a34a; }

.lime-section { background:#f8fafc; border:1.5px solid #e2e8f0; border-radius:12px;
                padding:1.2rem 1.4rem; margin:1.2rem 0; }
.lime-title   { font-size:0.85rem; font-weight:600; color:#0f1923; margin:0 0 0.3rem;
                text-transform:uppercase; letter-spacing:0.6px; font-family:'JetBrains Mono',monospace; }
.lime-legend  { font-size:0.8rem; color:#64748b; margin:0 0 1rem; }
.lime-text    { font-size:0.95rem; line-height:2.2; color:#1e293b; }
.lime-word    { padding:2px 5px; border-radius:4px; margin:0 1px; font-size:0.93rem; }
.lime-fake-strong { background:#fecaca; color:#7f1d1d; font-weight:600; }
.lime-fake-mild   { background:#fee2e2; color:#991b1b; }
.lime-real-strong { background:#bbf7d0; color:#14532d; font-weight:600; }
.lime-real-mild   { background:#dcfce7; color:#166534; }
.lime-neutral     { color:#334155; }

.news-card           { background:#fff; border:1.5px solid #e2e8f0; border-radius:10px;
                       padding:1rem 1.2rem; margin-bottom:0.8rem; }
.news-card-real      { border-left:4px solid #16a34a; }
.news-card-fake      { border-left:4px solid #dc2626; }
.news-card-uncertain { border-left:4px solid #d97706; }
.news-headline { font-size:0.95rem; font-weight:600; color:#0f1923; margin:0 0 0.4rem; }
.news-meta     { font-size:0.78rem; color:#94a3b8; margin:0 0 0.5rem;
                 font-family:'JetBrains Mono',monospace; }
.news-badge    { display:inline-block; font-size:0.72rem; font-weight:700; padding:2px 9px;
                 border-radius:20px; font-family:'JetBrains Mono',monospace; text-transform:uppercase; }
.badge-real      { background:#dcfce7; color:#16a34a; }
.badge-fake      { background:#fee2e2; color:#dc2626; }
.badge-uncertain { background:#fef3c7; color:#92400e; }

.hist-section { margin-top:2rem; border-top:1.5px solid #e8ecf0; padding-top:1.4rem; }
.hist-title   { font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;
                color:#64748b; font-family:'JetBrains Mono',monospace; margin-bottom:0.8rem; }

.sidebar-metric       { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                        padding:0.8rem 1rem; margin-bottom:0.6rem; text-align:center; }
.sidebar-metric-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.8px;
                        color:#94a3b8; font-weight:600; font-family:'JetBrains Mono',monospace; }
.sidebar-metric-val   { font-size:1.5rem; font-weight:700; color:#0f1923;
                        font-family:'JetBrains Mono',monospace; }

div[data-testid="stTabs"] button {
    font-family:'Inter',sans-serif; font-weight:500; font-size:0.9rem; }
div[data-testid="stButton"] button[kind="primary"] {
    background:#1d4ed8; border:none; border-radius:8px; font-weight:600;
    font-family:'Inter',sans-serif; padding:0.5rem 2rem; font-size:0.95rem; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#1e40af; }
div[data-testid="stTextArea"] textarea {
    border-radius:8px; font-family:'Inter',sans-serif;
    font-size:0.93rem; border:1.5px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        base = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base, "model.pkl")
        vectorizer_path = os.path.join(base, "vectorizer.pkl")

        if not os.path.exists(model_path):
            st.error(f"Missing: {model_path}")
            return None, None

        if not os.path.exists(vectorizer_path):
            st.error(f"Missing: {vectorizer_path}")
            return None, None

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def load_metrics():
    try:
        if os.path.exists("metrics.json"):
            with open("metrics.json") as f:
                return json.load(f)
    except Exception as e:
        print(f"Metrics error: {e}")

    return {}

print("LOADING MODEL...")
model, vectorizer = load_artifacts()
print("MODEL LOADED")

st.sidebar.success("App Started")
st.sidebar.write("Model Loaded:", model is not None)
st.sidebar.write("Vectorizer Loaded:", vectorizer is not None)

metrics           = load_metrics()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def strip_byline(text: str) -> str:
    text = re.sub(r'^[A-Z\s,]+\([^)]+\)\s*[-–]\s*', '', text.strip())
    text = re.sub(r'^\([^)]+\)\s*[-–]\s*', '', text.strip())
    return text

def clean_text(text: str) -> str:
    text = strip_byline(str(text))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\b(reuters|associated press|afp|pti|ani|ians)\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str):
    vec   = vectorizer.transform([clean_text(text)])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return ("Real" if label == 1 else "Fake"), float(proba[label]), proba

# ── LIME ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def get_lime(text: str):
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["Fake", "Real"])
        def predict_fn(texts):
            vecs = vectorizer.transform([clean_text(t) for t in texts])
            return model.predict_proba(vecs)
        exp = explainer.explain_instance(
            text, predict_fn, num_features=20, num_samples=500
        )
        return dict(exp.as_list())
    except Exception:
        return {}

def render_lime_html(text: str, weights: dict) -> str:
    if not weights:
        return text
    max_w = max(abs(v) for v in weights.values()) or 1
    words = text.split()
    parts = []
    for w in words:
        key    = w.lower().strip(string.punctuation)
        weight = weights.get(key, 0)
        norm   = weight / max_w
        if   norm >  0.4: cls = "lime-real-strong"
        elif norm >  0.1: cls = "lime-real-mild"
        elif norm < -0.4: cls = "lime-fake-strong"
        elif norm < -0.1: cls = "lime-fake-mild"
        else:             cls = "lime-neutral"
        tip = f'title="{key}: {weight:+.4f}"'
        parts.append(f'<span class="lime-word {cls}" {tip}>{w}</span>')
    return ' '.join(parts)

# ── URL scrape ────────────────────────────────────────────────────────────────
def scrape_url(url: str) -> str:
    return ""

# ── NewsAPI ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=600)
def fetch_live_news(query: str, language: str = "en", page_size: int = 12):
    params = {
        "q":        query,
        "language": language,
        "pageSize": page_size,
        "sortBy":   "publishedAt",
        "apiKey":   NEWS_API_KEY,
    }
    try:
        r    = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        data = r.json()
        return data.get("articles", []) if data.get("status") == "ok" else []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=600)
def fetch_top_headlines(country: str = "in", page_size: int = 12):
    params = {
        "country":  country,
        "pageSize": page_size,
        "apiKey":   NEWS_API_KEY,
    }
    try:
        r    = requests.get("https://newsapi.org/v2/top-headlines", params=params, timeout=10)
        data = r.json()
        return data.get("articles", []) if data.get("status") == "ok" else []
    except Exception:
        return []

def get_source_meta(source_id: str, source_name: str):
    if source_id and source_id in SOURCE_META:
        return SOURCE_META[source_id]
    indian_kw = ['hindu', 'ndtv', 'india', 'times of india', 'hindustan',
                 'express', 'economic times', 'mint', 'deccan', 'tribune', 'wire']
    if any(k in (source_name or "").lower() for k in indian_kw):
        return ("🇮🇳", source_name)
    return ("📰", source_name or "Unknown")

def analyse_articles(articles: list) -> list:
    results = []
    for art in articles:
        title   = art.get("title")   or ""
        desc    = art.get("description") or ""
        content = art.get("content") or ""
        text    = f"{title} {desc} {content}".strip()
        if not text or title == "[Removed]":
            continue
        sid  = (art.get("source") or {}).get("id",   "") or ""
        snam = (art.get("source") or {}).get("name", "") or ""
        flag, display_name = get_source_meta(sid, snam)
        verdict, confidence, proba = predict(text)
        results.append({
            "title":      title,
            "url":        art.get("url", "#"),
            "source":     display_name,
            "flag":       flag,
            "published":  (art.get("publishedAt") or "")[:10],
            "verdict":    verdict,
            "confidence": confidence,
            "fake_prob":  proba[0],
            "real_prob":  proba[1],
        })
    return results

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔍 Fake News Detector")
    st.markdown("---")

    if model is None:
        st.warning("Debug mode: model loading disabled.")
    else:
        st.markdown("**Model**")
        st.markdown(
            f"<div style='font-size:0.85rem;color:#334155;background:#f1f5f9;"
            f"padding:6px 10px;border-radius:6px;font-family:monospace'>"
            f"{metrics.get('best_model', '—')}</div>",
            unsafe_allow_html=True
        )
        st.markdown(" ")
        st.markdown("**Features**")
        st.markdown(
            f"<div style='font-size:0.85rem;color:#334155;background:#f1f5f9;"
            f"padding:6px 10px;border-radius:6px;font-family:monospace'>"
            f"TF-IDF · Unigrams+Bigrams · {metrics.get('vocab_size', 0):,} features</div>",
            unsafe_allow_html=True
        )

        if metrics:
            st.markdown("---")
            st.markdown("**Performance on Test Set**")
            cols = st.columns(2)
            for i, (label, key) in enumerate([
                ("Accuracy",  "accuracy"),
                ("F1 Score",  "f1_score"),
                ("Precision", "precision"),
                ("Recall",    "recall"),
            ]):
                with cols[i % 2]:
                    val   = metrics.get(key, 0) * 100
                    color = "#16a34a" if val >= 98 else "#d97706"
                    st.markdown(
                        f"<div class='sidebar-metric'>"
                        f"<div class='sidebar-metric-label'>{label}</div>"
                        f"<div class='sidebar-metric-val' style='color:{color}'>{val:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            cm = metrics.get("confusion_matrix")
            if cm:
                st.markdown("---")
                st.markdown("**Confusion Matrix**")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Fake", "Actual Real"],
                    columns=["Pred Fake", "Pred Real"]
                )
                st.dataframe(cm_df, use_container_width=True)

            all_models = metrics.get("all_models", {})
            if len(all_models) > 1:
                st.markdown("---")
                st.markdown("**Model Comparison**")
                best = metrics.get("best_model", "")
                rows = [{
                    "Model":    ("🏆 " if n == best else "") + n,
                    "Accuracy": f"{m['accuracy']*100:.2f}%",
                    "F1":       f"{m['f1_score']*100:.2f}%",
                } for n, m in all_models.items()]
                st.dataframe(
                    pd.DataFrame(rows).set_index("Model"),
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown(
            "<small style='color:#94a3b8'>Dataset: Kaggle Fake & Real News<br>"
            "44,898 articles · 80/20 split</small>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-badge">NLP · Machine Learning · TF-IDF + Linear SVM</div>
    <h1>🔍 Fake News Detector</h1>
    <p>Paste an article, upload a file, enter a URL — or check live news from India &amp; around the world.<br>
    Word-level explanation shows exactly what the model found.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("**Model not found.**\n\n```bash\npython train_model.py\n```")
    st.stop()

tab_text, tab_file, tab_url, tab_live = st.tabs([
    "✍️  Paste text", "📄  Upload file", "🔗  From URL", "📡  Live News"
])

input_text = ""

with tab_text:
    st.markdown(" ")
    t = st.text_area(
        "Article",
        height=200,
        placeholder="Paste the full article text here for best results.\nHeadlines work too but full articles give higher confidence.",
        label_visibility="collapsed",
    )
    if t.strip():
        input_text = t.strip()
    st.markdown(
        "<small style='color:#94a3b8'>💡 Tip: longer articles give more accurate results.</small>",
        unsafe_allow_html=True
    )

with tab_file:
    st.markdown(" ")
    uploaded = st.file_uploader("Upload .txt", type=["txt"], label_visibility="collapsed")
    if uploaded:
        input_text = uploaded.read().decode("utf-8", errors="ignore").strip()
        st.success(f"File loaded — {len(input_text):,} characters")
        with st.expander("Preview"):
            st.text(input_text[:1000] + ("…" if len(input_text) > 1000 else ""))

with tab_url:
    st.markdown(" ")
    url = st.text_input(
        "URL",
        placeholder="https://www.thehindu.com/...",
        label_visibility="collapsed"
    )
    if st.button("Fetch article →", disabled=not bool(url)):
        with st.spinner("Fetching…"):
            fetched = scrape_url(url)
        if fetched:
            input_text = fetched
            st.success(f"Fetched {len(fetched):,} characters.")
            with st.expander("Preview"):
                st.text(fetched[:1000] + ("…" if len(fetched) > 1000 else ""))
        else:
            st.error("Could not fetch — try pasting the text directly.")

# ── Live News Tab ─────────────────────────────────────────────────────────────
with tab_live:
    st.markdown(" ")
    st.markdown(
        "<div style='background:#f0f4ff;border:1.5px solid #c7d7fd;border-radius:10px;"
        "padding:1rem 1.2rem;margin-bottom:1.2rem'>"
        "<b style='color:#1d4ed8'>📡 Live News Analysis</b><br>"
        "<span style='font-size:0.88rem;color:#334155'>"
        "Fetches real headlines from NewsAPI and runs them through the detector instantly. "
        "Always verify with the original source.</span></div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        live_mode = st.radio(
            "Mode",
            ["🇮🇳 Top headlines — India", "🌐 Search any topic", "📰 Top headlines — Global"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with c2:
        live_count = st.selectbox("Count", [6, 10, 15, 20], index=1, label_visibility="collapsed")

    live_query = ""
    lang_code  = "en"
    if live_mode == "🌐 Search any topic":
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            live_query = st.text_input(
                "Topic",
                placeholder="e.g.  Modi  /  election  /  economy  /  climate",
                label_visibility="collapsed",
            )
        with sc2:
            lang_code = "en" if st.selectbox(
                "Lang", ["English", "Hindi"], label_visibility="collapsed"
            ) == "English" else "hi"

    if st.button("🔍 Fetch & Analyse", type="primary", key="live_btn"):
        with st.spinner("Fetching live news and analysing…"):
            if live_mode == "🇮🇳 Top headlines — India":
                articles = fetch_top_headlines(country="in", page_size=live_count)
            elif live_mode == "📰 Top headlines — Global":
                articles = fetch_top_headlines(country="us", page_size=live_count)
            else:
                if not live_query.strip():
                    st.warning("Please enter a search topic.")
                    articles = []
                else:
                    articles = fetch_live_news(live_query.strip(), language=lang_code, page_size=live_count)

        if not articles:
            st.error("No articles returned. Check your API key or internet connection.")
        else:
            results = analyse_articles(articles)
            if not results:
                st.warning("Articles fetched but could not be analysed.")
            else:
                real_c = sum(1 for r in results if r["verdict"] == "Real")
                fake_c = sum(1 for r in results if r["verdict"] == "Fake")
                avg_c  = sum(r["confidence"] for r in results) / len(results) * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Analysed",        len(results))
                m2.metric("✅ Likely Real",   real_c)
                m3.metric("❌ Likely Fake",   fake_c)
                m4.metric("Avg confidence",  f"{avg_c:.0f}%")

                st.markdown("---")

                fc1, fc2 = st.columns(2)
                with fc1:
                    show_filter = st.selectbox(
                        "Filter",
                        ["All articles", "Likely Real only", "Likely Fake only"],
                        label_visibility="collapsed",
                    )
                with fc2:
                    sort_by = st.selectbox(
                        "Sort",
                        ["Confidence (high → low)", "Date (newest first)", "Source"],
                        label_visibility="collapsed",
                    )

                filtered = results
                if show_filter == "Likely Real only":
                    filtered = [r for r in results if r["verdict"] == "Real"]
                elif show_filter == "Likely Fake only":
                    filtered = [r for r in results if r["verdict"] == "Fake"]

                if sort_by == "Confidence (high → low)":
                    filtered = sorted(filtered, key=lambda x: x["confidence"], reverse=True)
                elif sort_by == "Date (newest first)":
                    filtered = sorted(filtered, key=lambda x: x["published"], reverse=True)
                else:
                    filtered = sorted(filtered, key=lambda x: x["source"])

                for r in filtered:
                    is_real  = r["verdict"] == "Real"
                    conf_pct = r["confidence"] * 100
                    uncertain = conf_pct < 70

                    if uncertain:
                        card_cls  = "news-card news-card-uncertain"
                        badge_cls = "badge-uncertain"
                        badge_txt = f"⚠️ Uncertain {conf_pct:.0f}%"
                        bar_color = "#d97706"
                    elif is_real:
                        card_cls  = "news-card news-card-real"
                        badge_cls = "badge-real"
                        badge_txt = f"✅ Real {conf_pct:.0f}%"
                        bar_color = "#16a34a"
                    else:
                        card_cls  = "news-card news-card-fake"
                        badge_cls = "badge-fake"
                        badge_txt = f"❌ Fake {conf_pct:.0f}%"
                        bar_color = "#dc2626"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div class="news-headline">
                            {r['flag']} <a href="{r['url']}" target="_blank"
                               style="color:#0f1923;text-decoration:none">{r['title']}</a>
                        </div>
                        <div class="news-meta">{r['source']} · {r['published']}</div>
                        <div style="display:flex;align-items:center;gap:12px">
                            <span class="news-badge {badge_cls}">{badge_txt}</span>
                            <div style="flex:1;background:#e2e8f0;border-radius:6px;height:6px;overflow:hidden">
                                <div style="width:{conf_pct:.0f}%;height:6px;border-radius:6px;background:{bar_color}"></div>
                            </div>
                            <span style="font-size:0.75rem;color:#94a3b8;font-family:monospace;white-space:nowrap">
                                Real {r['real_prob']*100:.0f}% · Fake {r['fake_prob']*100:.0f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    "<small style='color:#94a3b8'>⚠️ Analyses <b>writing style</b>, "
                    "not factual accuracy. Always verify with the original source.</small>",
                    unsafe_allow_html=True
                )

                export_df = pd.DataFrame([{
                    "Headline":   r["title"],
                    "Source":     f"{r['flag']} {r['source']}",
                    "Date":       r["published"],
                    "Verdict":    r["verdict"],
                    "Confidence": f"{r['confidence']*100:.1f}%",
                    "Real prob":  f"{r['real_prob']*100:.1f}%",
                    "Fake prob":  f"{r['fake_prob']*100:.1f}%",
                    "URL":        r["url"],
                } for r in filtered])
                st.download_button(
                    "📥 Download results as CSV",
                    data=export_df.to_csv(index=False),
                    file_name=f"live_news_{datetime.date.today()}.csv",
                    mime="text/csv",
                )

# ── Manual analyse controls ───────────────────────────────────────────────────
st.markdown(" ")
col_lime, col_btn = st.columns([3, 1])
with col_lime:
    use_lime = st.checkbox(
        "Show word-level explanation (LIME)",
        value=True,
        help="Highlights which words pushed the model toward Fake (red) or Real (green)."
    )
with col_btn:
    analyse = st.button(
        "Analyse →",
        type="primary",
        disabled=not bool(input_text),
        use_container_width=True,
    )

# ── Result ────────────────────────────────────────────────────────────────────
if analyse:
    with st.spinner("Analysing…"):
        result, confidence, proba = predict(input_text)
        word_weights = get_lime(input_text) if use_lime else {}

    is_real  = result == "Real"
    card_cls = "result-real" if is_real else "result-fake"
    icon     = "✅" if is_real else "❌"
    bar_cls  = "conf-bar-fill-real" if is_real else "conf-bar-fill-fake"
    pct_cls  = "conf-pct-real" if is_real else "conf-pct-fake"

    st.markdown("---")
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="result-label">{icon} Likely {result} News</div>
        <div class="result-sub">Based on writing style, vocabulary patterns, and linguistic features</div>
        <div class="conf-wrap">
            <div class="conf-label">Confidence</div>
            <div class="{pct_cls} conf-pct">{confidence*100:.1f}%</div>
            <div class="conf-bar-bg">
                <div class="{bar_cls}" style="width:{confidence*100:.1f}%"></div>
            </div>
        </div>
        <div class="prob-row">
            <div class="prob-pill">
                <div class="prob-pill-label">Fake probability</div>
                <div class="prob-pill-val prob-fake-val">{proba[0]*100:.1f}%</div>
            </div>
            <div class="prob-pill">
                <div class="prob-pill-label">Real probability</div>
                <div class="prob-pill-val prob-real-val">{proba[1]*100:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if use_lime and word_weights:
        preview   = input_text[:700] + ("…" if len(input_text) > 700 else "")
        lime_html = render_lime_html(preview, word_weights)
        st.markdown(f"""
        <div class="lime-section">
            <div class="lime-title">Word-level Explanation (LIME)</div>
            <div class="lime-legend">
                <span style="background:#bbf7d0;color:#14532d;padding:1px 6px;border-radius:3px;
                      font-weight:600;font-size:0.78rem">GREEN</span> supports <b>Real</b>
                &nbsp;·&nbsp;
                <span style="background:#fecaca;color:#7f1d1d;padding:1px 6px;border-radius:3px;
                      font-weight:600;font-size:0.78rem">RED</span> supports <b>Fake</b>
                &nbsp;·&nbsp; Hover a word to see its weight
            </div>
            <div class="lime-text">{lime_html}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 Top contributing words"):
            top    = sorted(word_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
            top_df = pd.DataFrame(top, columns=["Word", "Weight"])
            top_df["Direction"] = top_df["Weight"].apply(lambda w: "→ Real" if w > 0 else "→ Fake")
            top_df["Weight"]    = top_df["Weight"].round(4)
            st.dataframe(top_df, use_container_width=True, hide_index=True)
    elif use_lime:
        st.info("LIME explanation unavailable for this input.")

    if len(input_text.split()) < 30:
        st.markdown(
            "<small style='color:#94a3b8'>ℹ️ Short text — paste a full article for "
            "a more reliable result.</small>",
            unsafe_allow_html=True
        )

    st.session_state.history.append({
        "Snippet":    input_text[:65] + ("…" if len(input_text) > 65 else ""),
        "Verdict":    result,
        "Confidence": f"{confidence*100:.1f}%",
        "Fake prob":  f"{proba[0]*100:.1f}%",
        "Real prob":  f"{proba[1]*100:.1f}%",
        "Words":      str(len(input_text.split())),
    })

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("""
    <div class="hist-section">
        <div class="hist-title">Session History</div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(st.session_state.history[::-1]),
        use_container_width=True,
        hide_index=True
    )
    if st.button("Clear history", type="secondary"):
        st.session_state.history = []
        st.rerun()