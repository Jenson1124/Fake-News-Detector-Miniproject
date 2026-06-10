"""
train_model.py — Fake News Detector | Training Pipeline
========================================================
Trains and compares Logistic Regression, Naive Bayes, and Linear SVM.

Supports:
  - US Kaggle dataset (Fake.csv + True.csv)
  - Indian fake news dataset (combined automatically when present)
  - Combined training for better cross-domain accuracy

Usage:
    python train_model.py              # full dataset
    python train_model.py --sample     # quick pipeline test
    python train_model.py --india-only # Indian dataset only
"""

import argparse, json, os, re, string, warnings
warnings.filterwarnings("ignore")

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words('english')) | {
    'reuters', 'ap', 'afp', 'said', 'would', 'also', 'one', 'two', 'three',
    'new', 'year', 'say', 'told', 'added', 'according', 'pti', 'ani', 'ians',
    'ubi', 'uniindia'  # Indian news agencies — same fix as Reuters
}

# ── 1. Text cleaning ──────────────────────────────────────────────────────────

def strip_byline(text: str) -> str:
    """
    Remove news agency bylines like:
      'WASHINGTON (Reuters) - ...'   → US
      'NEW DELHI (PTI) - ...'        → Indian
      '(ANI) - ...'                  → Indian
    """
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

# ── 2. Load US dataset ────────────────────────────────────────────────────────

def load_us_data(true_path: str, fake_path: str) -> pd.DataFrame:
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    true_df['label']  = 1
    fake_df['label']  = 0
    true_df['source'] = 'us'
    fake_df['source'] = 'us'
    df = pd.concat([true_df, fake_df], ignore_index=True)
    title = df.get('title', pd.Series(['']*len(df))).fillna('')
    body  = df.get('text',  pd.Series(['']*len(df))).fillna('')
    df['combined_text'] = (title + ' ' + body).str.strip()
    print(f"   US dataset  : {len(df):,} articles  (Real: {df['label'].sum():,} | Fake: {(df['label']==0).sum():,})")
    return df[['combined_text', 'label', 'source']]

# ── 3. Load Indian dataset ────────────────────────────────────────────────────

def load_indian_data(data_dir: str = 'dataset/indian') -> pd.DataFrame | None:
    """
    Tries to load the Indian fake news dataset from dataset/indian/
    Supports multiple common formats from Kaggle.
    Returns None if not found.
    """
    if not os.path.isdir(data_dir):
        return None

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        return None

    dfs = []
    for fname in csv_files:
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            # Detect label column
            if 'label' in df.columns:
                # Normalise label: FAKE/0 → 0, REAL/1 → 1
                df['label'] = df['label'].apply(
                    lambda x: 0 if str(x).strip().upper() in ('FAKE', '0', 'FALSE') else 1
                )
            elif 'fake' in df.columns:
                df['label'] = df['fake'].apply(lambda x: 0 if int(x) == 1 else 1)
            else:
                print(f"   ⚠️  {fname}: no recognised label column — skipping")
                continue

            # Build combined text
            title = df.get('title',   pd.Series(['']*len(df))).fillna('')
            body  = df.get('text',    pd.Series(['']*len(df))).fillna('')
            body  = body if body.str.len().sum() > 0 else df.get('article', pd.Series(['']*len(df))).fillna('')
            body  = body if body.str.len().sum() > 0 else df.get('content', pd.Series(['']*len(df))).fillna('')
            df['combined_text'] = (title + ' ' + body).str.strip()
            df['source'] = 'india'
            dfs.append(df[['combined_text', 'label', 'source']])
            print(f"   Loaded {fname}: {len(df):,} rows")
        except Exception as e:
            print(f"   ⚠️  Could not load {fname}: {e}")

    if not dfs:
        return None

    india_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"   Indian dataset: {len(india_df):,} articles  (Real: {india_df['label'].sum():,} | Fake: {(india_df['label']==0).sum():,})")
    return india_df

# ── 4. Feature engineering ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    print("\n🔧 Cleaning & vectorising …")
    df['cleaned'] = df['combined_text'].apply(clean_text)

    reuters_count = df['cleaned'].str.contains('reuters').sum()
    print(f"   'reuters' remaining after cleaning: {reuters_count} (should be ~0)")

    X, y = df['cleaned'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=60_000,    # larger vocab for cross-domain coverage
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print(f"   Train   : {X_train_vec.shape[0]:,}")
    print(f"   Test    : {X_test_vec.shape[0]:,}")
    print(f"   Vocab   : {X_train_vec.shape[1]:,}")
    return vectorizer, X_train_vec, X_test_vec, y_train, y_test

# ── 5. Train & compare ────────────────────────────────────────────────────────

MODELS = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0,
        solver='lbfgs', random_state=42
    ),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Linear SVM': CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
    ),
}

def train_and_compare(X_train_vec, X_test_vec, y_train, y_test):
    print(f"\n🚀 Training & comparing models …\n")
    print(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "─" * 65)

    results = {}
    best_f1, best_name, best_model = 0, None, None

    for name, clf in MODELS.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred).tolist()

        marker = ' ✓' if f1 > best_f1 else ''
        print(f"  {name:<25} {acc*100:>8.2f}%  {prec*100:>9.2f}%"
              f"  {rec*100:>7.2f}%  {f1*100:>7.2f}%{marker}")

        results[name] = {
            'accuracy':         round(acc,  4),
            'precision':        round(prec, 4),
            'recall':           round(rec,  4),
            'f1_score':         round(f1,   4),
            'confusion_matrix': cm,
        }
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, clf

    print(f"\n  🏆  Best model: {best_name}  (F1 = {best_f1*100:.2f}%)")
    return best_name, best_model, results

# ── 6. Save ───────────────────────────────────────────────────────────────────

def save(best_name, best_model, vectorizer, all_results, dataset_info):
    best = all_results[best_name]
    metrics = {
        'best_model':       best_name,
        'accuracy':         best['accuracy'],
        'precision':        best['precision'],
        'recall':           best['recall'],
        'f1_score':         best['f1_score'],
        'confusion_matrix': best['confusion_matrix'],
        'vocab_size':       len(vectorizer.vocabulary_),
        'ngram_range':      '1-2',
        'max_features':     60_000,
        'all_models':       all_results,
        'dataset_info':     dataset_info,
    }
    joblib.dump(best_model,  'model.pkl')
    joblib.dump(vectorizer,  'vectorizer.pkl')
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved: model.pkl | vectorizer.pkl | metrics.json")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample',     action='store_true')
    parser.add_argument('--india-only', action='store_true')
    parser.add_argument('--true',       default='dataset/True.csv')
    parser.add_argument('--fake',       default='dataset/Fake.csv')
    parser.add_argument('--india-dir',  default='dataset/indian')
    args = parser.parse_args()

    if args.sample:
        print('⚠️  Sample mode — tiny dataset for pipeline testing only.')

    all_dfs = []
    dataset_info = {}

    # Load US data
    if not args.india_only:
        if os.path.exists(args.true) and os.path.exists(args.fake):
            print("\n📊 Loading datasets …")
            us_df = load_us_data(args.true, args.fake)
            all_dfs.append(us_df)
            dataset_info['us_articles'] = len(us_df)
        else:
            print(f'\n❌ US dataset not found at {args.true} / {args.fake}')
            if not os.path.isdir(args.india_dir):
                print('   Run: python train_model.py --sample')
                return

    # Load Indian data
    if os.path.isdir(args.india_dir):
        print("\n📊 Loading Indian dataset …" if args.india_only else "")
        india_df = load_indian_data(args.india_dir)
        if india_df is not None:
            all_dfs.append(india_df)
            dataset_info['india_articles'] = len(india_df)
            print(f"   ✅ Indian dataset loaded — model will now recognise Indian vocabulary")
        else:
            print(f"   ℹ️  No Indian CSVs found in {args.india_dir}")
    else:
        print(f"\n   ℹ️  No Indian dataset found (dataset/indian/ missing)")
        print(f"   ℹ️  See download_indian_dataset.py for instructions")

    if not all_dfs:
        print('\n❌ No data to train on.')
        return

    df = pd.concat(all_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    dataset_info['total_articles'] = len(df)
    has_india = dataset_info.get('india_articles', 0) > 0

    print(f"\n   ─────────────────────────────")
    print(f"   TOTAL   : {len(df):,} articles")
    print(f"   Real    : {df['label'].sum():,}")
    print(f"   Fake    : {(df['label']==0).sum():,}")
    if has_india:
        print(f"   🇮🇳 Indian articles included!")
    print(f"   ─────────────────────────────")

    if args.sample:
        df = df.sample(n=min(2000, len(df)), random_state=42)

    vectorizer, X_train_vec, X_test_vec, y_train, y_test = build_features(df)
    best_name, best_model, all_results = train_and_compare(
        X_train_vec, X_test_vec, y_train, y_test
    )
    save(best_name, best_model, vectorizer, all_results, dataset_info)
    print('\n✅ Done! Run: streamlit run app.py\n')

if __name__ == '__main__':
    main()
