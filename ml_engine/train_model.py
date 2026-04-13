"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        NewsGuard — Fake News Detection ML Training Pipeline                 ║
║                                                                              ║
║  Supports:                                                                   ║
║    • Kaggle Fake News Dataset  (True.csv / Fake.csv)                        ║
║    • ISOT Fake News Dataset    (True.csv / Fake.csv)                        ║
║    • WELFake Dataset           (WELFake_Dataset.csv)                        ║
║    • Existing fake_or_real_news.csv                                         ║
║                                                                              ║
║  Models Trained:                                                             ║
║    1. Logistic Regression  ⭐ (Best for hackathon — fast + accurate)        ║
║    2. Naive Bayes                                                            ║
║    3. Random Forest                                                          ║
║    4. Ensemble (Voting Classifier of all three)                              ║
║                                                                              ║
║  Pipeline:                                                                   ║
║    Raw Text → Clean → TF-IDF → ML Model → REAL / FAKE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import re
import warnings
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import MaxAbsScaler

warnings.filterwarnings('ignore')

# ─── Paths ─────────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT   = os.path.join(DIR, 'ng_model.pkl')
VECTOR_OUT  = os.path.join(DIR, 'ng_vectorizer.pkl')
META_OUT    = os.path.join(DIR, 'ng_model_meta.json')
CM_OUT      = os.path.join(DIR, 'ng_confusion_matrix.png')

# ─── NLTK (optional, graceful fallback) ────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOP_WORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    STOP_WORDS = set()
    HAS_NLTK = False
    print("  ⚠  NLTK not installed — using basic cleaning (pip install nltk)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATASET LOADER
# Supports multiple dataset formats automatically
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_dir: str = DIR) -> pd.DataFrame:
    """
    Auto-detects and loads the best available dataset from data_dir.

    Priority:
      1. True.csv + Fake.csv   (Kaggle / ISOT format — ~44k articles)
      2. WELFake_Dataset.csv   (~72k articles)
      3. fake_or_real_news.csv (existing small dataset)

    Returns a DataFrame with columns [text, label]
    where label = 1 (REAL) or 0 (FAKE)
    """
    print("\n" + "═"*60)
    print("  STEP 1: LOADING DATASET")
    print("═"*60)

    # ── Option A: Kaggle / ISOT format (True.csv + Fake.csv) ──
    true_path = _find(data_dir, ['True.csv', 'true.csv', 'TRUE.csv'])
    fake_path = _find(data_dir, ['Fake.csv', 'fake.csv', 'FAKE.csv'])

    if true_path and fake_path:
        print(f"  ✅ Found Kaggle/ISOT format:")
        print(f"     True: {true_path}")
        print(f"     Fake: {fake_path}")
        df_real = pd.read_csv(true_path)
        df_fake = pd.read_csv(fake_path)

        # Combine title + text if available
        df_real['text'] = _combine_title_text(df_real)
        df_fake['text'] = _combine_title_text(df_fake)

        df_real['label'] = 1
        df_fake['label'] = 0

        df = pd.concat([df_real[['text', 'label']], df_fake[['text', 'label']]], ignore_index=True)
        print(f"  📊 REAL: {len(df_real):,} articles | FAKE: {len(df_fake):,} articles")
        return df.dropna(subset=['text'])

    # ── Option B: WELFake ──
    wel_path = _find(data_dir, ['WELFake_Dataset.csv', 'welfake.csv', 'WELFake.csv'])
    if wel_path:
        print(f"  ✅ Found WELFake dataset: {wel_path}")
        df = pd.read_csv(wel_path)
        # WELFake: label 0=fake, 1=real
        df['text'] = _combine_title_text(df)
        df = df[['text', 'label']].dropna()
        df['label'] = df['label'].astype(int)
        real = df['label'].sum()
        fake = len(df) - real
        print(f"  📊 REAL: {real:,} | FAKE: {fake:,}")
        return df

    # ── Option C: fake_or_real_news.csv (GossipCop-style) ──
    fn_path = _find(data_dir, ['fake_or_real_news.csv', 'news.csv'])
    if fn_path:
        print(f"  ✅ Found existing dataset: {fn_path}")
        df = pd.read_csv(fn_path)
        print(f"  📋 Columns: {list(df.columns)}")

        # Detect text column
        text_col = _detect_col(df, ['text', 'body', 'content', 'article'])
        label_col = _detect_col(df, ['label', 'class', 'target', 'category'])

        if text_col and label_col:
            df = df[[text_col, label_col]].copy()
            df.columns = ['text', 'label_raw']
            # Normalise label
            lmap = {}
            unique_labels = df['label_raw'].str.upper().unique() if df['label_raw'].dtype == object else df['label_raw'].unique()
            for l in unique_labels:
                ls = str(l).upper().strip()
                if ls in ['REAL', 'TRUE', '1', 'GENUINE', 'LEGIT']:
                    lmap[l] = 1
                else:
                    lmap[l] = 0
            df['label'] = df['label_raw'].map(lmap)
            df = df[['text', 'label']].dropna()
            df['label'] = df['label'].astype(int)
            real = df['label'].sum()
            fake = len(df) - real
            print(f"  📊 REAL: {real:,} | FAKE: {fake:,}")
            return df

    # ── No dataset found ──
    print("\n  ❌ NO DATASET FOUND!")
    print("  Please download one of the following datasets and place files in:")
    print(f"  {data_dir}\n")
    print("  OPTION A (Kaggle Fake News — ~22k per class):")
    print("    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("    → Extract: True.csv + Fake.csv\n")
    print("  OPTION B (ISOT Fake News Dataset — ~21k per class):")
    print("    https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php")
    print("    → Extract: True.csv + Fake.csv\n")
    print("  OPTION C (WELFake — 72k articles):")
    print("    https://zenodo.org/record/4561253")
    print("    → Extract: WELFake_Dataset.csv\n")
    sys.exit(1)

def _find(directory, filenames):
    for root, _, files in os.walk(directory):
        for fn in filenames:
            if fn in files:
                return os.path.join(root, fn)
    return None

def _combine_title_text(df):
    title_col = _detect_col(df, ['title', 'headline', 'heading'])
    text_col  = _detect_col(df, ['text', 'body', 'content', 'article'])
    if title_col and text_col:
        return (df[title_col].fillna('') + ' ' + df[text_col].fillna('')).str.strip()
    elif text_col:
        return df[text_col].fillna('')
    elif title_col:
        return df[title_col].fillna('')
    # Fallback: take the longest string column
    str_cols = df.select_dtypes(include='object').columns
    if len(str_cols):
        longest = max(str_cols, key=lambda c: df[c].fillna('').str.len().mean())
        return df[longest].fillna('')
    return pd.Series([''] * len(df))

def _detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str, stem: bool = False) -> str:
    """
    News article text cleaning pipeline:
      1. Lowercase
      2. Remove URLs, emails, HTML tags
      3. Remove wire service prefixes (REUTERS, AP, etc.)
      4. Remove punctuation & digits
      5. Remove extra whitespace
      6. Optional: stemming (disabled by default for speed)
    """
    if not isinstance(text, str) or len(text) < 3:
        return ''

    text = text.lower()

    # Remove wire service prefixes like "WASHINGTON (Reuters) -"
    text = re.sub(r'^[a-z\s]+\(.*?\)\s*[-–—]\s*', '', text)

    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optional stemming
    if stem and HAS_NLTK:
        tokens = text.split()
        tokens = [STEMMER.stem(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
        text = ' '.join(tokens)

    return text

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("  STEP 2: TEXT CLEANING")
    print("═"*60)
    total = len(df)
    print(f"  Cleaning {total:,} articles...")
    t0 = time.time()
    df = df.copy()
    df['text_clean'] = df['text'].apply(lambda x: clean_text(str(x)))
    df = df[df['text_clean'].str.len() > 30].reset_index(drop=True)
    elapsed = time.time() - t0
    removed = total - len(df)
    print(f"  ✅ Done in {elapsed:.1f}s  |  Removed {removed} empty/short articles")
    print(f"  ✅ Remaining: {len(df):,} articles")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TF-IDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorizer(X_train, max_features: int = 50000) -> TfidfVectorizer:
    """
    TF-IDF parameters tuned for fake news detection:
     - Unigrams + Bigrams (captures phrases like "fake news", "unnamed source")
     - max_df=0.85: ignore terms appearing in >85% of docs (too common)
     - min_df=2:    ignore terms appearing in <2 docs (noise)
     - sublinear_tf: log normalization for long articles
    """
    print("\n" + "═"*60)
    print("  STEP 3: TF-IDF VECTORIZATION")
    print("═"*60)
    print(f"  max_features={max_features:,} | ngram_range=(1,2) | sublinear_tf=True")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),          # unigrams + bigrams
        max_df=0.85,                 # ignore words in >85% of docs
        min_df=2,                    # ignore words in <2 docs
        sublinear_tf=True,           # apply log normalization
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b[a-zA-Z]{2,}\b',
    )
    vectorizer.fit(X_train)
    print(f"  ✅ Vocabulary size: {len(vectorizer.vocabulary_):,} terms")
    return vectorizer

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train):
    """Logistic Regression — Best for hackathon: fast, interpretable, ~98% accuracy"""
    print("\n  📌 Training Logistic Regression...")
    t0 = time.time()
    model = LogisticRegression(
        C=5.0,             # regularization strength (higher = less regularization)
        max_iter=1000,
        solver='saga',     # good for large datasets
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"     ⏱ {time.time()-t0:.1f}s")
    return model

def train_naive_bayes(X_train, y_train):
    """Multinomial Naive Bayes — extremely fast, works great on TF-IDF"""
    print("\n  📌 Training Multinomial Naive Bayes...")
    t0 = time.time()
    # NB requires non-negative features — MaxAbsScaler preserves sparsity
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    print(f"     ⏱ {time.time()-t0:.1f}s")
    return model

def train_random_forest(X_train, y_train):
    """Random Forest — slower but captures non-linear patterns"""
    print("\n  📌 Training Random Forest...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"     ⏱ {time.time()-t0:.1f}s")
    return model

def evaluate(model, X_test, y_test, name: str):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average='macro')
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
    except Exception:
        auc = 0.0
    print(f"\n  ┌─ {name} ─{'─'*(40-len(name))}")
    print(f"  │  Accuracy : {acc*100:.2f}%")
    print(f"  │  F1 Score : {f1*100:.2f}%")
    print(f"  │  AUC-ROC  : {auc:.4f}")
    print(f"  └{'─'*44}")
    return acc, f1, auc, preds

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE & VISUALISE
# ══════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(y_true, y_pred, model_name: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'],
                linewidths=0.5)
    ax.set_title(f'NewsGuard — {model_name}\nConfusion Matrix', fontsize=13, pad=12)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  📊 Confusion matrix saved → {path}")

def save_meta(model_name, acc, f1, auc, vocab_size, n_train, n_test, path):
    import json
    meta = {
        'model_name':  model_name,
        'accuracy':    round(acc, 4),
        'f1_score':    round(f1, 4),
        'auc_roc':     round(auc, 4),
        'vocab_size':  vocab_size,
        'n_train':     n_train,
        'n_test':      n_test,
        'trained_at':  time.strftime('%Y-%m-%dT%H:%M:%S'),
        'version':     '2.0'
    }
    with open(path, 'w') as fp:
        json.dump(meta, fp, indent=2)
    print(f"  💾 Model metadata → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  NewsGuard ML Training Pipeline v2.0")
    print("="*60)

    # -- STEP 1: Load --
    df = load_dataset(DIR)

    # -- STEP 2: Clean --
    df = preprocess_dataset(df)

    # -- SPLIT --
    X = df['text_clean']
    y = df['label']

    print(f"\n  Splitting: 80% train / 20% test (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -- STEP 3: Vectorize --
    MAX_FEAT = min(50000, max(5000, len(X_train) // 2))
    vectorizer = build_vectorizer(X_train, max_features=MAX_FEAT)

    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # -- STEP 4: Train models --
    print("\n" + "="*60)
    print("  STEP 4: TRAINING CLASSIFIERS")
    print("="*60)

    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    nb_model = train_naive_bayes(X_train_tfidf, y_train)

    # Skip Random Forest for very small datasets (slow and may overfit)
    train_rf = len(X_train) > 2000
    rf_model = train_random_forest(X_train_tfidf, y_train) if train_rf else None

    # -- STEP 5: Evaluate --
    print("\n" + "="*60)
    print("  STEP 5: EVALUATION RESULTS")
    print("="*60)

    results = {}
    lr_acc, lr_f1, lr_auc, lr_pred = evaluate(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    results['Logistic Regression'] = (lr_acc, lr_f1, lr_auc, lr_model, lr_pred)

    nb_acc, nb_f1, nb_auc, nb_pred = evaluate(nb_model, X_test_tfidf, y_test, "Naive Bayes")
    results['Naive Bayes'] = (nb_acc, nb_f1, nb_auc, nb_model, nb_pred)

    if rf_model:
        rf_acc, rf_f1, rf_auc, rf_pred = evaluate(rf_model, X_test_tfidf, y_test, "Random Forest")
        results['Random Forest'] = (rf_acc, rf_f1, rf_auc, rf_model, rf_pred)

    # -- Select best model --
    best_name = max(results, key=lambda k: results[k][0])  # by accuracy
    best_acc, best_f1, best_auc, best_model, best_pred = results[best_name]

    print(f"\n  >>> BEST MODEL: {best_name}")
    print(f"     Accuracy : {best_acc*100:.2f}%")
    print(f"     F1 Score : {best_f1*100:.2f}%")
    print(f"     AUC-ROC  : {best_auc:.4f}")

    print("\n" + classification_report(y_test, best_pred,
                                       target_names=['FAKE', 'REAL'],
                                       digits=4))

    # -- STEP 6: Save --
    print("\n" + "="*60)
    print("  STEP 6: SAVING ARTIFACTS")
    print("="*60)

    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(vectorizer, VECTOR_OUT)
    print(f"  [saved] Model      -> {MODEL_OUT}")
    print(f"  [saved] Vectorizer -> {VECTOR_OUT}")

    save_confusion_matrix(y_test, best_pred, best_name, CM_OUT)
    save_meta(best_name, best_acc, best_f1, best_auc,
              len(vectorizer.vocabulary_), len(X_train), len(X_test), META_OUT)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print(f"  ML Server will load: ng_model.pkl + ng_vectorizer.pkl")
    print("  Run:  python ml_engine/ml_server.py")
    print("="*60)
    print()


if __name__ == '__main__':
    main()

