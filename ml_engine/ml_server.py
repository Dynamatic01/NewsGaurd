"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       NewsGuard — Local ML Prediction Server                                ║
║                                                                              ║
║  Endpoint:  POST /api/ml/predict                                             ║
║  Port:      8000                                                             ║
║                                                                              ║
║  ML Pipeline:                                                                ║
║    News Text → Clean → TF-IDF → Logistic Regression → REAL / FAKE          ║
║                                                                              ║
║  Also integrates the existing DL (LSTM) model as a secondary signal.        ║
║  The Node.js server on port 5000 calls this as part of its pipeline.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import re
import json
import time
import pickle
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger('newsguard-ml')

DIR = os.path.dirname(os.path.abspath(__file__))

# ─── FastAPI / Uvicorn ───────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("\n❌ FastAPI / Uvicorn not installed.")
    print("   Run:  pip install fastapi uvicorn")
    sys.exit(1)

# ─── joblib ─────────────────────────────────────────────────────────────────
try:
    import joblib
except ImportError:
    print("\n❌ joblib not installed. Run:  pip install joblib")
    sys.exit(1)

# ─── numpy (required) ───────────────────────────────────────────────────────
import numpy as np

# ─── TensorFlow (optional — for LSTM fusion) ────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tf.get_logger().setLevel('ERROR')
    HAS_TF = True
except ImportError:
    HAS_TF = False
    log.warning("TensorFlow not installed — LSTM fusion disabled (classical model only)")

# ═══════════════════════════════════════════════════════════════════════════
# TEXT CLEANING  (same logic as train_model.py)
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str) or len(text) < 3:
        return ''
    text = text.lower()
    text = re.sub(r'^[a-z\s]+\(.*?\)\s*[-–—]\s*', '', text)   # wire prefix
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)      # URLs
    text = re.sub(r'<[^>]+>', ' ', text)                         # HTML
    text = re.sub(r'[^a-z\s]', ' ', text)                       # non-alpha
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

class ModelBundle:
    """Loads and manages both classical ML and optional DL models."""

    def __init__(self):
        self.classical_model  = None
        self.vectorizer       = None
        self.meta             = {}
        self.dl_model         = None
        self.dl_tokenizer     = None
        self.MAX_SEQ_LEN      = 500

        self._load_classical()
        self._load_dl()

    def _load_classical(self):
        """Load new-style ng_model.pkl first, fall back to classical_model.pkl"""
        # New training output
        new_model = os.path.join(DIR, 'ng_model.pkl')
        new_vec   = os.path.join(DIR, 'ng_vectorizer.pkl')
        # Legacy training output
        old_model = os.path.join(DIR, 'classical_model.pkl')
        old_vec   = os.path.join(DIR, 'tfidf_vectorizer.pkl')
        # Metadata
        meta_path = os.path.join(DIR, 'ng_model_meta.json')

        model_path = new_model if os.path.exists(new_model) else (old_model if os.path.exists(old_model) else None)
        vec_path   = new_vec   if os.path.exists(new_vec)   else (old_vec   if os.path.exists(old_vec)   else None)

        if model_path and vec_path:
            try:
                self.classical_model = joblib.load(model_path)
                self.vectorizer      = joblib.load(vec_path)
                log.info(f"✅ Classical ML model loaded: {os.path.basename(model_path)}")

                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        self.meta = json.load(f)
                    log.info(f"   Accuracy: {self.meta.get('accuracy', 0)*100:.2f}%  |  Model: {self.meta.get('model_name','?')}")
            except Exception as e:
                log.error(f"Failed to load classical models: {e}")
        else:
            log.warning("⚠ Classical ML model not found. Run train_model.py first.")

    def _load_dl(self):
        """Load LSTM DL model (optional — used for ensemble scoring)"""
        if not HAS_TF:
            return
        dl_path  = os.path.join(DIR, 'dl_model.keras')
        tok_path = os.path.join(DIR, 'dl_tokenizer.pkl')
        if os.path.exists(dl_path) and os.path.exists(tok_path):
            try:
                self.dl_model = tf.keras.models.load_model(dl_path)
                with open(tok_path, 'rb') as f:
                    self.dl_tokenizer = pickle.load(f)
                log.info("✅ LSTM DL model loaded (fusion enabled)")
            except Exception as e:
                log.warning(f"LSTM model load failed: {e}")

    def predict(self, text: str) -> dict:
        """
        Predict REAL / FAKE for input text.

        Returns:
          result        : 'REAL' | 'FAKE' | 'SUSPICIOUS'
          trust_score   : 0–100  (higher = more trustworthy)
          confidence    : 0–100  (how confident the model is)
          ml_prob       : raw probability of REAL (0.0–1.0)
          model_used    : name of model
          signals       : list of interpretability signals
        """
        if self.classical_model is None or self.vectorizer is None:
            return {
                'error': 'ML model not loaded. Run train_model.py first.',
                'result': 'UNVERIFIED',
                'trust_score': 50,
                'confidence': 0
            }

        cleaned = clean_text(text)
        if len(cleaned) < 10:
            return {
                'error': 'Text too short for ML analysis.',
                'result': 'UNVERIFIED',
                'trust_score': 50,
                'confidence': 0
            }

        X = self.vectorizer.transform([cleaned])

        # Classical model probability (REAL = class 1)
        try:
            proba = self.classical_model.predict_proba(X)[0]
            p_fake = float(proba[0])
            p_real = float(proba[1])
        except AttributeError:
            # For models without predict_proba (shouldn't happen with LR/NB/RF)
            pred = self.classical_model.predict(X)[0]
            p_real = 1.0 if pred == 1 else 0.0
            p_fake = 1.0 - p_real

        classical_prob = p_real  # probability of being REAL

        # LSTM fusion (if available)
        dl_prob = None
        if self.dl_model and self.dl_tokenizer:
            try:
                seq    = self.dl_tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=self.MAX_SEQ_LEN, padding='post', truncating='post')
                dl_prob = float(self.dl_model.predict(padded, verbose=0)[0][0])
            except Exception:
                dl_prob = None

        # Ensemble: blend classical (70%) + LSTM (30%) if available
        if dl_prob is not None:
            fused_prob = 0.70 * classical_prob + 0.30 * dl_prob
            model_used = f"{self.meta.get('model_name','ML')} + LSTM"
        else:
            fused_prob = classical_prob
            model_used = self.meta.get('model_name', 'Logistic Regression')

        # Classification thresholds
        SUSPICIOUS_LOW  = 0.38
        SUSPICIOUS_HIGH = 0.62

        if fused_prob > SUSPICIOUS_HIGH:
            result = 'REAL'
        elif fused_prob < SUSPICIOUS_LOW:
            result = 'FAKE'
        else:
            result = 'SUSPICIOUS'

        # Trust score: 0–100 map from fused_prob
        trust_score = int(round(fused_prob * 100))

        # Confidence: how far from the decision boundary (0.5)
        confidence = int(round(min(abs(fused_prob - 0.5) * 2, 1.0) * 100))

        # Interpretability signals
        signals = self._build_signals(text, cleaned, fused_prob, classical_prob, dl_prob)

        return {
            'result':        result,
            'trust_score':   trust_score,
            'confidence':    confidence,
            'ml_prob':       round(fused_prob, 4),
            'classical_prob': round(classical_prob, 4),
            'dl_prob':       round(dl_prob, 4) if dl_prob is not None else None,
            'model_used':    model_used,
            'model_accuracy': round(self.meta.get('accuracy', 0) * 100, 1),
            'signals':       signals,
        }

    def _build_signals(self, raw_text: str, cleaned: str, fused_prob: float,
                       classical_prob: float, dl_prob) -> list:
        signals = []

        # 1. Primary score
        signals.append({
            'name': 'ML Trust Probability',
            'value': f"{fused_prob*100:.1f}% REAL",
            'status': 'good' if fused_prob > 0.62 else ('bad' if fused_prob < 0.38 else 'neutral'),
            'icon': '🤖'
        })

        # 2. Clickbait / sensational language detection
        sensational_words = [
            'shocking', 'bombshell', 'breaking', 'exposed', 'banned', 'horrifying',
            'unbelievable', 'miracle', 'secret', 'hidden', 'they don\'t want you',
            'mainstream media', 'deep state', 'hoax', 'conspiracy', 'leaked'
        ]
        rt = raw_text.lower()
        hits = [w for w in sensational_words if w in rt]
        if hits:
            signals.append({
                'name': 'Sensational Language',
                'value': f"{len(hits)} trigger words found",
                'status': 'bad' if len(hits) > 3 else 'neutral',
                'icon': '🎭'
            })
        else:
            signals.append({
                'name': 'Language Pattern',
                'value': 'No sensational trigger words',
                'status': 'good',
                'icon': '✅'
            })

        # 3. Text length signal
        word_count = len(raw_text.split())
        if word_count < 50:
            signals.append({
                'name': 'Article Length',
                'value': f"{word_count} words — very short',",
                'status': 'neutral',
                'icon': '📏'
            })
        elif word_count > 300:
            signals.append({
                'name': 'Article Length',
                'value': f"{word_count} words — detailed reporting",
                'status': 'good',
                'icon': '📰'
            })

        # 4. Source citation pattern
        has_quote    = '"' in raw_text or '"' in raw_text or '"' in raw_text
        has_numbers  = bool(re.search(r'\d+\.?\d*\s*(%|percent|million|billion)', raw_text, re.I))
        if has_quote or has_numbers:
            signals.append({
                'name': 'Citations & Data',
                'value': 'Quotes and/or statistics present',
                'status': 'good',
                'icon': '📋'
            })

        # 5. LSTM agreement signal
        if dl_prob is not None:
            agreement = abs(classical_prob - dl_prob) < 0.2
            signals.append({
                'name': 'Model Agreement (ML + LSTM)',
                'value': f"Classical: {classical_prob*100:.0f}% | LSTM: {dl_prob*100:.0f}%",
                'status': 'good' if agreement else 'neutral',
                'icon': '🧠'
            })

        return signals


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NewsGuard ML Server",
    description="Local fake news detection ML API — No external API required",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Similarity Engine ──────────────────────────────────────────────────
try:
    from similarity_engine import get_engine as _get_sim_engine
    sim_engine = _get_sim_engine()
    log.info(f"✅ Similarity engine ready — {sim_engine.get_kb_stats()['total_articles']} articles in KB")
except Exception as _sim_err:
    sim_engine = None
    log.warning(f"⚠ Similarity engine unavailable: {_sim_err}")

# Load models at startup
bundle = ModelBundle()


class AnalyzeRequest(BaseModel):
    text:  str = ""
    url:   str = ""
    image: str = ""


@app.get("/")
def root():
    return {
        "service": "NewsGuard ML Server",
        "version": "2.0.0",
        "status":  "running",
        "model_loaded": bundle.classical_model is not None,
        "dl_loaded":    bundle.dl_model is not None,
        "model_info":   bundle.meta
    }


@app.get("/api/ml/health")
def health():
    return {
        "status":         "ok",
        "classical_model": bundle.classical_model is not None,
        "dl_model":        bundle.dl_model is not None,
        "model_accuracy":  bundle.meta.get("accuracy", 0),
        "model_name":      bundle.meta.get("model_name", "unknown"),
        "vocab_size":      bundle.meta.get("vocab_size", 0)
    }


@app.post("/api/ml/predict")
async def predict(req: AnalyzeRequest):
    """
    Predict whether news text is REAL or FAKE using local ML model.
    No external API calls — fully offline.
    """
    text = (req.text or "").strip()

    if not text or len(text) < 15:
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 15 characters."
        )

    t0 = time.time()
    result = bundle.predict(text)
    result['latency_ms'] = round((time.time() - t0) * 1000, 1)
    result['timestamp']  = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    return result


@app.post("/api/ml/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Legacy endpoint — compatible with existing frontend / Node.js calls.
    Wraps predict() and formats output to match the previous API schema.
    """
    text = (req.text or "").strip()

    if not text or len(text) < 15:
        return {"error": "Text too short for analysis.", "trust_score": 50}

    result = bundle.predict(text)

    if 'error' in result and result.get('trust_score') == 50:
        return result

    # Map to legacy schema expected by renderResults()
    verdict_map = {'REAL': 'True', 'FAKE': 'False', 'SUSPICIOUS': 'Suspicious'}
    return {
        "result":           verdict_map.get(result['result'], 'Suspicious'),
        "trust_score":      result['trust_score'],
        "risk_level":       'Low' if result['trust_score'] >= 60 else ('Medium' if result['trust_score'] >= 40 else 'High'),
        "confidence":       result['confidence'],
        "fact_summary":     (
            f"Local ML ({result['model_used']}) predicts this content is "
            f"{result['result']} with {result['confidence']}% confidence. "
            f"Trust probability: {result['ml_prob']*100:.1f}%. "
            f"Model trained accuracy: {result['model_accuracy']}%."
        ),
        "verified_sources": [],
        "ml_signals":       result.get('signals', []),
        "_model_used":      result['model_used'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityRequest(BaseModel):
    text:  str = ""
    title: str = ""


class AddArticleRequest(BaseModel):
    title:    str
    text:     str
    source:   str = "user-submitted"
    date:     str = ""
    category: str = "general"
    label:    str = "real"


@app.post("/api/similarity/check")
async def similarity_check(req: SimilarityRequest):
    """
    Check if similar news exists in the verified knowledge base.
    Uses TF-IDF vectorisation + Cosine Similarity.

    Returns verdict: LIKELY_REAL | PARTIALLY_SIMILAR | SUSPICIOUS
    """
    text = (req.text or "").strip()
    if not text or len(text) < 15:
        raise HTTPException(status_code=400, detail="Text must be at least 15 characters.")

    if sim_engine is None:
        raise HTTPException(status_code=503, detail="Similarity engine not loaded. Check server logs.")

    result = sim_engine.check(text, title=req.title or "")
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


@app.post("/api/similarity/add")
async def similarity_add(req: AddArticleRequest):
    """
    Add a new verified article to the knowledge base.
    The TF-IDF index is automatically rebuilt after addition.
    """
    if sim_engine is None:
        raise HTTPException(status_code=503, detail="Similarity engine not loaded.")

    article = {
        "title":    req.title.strip(),
        "text":     req.text.strip(),
        "source":   req.source.strip(),
        "date":     req.date or time.strftime("%Y-%m-%d"),
        "category": req.category.strip(),
        "label":    req.label.strip(),
    }

    result = sim_engine.add_article(article)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to add article"))
    return result


@app.get("/api/similarity/stats")
def similarity_stats():
    """
    Returns knowledge base statistics: total articles, categories, top sources.
    """
    if sim_engine is None:
        return {"ready": False, "total_articles": 0, "message": "Similarity engine not loaded"}
    return sim_engine.get_kb_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  NewsGuard ML Server v2.1")
    print("  Local Fake News Detection — No API Required")
    print("█"*60)
    print(f"  Classical Model : {'✅ Loaded' if bundle.classical_model else '❌ Not found (run train_model.py)'}")
    print(f"  LSTM Model      : {'✅ Loaded' if bundle.dl_model else '⚪ Not available'}")
    if bundle.meta:
        print(f"  Model Name      : {bundle.meta.get('model_name','?')}")
        print(f"  Accuracy        : {bundle.meta.get('accuracy',0)*100:.2f}%")
        print(f"  Vocab Size      : {bundle.meta.get('vocab_size',0):,}")
    if sim_engine:
        stats = sim_engine.get_kb_stats()
        print(f"  Similarity KB   : ✅ {stats['total_articles']} articles · TF-IDF + Cosine")
    else:
        print(f"  Similarity KB   : ⚪ Not loaded")
    print(f"\n  🟢 API → http://127.0.0.1:8000/api/ml/predict")
    print(f"  🟢 API → http://127.0.0.1:8000/api/similarity/check")
    print(f"  🟢 API → http://127.0.0.1:8000/api/similarity/stats")
    print("█"*60 + "\n")

    uvicorn.run("ml_server:app", host="127.0.0.1", port=8000, reload=False)
