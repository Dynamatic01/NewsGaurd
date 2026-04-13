"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       NewsGuard — Similarity Engine                                          ║
║                                                                              ║
║  Compares any news text against a curated Knowledge Base (KB) of verified   ║
║  real articles using TF-IDF vectorization + Cosine Similarity.              ║
║                                                                              ║
║  Verdict logic:                                                              ║
║    max_similarity > HIGH_THRESHOLD  → LIKELY REAL   (similar to known facts)║
║    max_similarity > LOW_THRESHOLD   → PARTIALLY SIMILAR                     ║
║    max_similarity < LOW_THRESHOLD   → SUSPICIOUS    (no known match)        ║
║                                                                              ║
║  Used by: ml_server.py  →  Node.js server.js  →  newsguard.html            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import time
import logging

log = logging.getLogger("newsguard-similarity")

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Thresholds ────────────────────────────────────────────────────────────────
HIGH_THRESHOLD   = 0.30   # above this → LIKELY REAL (strong match)
MEDIUM_THRESHOLD = 0.12   # above this → PARTIALLY SIMILAR
# below MEDIUM_THRESHOLD  → SUSPICIOUS (no known matching article)

TOP_N = 5                  # return top-N matching articles


# ── Dependency imports ────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.warning("scikit-learn not installed — similarity engine disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str) or len(text) < 3:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)   # URLs / emails
    text = re.sub(r"<[^>]+>", " ", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                     # non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityEngine:
    """
    TF-IDF + Cosine Similarity engine backed by a JSON knowledge base file.

    Public methods:
      check(text)           → dict with verdict, score, top matches
      add_article(article)  → appends to KB and rebuilds index
      get_kb_stats()        → KB metadata
    """

    def __init__(self, kb_path: str | None = None):
        self.kb_path = kb_path or os.path.join(DIR, "news_kb.json")
        self.articles: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.ready = False
        self._load_and_build()

    # ── Load KB & build TF-IDF index ─────────────────────────────────────────
    def _load_and_build(self):
        if not HAS_SKLEARN:
            log.warning("scikit-learn missing — similarity engine not available")
            return

        try:
            with open(self.kb_path, encoding="utf-8") as f:
                self.articles = json.load(f)
            log.info(f"✅ Loaded {len(self.articles)} articles from knowledge base")
        except FileNotFoundError:
            self.articles = []
            log.warning(f"⚠ news_kb.json not found at {self.kb_path}")
        except json.JSONDecodeError as e:
            self.articles = []
            log.error(f"Failed to parse news_kb.json: {e}")

        self._rebuild_index()

    def _rebuild_index(self):
        """Build or rebuild TF-IDF matrix from current articles list."""
        if not HAS_SKLEARN or not self.articles:
            self.ready = False
            return

        # Combine title + text for each article into a single document
        docs = [
            clean_text(f"{a.get('title', '')} {a.get('text', '')}")
            for a in self.articles
        ]

        docs = [d for d in docs if d]  # filter empties
        if not docs:
            self.ready = False
            return

        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),       # unigrams + bigrams for better precision
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,        # apply log normalization: 1+log(tf)
            stop_words="english",
        )

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)
            self.ready = True
            log.info(
                f"✅ TF-IDF index built: {self.tfidf_matrix.shape[0]} docs × "
                f"{self.tfidf_matrix.shape[1]} features"
            )
        except Exception as e:
            log.error(f"TF-IDF build failed: {e}")
            self.ready = False

    # ── Main API ──────────────────────────────────────────────────────────────
    def check(self, text: str, title: str = "") -> dict:
        """
        Compare input text against the knowledge base.

        Returns:
          verdict      : 'LIKELY_REAL' | 'PARTIALLY_SIMILAR' | 'SUSPICIOUS'
          max_score    : float 0-1 (highest cosine similarity)
          avg_score    : float 0-1 (average of top-N matches)
          confidence   : 0-100 integer
          top_matches  : list of dicts [{title, source, score, category, date}]
          kb_size      : number of articles in KB
          explanation  : human-readable explanation
        """
        if not self.ready:
            return self._unavailable()

        query = clean_text(f"{title} {text}")
        if len(query) < 15:
            return {
                "verdict": "INSUFFICIENT_TEXT",
                "max_score": 0,
                "avg_score": 0,
                "confidence": 0,
                "top_matches": [],
                "kb_size": len(self.articles),
                "explanation": "Text too short for similarity analysis.",
                "engine": "TF-IDF + Cosine Similarity",
            }

        t0 = time.time()

        # Vectorise query
        q_vec = self.vectorizer.transform([query])

        # Compute cosine similarities against all KB articles
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Top-N indices (sorted desc)
        top_idx = np.argsort(sims)[::-1][:TOP_N]

        max_score = float(sims[top_idx[0]]) if len(top_idx) else 0.0
        avg_score = float(np.mean(sims[top_idx])) if len(top_idx) else 0.0

        # Build top matches
        top_matches = []
        for idx in top_idx:
            score = float(sims[idx])
            if score < 0.01:
                break
            a = self.articles[idx]
            top_matches.append({
                "id":       a.get("id", ""),
                "title":    a.get("title", ""),
                "source":   a.get("source", ""),
                "date":     a.get("date", ""),
                "category": a.get("category", ""),
                "label":    a.get("label", "real"),
                "score":    round(score, 4),
                "pct":      round(score * 100, 1),
            })

        # Verdict
        if max_score >= HIGH_THRESHOLD:
            verdict     = "LIKELY_REAL"
            confidence  = min(int(max_score * 200), 100)
            explanation = (
                f"Strong similarity ({max_score*100:.1f}%) found with "
                f"verified article: \"{top_matches[0]['title'][:80]}…\" "
                f"(Source: {top_matches[0]['source']}). "
                "The content closely matches our database of verified news."
            )
        elif max_score >= MEDIUM_THRESHOLD:
            verdict     = "PARTIALLY_SIMILAR"
            confidence  = min(int(max_score * 300), 70)
            explanation = (
                f"Partial similarity ({max_score*100:.1f}%) found with "
                f"related content in our knowledge base. "
                "Some topic overlap exists but the specific claims could not "
                "be fully matched to a verified article."
            )
        else:
            verdict     = "SUSPICIOUS"
            confidence  = min(int((1 - max_score) * 100), 90)
            explanation = (
                f"No significant match found in our database of "
                f"{len(self.articles)} verified articles "
                f"(highest similarity: {max_score*100:.1f}%). "
                "This does not necessarily mean the article is false — it may "
                "report a new event not yet in our knowledge base. "
                "Verify with trusted sources."
            )

        latency = round((time.time() - t0) * 1000, 1)

        return {
            "verdict":     verdict,
            "max_score":   round(max_score, 4),
            "avg_score":   round(avg_score, 4),
            "max_pct":     round(max_score * 100, 1),
            "confidence":  confidence,
            "top_matches": top_matches,
            "kb_size":     len(self.articles),
            "explanation": explanation,
            "threshold":   {"high": HIGH_THRESHOLD, "medium": MEDIUM_THRESHOLD},
            "engine":      "TF-IDF (bi-gram) + Cosine Similarity",
            "latency_ms":  latency,
        }

    # ── Add article to KB ─────────────────────────────────────────────────────
    def add_article(self, article: dict) -> dict:
        """
        Add a new article to the knowledge base and rebuild the TF-IDF index.

        Required fields: title (str), text (str)
        Optional fields: id, source, date, category, label
        """
        required = {"title", "text"}
        if not required.issubset(article.keys()):
            return {"ok": False, "error": f"Missing required fields: {required}"}
        if len(article.get("text", "")) < 30:
            return {"ok": False, "error": "Article text too short (< 30 chars)"}

        # Generate ID if missing
        if "id" not in article:
            article["id"] = f"user_{int(time.time()*1000)}"

        article.setdefault("source", "user-submitted")
        article.setdefault("date", time.strftime("%Y-%m-%d"))
        article.setdefault("category", "general")
        article.setdefault("label", "real")

        self.articles.append(article)

        # Persist to disk
        try:
            with open(self.kb_path, "w", encoding="utf-8") as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Failed to save KB: {e}")

        # Rebuild index
        self._rebuild_index()

        return {
            "ok":       True,
            "id":       article["id"],
            "kb_size":  len(self.articles),
            "message":  "Article added and index rebuilt successfully",
        }

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_kb_stats(self) -> dict:
        categories = {}
        sources    = {}
        for a in self.articles:
            cat = a.get("category", "general")
            src = a.get("source", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            sources[src]    = sources.get(src, 0) + 1

        return {
            "total_articles": len(self.articles),
            "ready":          self.ready,
            "categories":     categories,
            "top_sources":    dict(sorted(sources.items(), key=lambda x: -x[1])[:10]),
            "engine":         "TF-IDF + Cosine Similarity",
            "thresholds": {
                "high_threshold":   HIGH_THRESHOLD,
                "medium_threshold": MEDIUM_THRESHOLD,
            },
        }

    # ── Fallback ──────────────────────────────────────────────────────────────
    def _unavailable(self) -> dict:
        return {
            "verdict":     "UNAVAILABLE",
            "max_score":   0,
            "avg_score":   0,
            "max_pct":     0,
            "confidence":  0,
            "top_matches": [],
            "kb_size":     len(self.articles),
            "explanation": "Similarity engine not ready. scikit-learn may not be installed.",
            "engine":      "Unavailable",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON (loaded once on server startup)
# ═══════════════════════════════════════════════════════════════════════════════
_engine: SimilarityEngine | None = None


def get_engine() -> SimilarityEngine:
    global _engine
    if _engine is None:
        _engine = SimilarityEngine()
    return _engine


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    engine = SimilarityEngine()
    print(f"\n[Stats] {engine.get_kb_stats()}\n")

    test_texts = [
        "Federal Reserve raised interest rates by 25 basis points as inflation remains above 2 percent target",
        "Scientists discover secret cancer cure that big pharma is hiding from the public exposed today",
        "India's ISRO Chandrayaan-3 mission successfully launched towards the Moon's south pole",
        "Aliens have landed in Nevada and the government is covering it up shocking truth revealed",
        "Bitcoin price reached record highs driven by ETF inflows ahead of halving event",
    ]

    for t in test_texts:
        result = engine.check(t[:80])
        print(f"TEXT : {t[:70]}...")
        print(f"  Verdict: {result['verdict']}  |  Max sim: {result['max_pct']}%  |  Confidence: {result['confidence']}")
        if result["top_matches"]:
            m = result["top_matches"][0]
            print(f"  Best match: [{m['score']}] {m['title'][:60]}... ({m['source']})")
        print()
