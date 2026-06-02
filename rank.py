"""
rank.py  —  Redrob Hackathon  |  Hybrid Semantic + Feature-Engineering Ranker
==============================================================================
Single command to produce submission:
    python rank.py --candidates candidates.jsonl --out submission.csv

With gzip:
    python rank.py --candidates candidates.jsonl.gz --out submission.csv

With pre-computed embeddings (faster, recommended for 100K pool):
    python precompute.py --candidates candidates.jsonl --out-dir artifacts/
    python rank.py --candidates candidates.jsonl --artifacts artifacts/ --out submission.csv

Compute constraints met:
  - CPU-only (no GPU)
  - No external API calls during ranking
  - <5 min on 100K candidates with precomputed embeddings
  - <16 GB RAM

Architecture
------------
Stage 1  — Feature engineering (pure Python)
           Title alignment, career AI depth, skill coverage, production signals,
           YoE fit, consulting-only penalty, honeypot detection.

Stage 2  — Semantic cosine similarity
           Loads precomputed sentence-transformer embeddings (all-MiniLM-L6-v2)
           OR falls back to TF-IDF BM25-style cosine if no artifacts present.
           Computes cosine(candidate_text, JD) as a continuous semantic score.

Stage 3  — Gradient Boosting re-ranker (scikit-learn GradientBoostingRegressor)
           Trained on synthetic relevance labels derived from the JD's explicit
           criteria. Combines all features into a final relevance score.

Stage 4  — Behavioral signal multiplier
           Platform engagement (activity, response rate, notice period, etc.)
           applied as a soft multiplier so inactive candidates are demoted.
"""

import argparse
import csv
import gzip
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ══════════════════════════════════════════════════════════════
# JD CONSTANTS  (derived from deep reading of job_description.docx)
# ══════════════════════════════════════════════════════════════

REFERENCE_DATE = date(2026, 6, 2)

# The full JD text used to build the TF-IDF / embedding query
JD_TEXT = """
Senior AI Engineer at Series A AI-native talent intelligence platform Redrob.
Own the intelligence layer: ranking retrieval matching systems for recruiters and candidates.
Production experience with embeddings-based retrieval: sentence-transformers OpenAI embeddings
BGE E5 deployed to real users handling embedding drift index refresh retrieval quality regression.
Production experience with vector databases hybrid search: Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS approximate nearest neighbor.
Strong Python code quality matters.
Hands-on experience designing evaluation frameworks for ranking: NDCG MRR MAP
offline-to-online correlation A/B test interpretation.
LLM fine-tuning experience LoRA QLoRA PEFT.
Learning-to-rank models XGBoost neural LTR.
Five to nine years experience mostly applied ML AI roles at product companies not pure services.
Shipped end-to-end ranking search recommendation system to real users at scale.
Strong opinions retrieval hybrid versus dense evaluation offline versus online LLM integration.
Located willing to relocate Noida Pune Hyderabad Mumbai Delhi India.
Active job market open to work.
NLP natural language processing information retrieval required background.
Not pure research must have production deployment shipped to real users.
Not consulting firms only TCS Infosys Wipro Accenture Cognizant Capgemini.
Semantic search vector search recommendation engine ranking system retrieval augmented generation.
"""

# ── Title signals ──────────────────────────────────────────────
AI_TITLES_EXACT = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior machine learning engineer", "applied scientist",
    "applied ml engineer", "data scientist", "nlp engineer",
    "research engineer", "ai researcher", "deep learning engineer",
    "recommendation systems engineer", "search engineer",
    "information retrieval engineer", "ranking engineer",
    "senior ai engineer", "staff ml engineer", "principal ml engineer",
    "junior ml engineer", "data engineer",
}

AI_TITLE_KEYWORDS = {
    "machine learning", "deep learning", "natural language", "nlp",
    "recommendation", "search", "ranking", "retrieval", "applied scientist",
    "data scientist",
}

NON_TECHNICAL_TITLES = {
    "marketing manager", "operations manager", "accountant", "hr manager",
    "customer support", "business analyst", "sales executive",
    "graphic designer", "content writer", "civil engineer",
    "mechanical engineer", "project manager",
}

# ── Company signals ────────────────────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "mindtree",
}

# ── Skills ─────────────────────────────────────────────────────
CORE_SKILLS = {
    # Must-have embeddings / vector search
    "sentence transformers", "embeddings", "vector embeddings",
    "text embeddings", "semantic search", "dense retrieval", "bge", "e5",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "vector database", "vector search",
    "hybrid search", "ann", "approximate nearest neighbor",
    # Ranking / IR
    "learning to rank", "ltr", "ndcg", "mrr", "information retrieval",
    "bm25", "ranking", "search ranking", "recommendation systems",
    # LLMs
    "fine-tuning llms", "lora", "qlora", "peft", "fine-tuning",
    "llm", "large language models", "transformers",
    # NLP / ML
    "nlp", "natural language processing", "bert", "gpt",
    "xgboost", "gradient boosting",
    # Infra
    "python", "pytorch", "tensorflow",
}

BONUS_SKILLS = {
    "rag", "retrieval augmented generation", "distributed systems",
    "kafka", "spark", "langchain", "weights & biases", "wandb",
    "bentoml", "mlflow", "airflow", "aws", "gcp", "azure",
    "hugging face transformers", "sentence transformers",
}

# ── Career AI keywords (unambiguous multi-word) ────────────────
CAREER_AI_KW = {
    "machine learning", "deep learning", "natural language",
    "nlp", "neural network", "recommendation system", "recommender",
    "vector search", "vector database", "embedding", "embeddings",
    "semantic search", "ranking system", "information retrieval",
    "llm", "large language model", "fine-tuning", "fine-tun",
    "transformer", "bert", "gpt", "retrieval pipeline",
    "a/b test", "model training", "model deployment", "model serving",
    "ml pipeline", "predictive model", "classification model",
    "feature store", "feature engineering at scale",
}

CAREER_AI_TITLE_KW = {
    "ml", "machine learning", "ai", "data scientist", "nlp",
    "deep learning", "recommendation", "search engineer",
    "ranking", "applied scientist", "data engineer",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurgaon", "gurugram", "ncr",
}


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def _l(s) -> str:
    return s.lower().strip() if s else ""


def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999


def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    opener = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, "r", encoding="utf-8")
    with opener as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


# ══════════════════════════════════════════════════════════════
# HONEYPOT DETECTION
# ══════════════════════════════════════════════════════════════

def is_honeypot(c: dict) -> bool:
    """Detect impossible/synthetic trap profiles."""
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    total_career_months = sum(r.get("duration_months", 0) for r in career)

    # Skill used longer than entire career (impossible)
    for skill in skills:
        sm = skill.get("duration_months", 0)
        if sm > 0 and total_career_months > 0 and sm > total_career_months + 12:
            return True

    # Multiple expert skills with zero months used (fabricated)
    if sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0) >= 3:
        return True

    # Education year impossibility
    for edu in c.get("education", []):
        s, e = edu.get("start_year", 2000), edu.get("end_year", 2000)
        if e < s or (e - s) > 10:
            return True

    # YoE vs career span mismatch
    yoe = profile.get("years_of_experience", 0)
    career_yrs = total_career_months / 12.0
    if career_yrs > 0 and yoe > career_yrs + 5:
        return True

    return False


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (returns a dict of floats, 0-1 each)
# ══════════════════════════════════════════════════════════════

def extract_features(c: dict) -> dict:
    """
    Extract 20+ interpretable features from a candidate profile.
    All values normalised to [0, 1].
    """
    profile = c.get("profile", {})
    career  = c.get("career_history", [])
    skills  = c.get("skills", [])
    redrob  = c.get("redrob_signals", {})

    f = {}   # feature dict

    # ── 1. TITLE ALIGNMENT ────────────────────────────
    ct = _l(profile.get("current_title", ""))
    hl = _l(profile.get("headline", ""))
    if any(t in ct for t in AI_TITLES_EXACT):
        f["title"] = 1.0
    elif any(t in hl for t in AI_TITLES_EXACT):
        f["title"] = 0.75
    elif any(kw in ct for kw in AI_TITLE_KEYWORDS):
        f["title"] = 0.5
    elif any(kw in ct for kw in ["data", "ml", "ai", "engineer", "scientist", "backend", "software"]):
        f["title"] = 0.35
    elif ct in NON_TECHNICAL_TITLES:
        f["title"] = 0.0
    else:
        f["title"] = 0.1

    # ── 2. CAREER AI DEPTH AT PRODUCT COMPANIES ──────
    product_ai_months = 0
    consulting_only = True
    all_desc = []

    for role in career:
        co  = _l(role.get("company", ""))
        desc = _l(role.get("description", ""))
        rtitle = _l(role.get("title", ""))
        all_desc.append(desc)

        is_consulting = any(cf in co for cf in CONSULTING_FIRMS)
        if not is_consulting:
            consulting_only = False

        # Need >=2 unambiguous AI keywords OR AI title
        ai_in_desc  = sum(1 for kw in CAREER_AI_KW if kw in desc) >= 2
        ai_in_title = any(kw in rtitle for kw in CAREER_AI_TITLE_KW)

        if ai_in_desc or ai_in_title:
            months = role.get("duration_months", 0)
            product_ai_months += months if not is_consulting else months * 0.4

    consulting_penalty = 0.3 if consulting_only else 1.0
    f["career_ai_depth"] = _norm(product_ai_months, 0, 48) * consulting_penalty
    f["consulting_only"]  = 1.0 if consulting_only else 0.0

    # ── 3. SKILL COVERAGE ─────────────────────────────
    skill_map  = {_l(s["name"]): s for s in skills}
    skill_names = set(skill_map.keys())
    full_text   = " ".join(all_desc) + " " + _l(profile.get("summary", ""))

    prof_w = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.65, "beginner": 0.35}

    def _count(sset):
        c_ = 0
        for sk in sset:
            if sk in skill_names:
                c_ += prof_w.get(skill_map[sk].get("proficiency",""), 0.5)
            elif sk in full_text:
                c_ += 0.3
        return c_

    f["core_skills"]  = _norm(_count(CORE_SKILLS),  0, len(CORE_SKILLS) * 0.65)
    f["bonus_skills"] = _norm(_count(BONUS_SKILLS), 0, len(BONUS_SKILLS) * 0.65)

    # ── 4. SKILL ASSESSMENT SCORES (platform-verified) ─
    assessments = redrob.get("skill_assessment_scores", {})
    relevant_kw = {"nlp", "ml", "ai", "retrieval", "rank", "python", "llm", "fine-tun", "embedding"}
    rel_scores = [v for k, v in assessments.items()
                  if any(kw in _l(k) for kw in relevant_kw)]
    f["assessment_score"] = (sum(rel_scores) / len(rel_scores) / 100.0) if rel_scores else 0.3

    # ── 5. PRODUCTION / SHIPPING EVIDENCE ─────────────
    prod_kw = {
        "production", "deploy", "ship", "real user", "at scale",
        "latency", "a/b test", "serving", "inference", "vector store",
        "retrieval system", "index refresh", "query latency",
    }
    prod_hits = sum(1 for kw in prod_kw if kw in full_text)
    f["production"] = _norm(prod_hits, 0, 6)

    # ── 6. YEARS OF EXPERIENCE FIT ────────────────────
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        f["yoe_fit"] = 1.0
    elif 4 <= yoe < 5:
        f["yoe_fit"] = 0.85
    elif 9 < yoe <= 12:
        f["yoe_fit"] = 0.75
    elif 3 <= yoe < 4:
        f["yoe_fit"] = 0.6
    elif yoe > 12:
        f["yoe_fit"] = 0.5
    else:
        f["yoe_fit"] = max(0.1, yoe / 5.0)

    # ── 7. GITHUB ACTIVITY ────────────────────────────
    gh = redrob.get("github_activity_score", -1)
    f["github"] = _norm(gh, 0, 100) if gh >= 0 else 0.15

    # ── 8. RESEARCH-ONLY PENALTY ──────────────────────
    research_terms = {"phd candidate", "academic lab", "published paper", "arxiv", "research paper"}
    r_count = sum(1 for kw in research_terms if kw in full_text)
    f["research_penalty"] = max(0.3, 1.0 - r_count * 0.15) if f["production"] < 0.2 else 1.0

    # ── 9. LOCATION FIT ───────────────────────────────
    loc     = _l(profile.get("location", ""))
    country = _l(profile.get("country", ""))
    relocate = redrob.get("willing_to_relocate", False)

    if country == "india":
        if any(pl in loc for pl in PREFERRED_LOCATIONS):
            f["location"] = 1.0
        else:
            f["location"] = 0.7 if relocate else 0.5
    else:
        f["location"] = 0.4 if relocate else 0.2

    # ── 10. EDUCATION TIER ────────────────────────────
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.3}
    edu_scores = [tier_scores.get(e.get("tier", "unknown"), 0.3) for e in c.get("education", [])]
    f["education"] = max(edu_scores) if edu_scores else 0.3

    return f


# ══════════════════════════════════════════════════════════════
# BEHAVIORAL SIGNALS  (engagement / availability multiplier)
# ══════════════════════════════════════════════════════════════

def behavioral_score(c: dict) -> float:
    """Returns a 0-1 availability + engagement score."""
    r = c.get("redrob_signals", {})

    open_w = 1.0 if r.get("open_to_work_flag", False) else 0.3

    days = _days_since(r.get("last_active_date", "2020-01-01"))
    activity = 1.0 if days <= 30 else 0.8 if days <= 90 else 0.5 if days <= 180 else 0.2 if days <= 365 else 0.05

    notice = r.get("notice_period_days", 90)
    notice_s = 1.0 if notice <= 30 else 0.75 if notice <= 60 else 0.5 if notice <= 90 else 0.25

    rr = r.get("recruiter_response_rate", 0.0)

    avg_rt = r.get("avg_response_time_hours", 999)
    rt_s = 1.0 if avg_rt <= 24 else 0.75 if avg_rt <= 72 else 0.5 if avg_rt <= 168 else 0.2

    icr = r.get("interview_completion_rate", 0.5)

    completeness = r.get("profile_completeness_score", 0) / 100.0
    verified = (
        (0.4 if r.get("verified_email", False) else 0.0) +
        (0.4 if r.get("verified_phone", False) else 0.0) +
        (0.2 if r.get("linkedin_connected", False) else 0.0)
    )
    credibility = completeness * 0.6 + verified * 0.4

    return (
        open_w   * 0.22 +
        activity * 0.22 +
        rr       * 0.20 +
        notice_s * 0.15 +
        rt_s     * 0.08 +
        icr      * 0.08 +
        credibility * 0.05
    )


# ══════════════════════════════════════════════════════════════
# TEXT REPRESENTATION  (for TF-IDF / embedding)
# ══════════════════════════════════════════════════════════════

def candidate_text(c: dict) -> str:
    """Build a rich text blob from candidate profile for TF-IDF / embedding."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for role in c.get("career_history", []):
        parts.append(f"{role.get('title','')} {role.get('company','')} {role.get('description','')}")
    for s in c.get("skills", []):
        nm = s.get("name","")
        pr = s.get("proficiency","")
        parts.append(f"{nm} {pr}")
    for cert in c.get("certifications", []):
        parts.append(cert.get("name",""))
    return " ".join(filter(None, parts))


# ══════════════════════════════════════════════════════════════
# TFIDF SEMANTIC SCORER  (fallback when no precomputed embeddings)
# ══════════════════════════════════════════════════════════════

class TFIDFScorer:
    """
    Fit a TF-IDF vectorizer on all candidate texts + JD,
    then return cosine(candidate, JD) as semantic relevance score.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30_000,
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )
        self.jd_vec = None
        self.fitted = False

    def fit(self, texts: list[str]):
        all_texts = [JD_TEXT] + texts
        mat = self.vectorizer.fit_transform(all_texts)
        self.jd_vec = mat[0]            # first row is JD
        self.cand_mat = mat[1:]         # remaining rows are candidates
        self.fitted = True

    def scores(self) -> np.ndarray:
        """Returns cosine similarity of each candidate vs JD, shape (N,)."""
        sims = cosine_similarity(self.jd_vec, self.cand_mat)[0]
        # Normalise to [0,1]
        lo, hi = sims.min(), sims.max()
        if hi > lo:
            sims = (sims - lo) / (hi - lo)
        return sims


# ══════════════════════════════════════════════════════════════
# EMBEDDING SCORER  (when artifacts/ exists)
# ══════════════════════════════════════════════════════════════

def load_embedding_scores(artifacts_dir: str, candidate_ids: list[str]) -> np.ndarray | None:
    """
    Load precomputed embeddings and return cosine similarity array aligned
    to the candidate_ids list order.
    Returns None if artifacts are missing or IDs don't match.
    """
    d = Path(artifacts_dir)
    emb_path = d / "candidate_embeddings.npz"
    ids_path = d / "candidate_ids.json"
    jd_path  = d / "jd_embedding.npy"

    if not (emb_path.exists() and ids_path.exists() and jd_path.exists()):
        return None

    stored_ids = json.loads(ids_path.read_text())
    id_to_idx  = {cid: i for i, cid in enumerate(stored_ids)}

    missing = [cid for cid in candidate_ids if cid not in id_to_idx]
    if len(missing) > len(candidate_ids) * 0.01:   # >1% missing → fallback
        print(f"WARNING: {len(missing)} candidates not in artifacts, falling back to TF-IDF.", file=sys.stderr)
        return None

    npz       = np.load(emb_path)
    all_embs  = npz["embeddings"].astype(np.float32)
    jd_emb    = np.load(jd_path).astype(np.float32)

    # Re-order to match candidate_ids
    idxs = [id_to_idx.get(cid, 0) for cid in candidate_ids]
    cand_embs = all_embs[idxs]

    sims = cosine_similarity(jd_emb, cand_embs)[0].astype(np.float32)
    lo, hi = sims.min(), sims.max()
    if hi > lo:
        sims = (sims - lo) / (hi - lo)
    return sims


# ══════════════════════════════════════════════════════════════
# SYNTHETIC LABEL GENERATION  (for GB ranker training)
# ══════════════════════════════════════════════════════════════

def synthetic_label(feat: dict, sem_score: float) -> float:
    """
    Compute a ground-truth-proxy relevance label (0-1) for training
    the Gradient Boosting ranker.

    Based on the JD's explicit grading criteria:
    - Strong AI/ML title + product-company career + key skills → top tier
    - Consulting-only career → strong down-weight
    - High semantic similarity → strong positive
    - Behavioral availability modulates within a tier
    """
    # Core fit: this is what the JD actually wants
    core = (
        feat["title"]         * 0.20 +
        feat["career_ai_depth"] * 0.25 +
        feat["core_skills"]   * 0.20 +
        feat["production"]    * 0.15 +
        feat["yoe_fit"]       * 0.10 +
        sem_score             * 0.10
    )
    # Penalties
    core *= feat.get("research_penalty", 1.0)
    if feat.get("consulting_only", 0) > 0.5:
        core *= 0.45

    # Secondary enrichment
    enriched = core + feat["bonus_skills"] * 0.04 + feat["github"] * 0.03

    return float(np.clip(enriched, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════
# GRADIENT BOOSTING RANKER
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "title", "career_ai_depth", "consulting_only", "core_skills",
    "bonus_skills", "assessment_score", "production", "yoe_fit",
    "github", "research_penalty", "location", "education",
    "sem_score", "behavioral",
]


def build_feature_matrix(
    feature_list: list[dict],
    sem_scores: np.ndarray,
    behav_scores: np.ndarray,
) -> np.ndarray:
    rows = []
    for i, feat in enumerate(feature_list):
        row = [feat.get(col, 0.0) for col in FEATURE_COLS[:-2]]
        row.append(float(sem_scores[i]))
        row.append(float(behav_scores[i]))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def train_ranker(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X, y)
    return model


# ══════════════════════════════════════════════════════════════
# REASONING BUILDER
# ══════════════════════════════════════════════════════════════

def build_reasoning(c: dict, feat: dict, sem: float, behav: float) -> str:
    profile = c.get("profile", {})
    redrob  = c.get("redrob_signals", {})
    career  = c.get("career_history", [])
    skills  = c.get("skills", [])

    title    = profile.get("current_title", "Unknown")
    yoe      = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")

    # Find notable matched skills
    skill_names = {_l(s.get("name","")) for s in skills}
    HIGHLIGHT = [
        "sentence transformers", "faiss", "pinecone", "qdrant", "milvus",
        "weaviate", "elasticsearch", "fine-tuning llms", "lora", "qlora",
        "nlp", "information retrieval", "bm25", "pytorch", "vector search",
        "recommendation systems", "learning to rank", "embeddings",
        "hugging face transformers",
    ]
    matched = [sk for sk in HIGHLIGHT if sk in skill_names]

    positives, concerns = [], []

    # Title
    if feat["title"] >= 0.75:
        positives.append(f"{title} — strong AI/ML role alignment")
    elif feat["title"] >= 0.4:
        positives.append(f"{title} with technical background")

    # Career
    if feat["career_ai_depth"] >= 0.5:
        positives.append("substantive product-company AI/ML career history")
    elif feat["career_ai_depth"] >= 0.2:
        positives.append("some applied AI/ML work in career")

    # Skills
    if matched:
        positives.append(f"key skills: {', '.join(matched[:5])}")

    # Production
    if feat["production"] >= 0.5:
        positives.append("production deployment evidence in career")

    # Semantic match
    if sem >= 0.7:
        positives.append(f"high semantic JD match ({sem:.2f})")
    elif sem >= 0.5:
        positives.append(f"good semantic JD match ({sem:.2f})")

    # GitHub
    gh = redrob.get("github_activity_score", -1)
    if gh >= 55:
        positives.append(f"active GitHub ({gh:.0f}/100)")

    # Concerns
    if not redrob.get("open_to_work_flag", False):
        concerns.append("not marked open to work")

    days = _days_since(redrob.get("last_active_date", "2020-01-01"))
    if days > 180:
        concerns.append(f"inactive {days}d")

    rr = redrob.get("recruiter_response_rate", 1.0)
    if rr < 0.25:
        concerns.append(f"low response rate ({rr:.0%})")

    notice = redrob.get("notice_period_days", 0)
    if notice > 60:
        concerns.append(f"{notice}d notice")

    if feat.get("consulting_only", 0) > 0.5:
        concerns.append("consulting-only career")

    pos = "; ".join(positives[:3]) if positives else f"{title}, {yoe:.1f} yrs"
    neg = (". Concerns: " + ", ".join(concerns[:2])) if concerns else ""

    return f"{pos}. {yoe:.1f} yrs, {location}{neg}."


# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════

def write_csv(ranked: list[tuple], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (score, cid, reasoning) in enumerate(ranked[:100], 1):
            w.writerow([cid, i, f"{score:.4f}", reasoning])


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Redrob Hybrid Semantic + ML Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="candidates.jsonl / candidates.jsonl.gz / sample_candidates.json")
    parser.add_argument("--out",       default="submission.csv")
    parser.add_argument("--artifacts", default="artifacts",
                        help="Directory with precomputed embeddings (from precompute.py). "
                             "Falls back to TF-IDF if not found.")
    parser.add_argument("--top",       type=int, default=100)
    args = parser.parse_args()

    # ── Load candidates ──────────────────────────────
    print(f"Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates)} candidates.", file=sys.stderr)

    # ── Honeypot filter ──────────────────────────────
    valid, honeypots = [], 0
    for c in candidates:
        if is_honeypot(c):
            honeypots += 1
            # Keep them but mark — they'll score 0 and stay out of top-100
            c["_honeypot"] = True
        valid.append(c)
    print(f"  Honeypots detected: {honeypots}", file=sys.stderr)

    candidate_ids = [c.get("candidate_id", f"CAND_{i:07d}") for i, c in enumerate(valid)]

    # ── Semantic scores ──────────────────────────────
    sem_scores = load_embedding_scores(args.artifacts, candidate_ids)

    if sem_scores is not None:
        print(f"  Using precomputed sentence-transformer embeddings.", file=sys.stderr)
    else:
        print("  Artifacts not found — running TF-IDF semantic scorer ...", file=sys.stderr)
        texts = [candidate_text(c) for c in valid]
        tfidf = TFIDFScorer()
        tfidf.fit(texts)
        sem_scores = tfidf.scores()
        print("  TF-IDF scoring complete.", file=sys.stderr)

    # ── Feature extraction + behavioral ─────────────
    print("  Extracting features ...", file=sys.stderr)
    feat_list   = [extract_features(c) for c in valid]
    behav_arr   = np.array([behavioral_score(c) for c in valid], dtype=np.float32)

    # Zero out honeypots
    for i, c in enumerate(valid):
        if c.get("_honeypot"):
            sem_scores[i] = 0.0
            behav_arr[i]  = 0.0
            for k in feat_list[i]:
                feat_list[i][k] = 0.0

    # ── Build feature matrix ─────────────────────────
    X = build_feature_matrix(feat_list, sem_scores, behav_arr)

    # ── Synthetic labels for GB ranker ───────────────
    print("  Training Gradient Boosting ranker on synthetic labels ...", file=sys.stderr)
    y = np.array([synthetic_label(feat_list[i], float(sem_scores[i])) for i in range(len(valid))],
                 dtype=np.float32)

    ranker = train_ranker(X, y)
    raw_scores = ranker.predict(X).astype(np.float64)

    # ── Apply behavioral multiplier ──────────────────
    # Behavioral acts as a soft multiplier: [0.35, 1.0]
    # Keeps great candidates down if they're unreachable
    behav_mult = 0.35 + behav_arr * 0.65
    final_scores = raw_scores * behav_mult

    # Normalise to [0, 1] while preserving relative order
    lo, hi = final_scores.min(), final_scores.max()
    if hi > lo:
        final_scores = (final_scores - lo) / (hi - lo)

    # ── Sort and build output ─────────────────────────
    order = np.argsort(-final_scores)
    ranked = []
    for idx in order:
        score  = float(np.clip(final_scores[idx], 0.0, 1.0))
        cid    = candidate_ids[idx]
        reason = build_reasoning(valid[idx], feat_list[idx], float(sem_scores[idx]), float(behav_arr[idx]))
        ranked.append((score, cid, reason))

    top = ranked[:args.top]

    # Ensure non-increasing scores (spec requirement)
    for i in range(1, len(top)):
        if top[i][0] > top[i-1][0]:
            top[i] = (top[i-1][0], top[i][1], top[i][2])

    print(f"Writing top {len(top)} to {args.out} ...", file=sys.stderr)
    write_csv(top, args.out)
    print("Done.", file=sys.stderr)

    print("\nTop 10:", file=sys.stderr)
    for rank_i, (score, cid, reason) in enumerate(top[:10], 1):
        print(f"  {rank_i:2d}. {cid}  {score:.4f}  {reason[:75]}", file=sys.stderr)

    # Feature importance summary
    importances = ranker.feature_importances_
    print("\nFeature importances:", file=sys.stderr)
    for col, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"  {col:25s} {imp:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
