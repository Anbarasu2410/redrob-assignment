"""
precompute.py — One-time pre-computation step (may exceed 5-min ranking budget)
=================================================================================
Generates:
  - artifacts/candidate_embeddings.npz  — sentence-transformer embeddings per candidate
  - artifacts/candidate_ids.json        — ordered list of candidate IDs matching rows

Usage:
    python precompute.py --candidates candidates.jsonl --out-dir artifacts/
    python precompute.py --candidates candidates.jsonl.gz --out-dir artifacts/
    python precompute.py --candidates sample_candidates.json --out-dir artifacts/

Model: all-MiniLM-L6-v2 (free, 80MB, CPU-friendly, ~500 candidates/sec)
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────
# JD text (verbatim key passages for embedding)
# ─────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer role at Series A AI-native talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems.
Production experience with embeddings-based retrieval systems: sentence-transformers,
OpenAI embeddings, BGE, E5, deployed to real users.
Production experience with vector databases or hybrid search: Pinecone, Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python. Code quality matters.
Hands-on experience designing evaluation frameworks for ranking: NDCG, MRR, MAP,
offline-to-online correlation, A/B test interpretation.
LLM fine-tuning experience: LoRA, QLoRA, PEFT.
Learning-to-rank models: XGBoost, neural LTR.
5-9 years experience, mostly in applied ML/AI roles at product companies.
Shipped at least one end-to-end ranking, search, or recommendation system to real users at scale.
Strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online),
LLM integration (fine-tune vs prompt).
Located in or willing to relocate to Noida or Pune, India.
Active in job market.
Not pure research, must have production deployment.
NLP and information retrieval background required.
Not consulting firms only (TCS, Infosys, Wipro, Accenture, Cognizant).
"""


def build_candidate_text(c: dict) -> str:
    """Concatenate the richest text fields of a candidate profile for embedding."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
    ]

    # Career descriptions
    for role in c.get("career_history", []):
        parts.append(f"{role.get('title', '')} at {role.get('company', '')}: {role.get('description', '')}")

    # Skills
    skill_parts = []
    for s in c.get("skills", []):
        skill_parts.append(f"{s.get('name', '')} ({s.get('proficiency', '')})")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    # Certifications
    for cert in c.get("certifications", []):
        parts.append(f"Certification: {cert.get('name', '')} from {cert.get('issuer', '')}")

    return " | ".join(filter(None, parts))


def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".gz":
        opener = gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = open(p, "r", encoding="utf-8")

    with opener as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Pre-compute sentence-transformer embeddings")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidates from {args.candidates}...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.", file=sys.stderr)

    print(f"Loading sentence-transformer model: {args.model} ...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # Encode JD
    print("Encoding JD...", file=sys.stderr)
    jd_emb = model.encode([JD_TEXT.strip()], normalize_embeddings=True, show_progress_bar=False)

    # Encode all candidates in batches
    print(f"Encoding {len(candidates)} candidates (batch={args.batch_size})...", file=sys.stderr)
    texts = [build_candidate_text(c) for c in candidates]
    cand_ids = [c.get("candidate_id", f"CAND_{i:07d}") for i, c in enumerate(candidates)]

    cand_embs = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Save artifacts
    emb_path = out_dir / "candidate_embeddings.npz"
    ids_path = out_dir / "candidate_ids.json"
    jd_path = out_dir / "jd_embedding.npy"

    np.savez_compressed(emb_path, embeddings=cand_embs)
    np.save(jd_path, jd_emb)
    with open(ids_path, "w") as f:
        json.dump(cand_ids, f)

    print(f"Saved embeddings → {emb_path}", file=sys.stderr)
    print(f"Saved JD embedding → {jd_path}", file=sys.stderr)
    print(f"Saved candidate IDs → {ids_path}", file=sys.stderr)
    print("Pre-computation complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
