# Redrob Hackathon — AI Candidate Ranker

Ranks candidates for the **Senior AI Engineer** role using a hybrid system:
sentence-transformer semantic similarity + engineered features + Gradient Boosting re-ranker + behavioral signal multiplier.

## Architecture

```
candidates.jsonl
       │
       ▼
precompute.py  →  artifacts/  (embeddings, one-time)
       │
       ▼
rank.py  →  submission.csv
```

### Scoring pipeline

| Stage | Method | What it captures |
|---|---|---|
| Semantic | sentence-transformers `all-MiniLM-L6-v2` cosine similarity vs JD | Holistic profile–JD match |
| Features | 14 engineered signals | Title, career AI depth, skills, production evidence, YoE, GitHub, location |
| Re-ranker | scikit-learn `GradientBoostingRegressor` | Learned combination of all features |
| Multiplier | Behavioral signals | Demotes unreachable/inactive candidates |

### Key JD-aligned design decisions
- Consulting-only careers (TCS/Infosys/Wipro with no product history) penalised 55%
- Honeypot detection removes impossible profiles before ranking
- Behavioral multiplier range `[0.35, 1.0]` — inactive candidates can't top the list
- Semantic score from sentence-transformers captures meaning, not keyword overlap
- Production deployment evidence extracted from career descriptions

## Requirements

```bash
pip install scikit-learn sentence-transformers numpy scipy python-docx
```

## Reproduce submission

### Step 1 — Pre-compute embeddings (one-time, outside 5-min budget)
```bash
python precompute.py --candidates candidates.jsonl.gz --out-dir artifacts/
```

### Step 2 — Run ranker (must finish in <5 min, CPU-only)
```bash
python rank.py --candidates candidates.jsonl.gz --artifacts artifacts/ --out submission.csv
```

### Step 3 — Validate
```bash
python validate.py --submission submission.csv --candidates candidates.jsonl.gz
```

### Sample data test
```bash
python precompute.py --candidates sample_candidates.json --out-dir artifacts/
python rank.py --candidates sample_candidates.json --artifacts artifacts/ --out submission.csv
```

## Compute budget compliance

| Constraint | Status |
|---|---|
| CPU-only | ✓ |
| No external API calls during ranking | ✓ |
| ≤5 min ranking step | ✓ (~45s on 100K) |
| ≤16 GB RAM | ✓ (~1.5 GB) |

## File structure

```
rank.py                  — main ranker (hybrid ML pipeline)
precompute.py            — one-time embedding pre-computation
validate.py              — submission format validator
requirements.txt         — dependencies
submission_metadata.yaml — team metadata
candidate_schema.json    — reference schema
sample_candidates.json   — 50-candidate sample for testing
.gitignore               — excludes large data files and artifacts
```
