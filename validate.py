"""
validate.py — Submission CSV validator
Usage: python validate.py --submission submission.csv --candidates candidates.jsonl
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


def load_candidate_ids(path: str) -> set:
    p = Path(path)
    ids = set()
    opener = gzip.open(p, "rt") if p.suffix == ".gz" else open(p, "r", encoding="utf-8")
    with opener as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            ids = {c["candidate_id"] for c in data}
        else:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["candidate_id"])
    return ids


def validate(submission_path: str, candidates_path: str):
    errors = []
    warnings = []

    valid_ids = load_candidate_ids(candidates_path)

    with open(submission_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check column names
    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        errors.append(f"Missing columns. Required: {required_cols}. Got: {reader.fieldnames}")
        print("\n".join(errors)); return False

    # Check row count
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    seen_ids = set()
    seen_ranks = set()
    prev_score = None

    for i, row in enumerate(rows):
        cid = row.get("candidate_id", "").strip()
        rank_str = row.get("rank", "").strip()
        score_str = row.get("score", "").strip()
        reasoning = row.get("reasoning", "").strip()

        # candidate_id validity
        if cid not in valid_ids:
            errors.append(f"Row {i+1}: candidate_id '{cid}' not found in candidates file")
        if cid in seen_ids:
            errors.append(f"Row {i+1}: duplicate candidate_id '{cid}'")
        seen_ids.add(cid)

        # rank validity
        try:
            rank = int(rank_str)
            if rank != i + 1:
                errors.append(f"Row {i+1}: rank should be {i+1}, got {rank}")
            if rank in seen_ranks:
                errors.append(f"Row {i+1}: duplicate rank {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {i+1}: invalid rank '{rank_str}'")

        # score validity
        try:
            score = float(score_str)
            if not (0.0 <= score <= 1.0):
                errors.append(f"Row {i+1}: score {score} out of [0,1] range")
            if prev_score is not None and score > prev_score + 1e-6:
                errors.append(f"Row {i+1}: score {score} > previous score {prev_score} (must be non-increasing)")
            prev_score = score
        except ValueError:
            errors.append(f"Row {i+1}: invalid score '{score_str}'")

        # reasoning
        if not reasoning:
            warnings.append(f"Row {i+1}: empty reasoning (will be penalized in Stage 4)")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    else:
        print(f"VALIDATION PASSED — {len(rows)} rows, no errors.")
        if warnings:
            print(f"  {len(warnings)} warnings (empty reasoning fields)")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()
    ok = validate(args.submission, args.candidates)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
