from itertools import combinations
from pathlib import Path

import numpy as np
from rich import print
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen3-Embedding-0.6B"

SURVEYS = ["depression", "dependence", "gaslighting"]

TASK = (
    "Given a survey question, represent the underlying construct "
    "it measures, ignoring wording and response format"
)

PROMPT = f"Instruct: {TASK}\nQuery:"

# Above this, we consider two items to be “similar”.
THRESHOLD = 0.3

# Convenience for printing.
LINE_LENGTH_MAX = 120


def main() -> None:
    model: SentenceTransformer = SentenceTransformer(MODEL)

    embedding_map: dict[str, tuple[list[str], np.ndarray]] = {}

    # 1. Embed all questions.
    for survey in SURVEYS:
        path = Path(__file__).parents[2] / "surveys" / f"{survey}.txt"
        with open(path) as s:
            questions: list[str] = [l for l in (line.strip() for line in s) if l]
            emb = model.encode(questions, prompt=PROMPT, normalize_embeddings=True)
            embedding_map[survey] = (questions, emb)

    # 2. Center on the mean of all items and renormalise
    all_emb = np.vstack([e for _, e in embedding_map.values()])
    mu = all_emb.mean(axis=0)
    for survey, (questions, emb) in embedding_map.items():
        c = emb - mu
        embedding_map[survey] = (
            questions,
            c / np.linalg.norm(c, axis=1, keepdims=True),
        )

    # 3. Compute pairwise similarities between surveys
    over_threshold: list[tuple[float, str, str, int, int, str, str]] = []
    for a, b in combinations(SURVEYS, 2):
        qa, ea = embedding_map[a]
        qb, eb = embedding_map[b]
        sim = ea @ eb.T
        print(f"\n=== {a} ←→ {b} ===")
        # Items with the strongest match first
        for i in sim.max(axis=1).argsort()[::-1]:
            row = sim[i]
            print(f"{qa[i][:LINE_LENGTH_MAX]}")
            matches = [j for j in row.argsort()[::-1] if row[j] >= THRESHOLD]
            if not matches:
                print("  [#808080](no match above threshold)[/]")
            for j in matches:
                print(f"  {row[j]:.3f}  {qb[j][:LINE_LENGTH_MAX]}")
                qai = qa[i]
                qbj = qb[j]
                s = row[j]
                over_threshold.append((s, qai, qbj, i, j, a, b))

    # 4. Lastly, print every pair that crossed THRESHOLD
    print("\nOver threshold:")
    for sim, q1, q2, i, j, d1, d2 in sorted(
        over_threshold, key=lambda x: x[0], reverse=True
    ):
        print(f"{sim:.3f} {d1} #{i} / {d2} #{j}\n · {q1}\n · {q2}")
