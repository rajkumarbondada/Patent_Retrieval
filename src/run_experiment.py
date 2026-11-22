"""
run_experiment.py
-----------------
Retrieval experiment (Chapter 3):

  BM25   : Okapi BM25, k1=1.2, b=0.75, lowercase + Porter stemming
  Dense  : sentence-transformers models (base and fine-tuned), indexed with
           FAISS IndexIVFFlat (nlist=256, nprobe=16, inner product on
           L2-normalised vectors == cosine similarity)
  Metrics: Precision@K, MAP@K, NDCG@K for K in {5, 10, 20}
  Timing : wall-clock time of BM25 indexing/search and of encoding + FAISS per model (timings.json)
  Tests  : paired Wilcoxon signed-rank tests on per-query scores
           (every dense model vs BM25, and every fine-tuned model vs its
           base model), with Holm correction for multiple comparisons

Models are given as "Label=path_or_hub_id". Comparisons between two models
are given as "Label A|Label B" (A is compared with B).

Step 1, create the patent-level split only:
    python run_experiment.py --corpus patents_G_H.jsonl --split-only --out results

Step 2, evaluate everything:
    python run_experiment.py --corpus patents_G_H.jsonl --out results \
        --models "MiniLM (base)=sentence-transformers/all-MiniLM-L6-v2" \
                 "MiniLM (fine-tuned)=models/minilm-gh" \
                 "PatentSBERTa (base)=AI-Growth-Lab/PatentSBERTa" \
                 "PatentSBERTa (fine-tuned)=models/patentsberta-gh" \
        --pairs "MiniLM (fine-tuned)|MiniLM (base)" \
                "PatentSBERTa (fine-tuned)|PatentSBERTa (base)"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from evaluation import paired_significance, per_query_scores, summarise

logger = logging.getLogger("experiment")
BM25_NAME = "BM25"
TIMINGS: dict[str, dict] = {}


@dataclass(frozen=True)
class Patent:
    patent_id: str
    title: str
    abstract: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


# ------------------------------------------------------------------ data
def load_corpus(path: Path) -> tuple[list[Patent], dict[str, list[str]], set[str]]:
    """Returns patents, relevance links per patent, and the ids flagged as queries."""
    patents, citations, query_ids = [], {}, set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            patents.append(Patent(r["patent_id"], r["title"], r["abstract"]))
            citations[r["patent_id"]] = r.get("citations", [])
            if r.get("is_query"):
                query_ids.add(r["patent_id"])
    return patents, citations, query_ids


def split_by_queries(ids: list[str], citations: dict[str, list[str]], query_ids: set[str],
                     test_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Patent-level split of the query families. A test query and every family
    linked to it are removed from training, so no relevance link leaks into
    fine-tuning."""
    rng = random.Random(seed)
    qs = sorted(query_ids)
    rng.shuffle(qs)
    test = set(qs[:max(1, int(len(qs) * test_fraction))])
    held_out = set(test)
    for q in test:
        held_out.update(citations.get(q, []))
    return set(ids) - held_out, test


def build_qrels(test_ids: set[str], citations: dict[str, list[str]],
                corpus_ids: set[str]) -> dict[str, set[str]]:
    qrels = {}
    for pid in test_ids:
        rel = {c for c in citations.get(pid, []) if c in corpus_ids and c != pid}
        if rel:
            qrels[pid] = rel
    return qrels


# ------------------------------------------------------------ BM25
_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = None


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


class Bm25Index:
    """Okapi BM25 via sparse matrices:
    score(q,d) = sum_t idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl))."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer

        self._cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self._cv.fit_transform(texts).tocoo().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.tocsr().sum(axis=1)).ravel()
        avgdl = float(dl.mean())
        df = np.asarray((tf.tocsr() > 0).sum(axis=0)).ravel()
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        denom = tf.data + k1 * (1.0 - b + b * dl[tf.row] / avgdl)
        data = tf.data * (k1 + 1.0) / denom * idf[tf.col]
        self._w = sparse.csr_matrix((data, (tf.row, tf.col)), shape=tf.shape).T.tocsr()

    def scores(self, query_texts: list[str]) -> np.ndarray:
        q = self._cv.transform(query_texts).astype(np.float32)
        return (q @ self._w).toarray()


def top_ranked(row: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, len(row) - 1)
    part = np.argpartition(-row, top_k)[:top_k]
    return part[np.argsort(-row[part])]


def bm25_run(texts: list[str], ids: list[str], query_idx: list[int],
             top_k: int, batch: int = 256) -> dict[str, list[str]]:
    t0 = time.perf_counter()
    index = Bm25Index(texts)
    TIMINGS["BM25"] = {"index_build_s": round(time.perf_counter() - t0, 2)}
    t1 = time.perf_counter()
    run: dict[str, list[str]] = {}
    for s in range(0, len(query_idx), batch):
        chunk = query_idx[s:s + batch]
        sc = index.scores([texts[i] for i in chunk])
        for row, qi in zip(sc, chunk):
            row[qi] = -np.inf  # a patent is never its own answer
            run[ids[qi]] = [ids[j] for j in top_ranked(row, top_k)]
    TIMINGS["BM25"]["search_s"] = round(time.perf_counter() - t1, 2)
    TIMINGS["BM25"]["n_queries"] = len(query_idx)
    return run


# ------------------------------------------------------------ dense + FAISS
def encode(model_path: str, texts: list[str], batch_size: int = 64, label: str = "") -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info("Encoding with %s", model_path)
    model = SentenceTransformer(model_path)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True).astype("float32")
    secs = time.perf_counter() - t0
    TIMINGS[label or model_path] = {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "dimension": int(emb.shape[1]), "max_seq_length": int(model.max_seq_length),
        "n_docs": len(texts), "encode_s": round(secs, 2), "docs_per_s": round(len(texts) / secs, 1),
        "ms_per_doc": round(1000 * secs / len(texts), 2),
        "device": "cuda" if torch.cuda.is_available() else "cpu"}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


def faiss_run(emb: np.ndarray, ids: list[str], query_idx: list[int], top_k: int,
              kind: str = "ivf", nlist: int = 256, nprobe: int = 16, label: str = "") -> dict[str, list[str]]:
    import faiss

    t0 = time.perf_counter()
    x = np.ascontiguousarray(emb, dtype="float32").copy()
    faiss.normalize_L2(x)  # inner product on unit vectors == cosine similarity
    d = x.shape[1]
    if kind == "exact":
        index = faiss.IndexFlatIP(d)
    else:
        nlist = max(1, min(nlist, len(x) // 39))  # FAISS needs >=39 points per centroid
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
        index.nprobe = min(nprobe, nlist)
    index.add(x)
    logger.info("FAISS %s index: %d vectors, dim=%d", kind, len(x), d)
    _, nbrs = index.search(x[query_idx], top_k + 1)
    if label:
        TIMINGS.setdefault(label, {})["faiss_build_search_s"] = round(time.perf_counter() - t0, 2)
    return {ids[qi]: [ids[j] for j in row if j != -1 and j != qi][:top_k]
            for qi, row in zip(query_idx, nbrs)}


# ------------------------------------------------------------ statistics
def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are left as NaN)."""
    idx = [i for i, p in enumerate(p_values) if p == p]
    order = sorted(idx, key=lambda i: p_values[i])
    m, running, adj = len(order), 0.0, [float("nan")] * len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adj[i] = running
    return adj


def parse_models(specs: list[str]) -> dict[str, str]:
    models = {}
    for s in specs:
        label, sep, path = s.partition("=")
        if not sep:
            label, path = s, s
        models[label.strip()] = path.strip()
    return models


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=[], help='"Label=path_or_hub_id"')
    ap.add_argument("--pairs", nargs="*", default=[], help='"Label A|Label B" extra comparisons')
    ap.add_argument("--split-only", action="store_true", help="write train/test ids and stop")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--index", choices=["ivf", "exact"], default="ivf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    patents, citations, query_ids = load_corpus(args.corpus)
    if not query_ids:
        raise SystemExit("The corpus has no records flagged is_query: rebuild it with prepare_dapfam.py.")
    ids = [p.patent_id for p in patents]
    train_ids, test_ids = split_by_queries(ids, citations, query_ids, args.test_fraction, args.seed)
    (args.out / "train_ids.txt").write_text("\n".join(sorted(train_ids)))
    (args.out / "test_ids.txt").write_text("\n".join(sorted(test_ids)))
    qrels = build_qrels(test_ids, citations, set(ids))
    query_idx = [i for i, pid in enumerate(ids) if pid in qrels]
    split_stats = {"corpus": len(patents), "train_patents": len(train_ids),
                   "test_queries": len(test_ids), "test_queries_with_relevant": len(qrels),
                   "relevant_links_in_test": sum(len(v) for v in qrels.values())}
    (args.out / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
    logger.info("Split: %s", split_stats)
    if not qrels:
        raise SystemExit("No test query has an in-corpus relevant patent.")
    if args.split_only:
        return

    texts = [p.text for p in patents]
    depth = max(args.ks)
    runs = {BM25_NAME: bm25_run(texts, ids, query_idx, depth)}
    for label, path in parse_models(args.models).items():
        runs[label] = faiss_run(encode(path, texts, label=label), ids, query_idx, depth, kind=args.index, label=label)
    (args.out / "timings.json").write_text(json.dumps(TIMINGS, indent=2))

    comparisons = [(n, BM25_NAME) for n in runs if n != BM25_NAME]
    for spec in args.pairs:
        a, _, b = (s.strip() for s in spec.partition("|"))
        if a in runs and b in runs:
            comparisons.append((a, b))
        else:
            logger.warning("Skipping comparison %r: unknown label", spec)

    rows, sig_rows, pq_rows = [], [], []
    for k in args.ks:
        per_system = {n: per_query_scores(r, qrels, k) for n, r in runs.items()}
        for n, pq in per_system.items():
            for qid, sc in pq.items():
                pq_rows.append({"query_id": qid, "n_relevant": len(qrels[qid]), "system": n, "K": k,
                                "P@K": round(sc[f"P@{k}"], 6), "AP@K": round(sc["AP"], 6),
                                "NDCG@K": round(sc[f"NDCG@{k}"], 6)})
        for n, pq in per_system.items():
            s = summarise(pq)
            rows.append({"system": n, "K": k, "P@K": round(s[f"P@{k}"], 4),
                         "MAP@K": round(s["MAP"], 4), "NDCG@K": round(s[f"NDCG@{k}"], 4),
                         "n_queries": s["n_queries"]})
            logger.info("%-28s K=%-2d P@K=%.4f MAP@K=%.4f NDCG@K=%.4f", n, k,
                        s[f"P@{k}"], s["MAP"], s[f"NDCG@{k}"])
        for a, b in comparisons:
            for metric, label in ((f"P@{k}", "P@K"), ("AP", "MAP@K"), (f"NDCG@{k}", "NDCG@K")):
                r = paired_significance(per_system[a], per_system[b], metric)
                sig_rows.append({"system": a, "compared_with": b, "K": k, "metric": label, **r})

    for r, p in zip(sig_rows, holm([r["p_value"] for r in sig_rows])):
        r["p_holm"] = p
        r["significant_0.05"] = bool(p == p and p < 0.05)

    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (args.out / "significance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sig_rows[0]))
        w.writeheader()
        w.writerows(sig_rows)
    with (args.out / "per_query_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pq_rows[0]))
        w.writeheader()
        w.writerows(pq_rows)
    logger.info("Wrote results.csv, significance.csv and per_query_scores.csv to %s", args.out)


if __name__ == "__main__":
    main()
