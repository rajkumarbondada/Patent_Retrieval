"""
prepare_dapfam.py
-----------------
Fetches the public DAPFAM benchmark (Ayaou, Cavallucci and Chibane, 2025) from
the Hugging Face Hub ENTIRELY ONLINE (no login, nothing to download by hand) and
writes the corpus file used by run_experiment.py / finetune.py.

DAPFAM (datalyes/DAPFAM_patent): family-level English patents with title,
abstract, claims, description, IPC codes; a query set; and a relation table with
citation-based relevance scores (1 = cited by / citing the query, 0 = random
negative) plus IN/OUT-domain labels.

What this script does
  1. discovers the three tables (corpus / queries / relations) on the Hub;
  2. keeps only QUERY families with at least one IPC code in Section G or H;
  3. keeps every target family linked to those queries (positives and
     negatives), so every relevance label stays resolvable;
  4. extracts the first claim as the "independent claim" (heuristic: text up to
     the start of claim 2) and writes one JSON object per patent.

Output record:
  patent_id, title, abstract, independent_claim, cpc_codes (IPC 3-char classes),
  citations (ids of positively linked families, only for queries), is_query

Usage:
    python prepare_dapfam.py --out patents_G_H.jsonl [--max-queries 0] [--sections GH]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

REPO = "datalyes/DAPFAM_patent"

# Column names from the DAPFAM paper (Tables 4 and 6) plus BEIR/MTEB-style aliases.
# Hyphens are normalised to underscores before matching.
Q_ID_COLS = ["query_id", "_id", "id", "qid", "family_id", "patent_id"]
T_ID_COLS = ["relevant_id", "_id", "id", "corpus_id", "doc_id", "family_id", "patent_id"]
REL_Q_COLS = ["query_id", "qid"]
REL_T_COLS = ["relevant_id", "corpus_id", "doc_id", "docid"]
TITLE_COLS = ["title_en", "title"]
ABSTRACT_COLS = ["abstract_en", "abstract", "text"]
CLAIMS_COLS = ["claims_text", "claims"]
IPC3_COLS = ["classifications_ipcr_list_first_three_chars_list", "ipc3", "ipc3_list"]
IPC_STR_COLS = ["ipcr_codes_str", "ipcr_codes", "ipc_codes"]
REL_COLS = ["relevance_score", "relevance", "score"]
KNOWN_COLS = set(Q_ID_COLS + T_ID_COLS + REL_Q_COLS + REL_T_COLS + TITLE_COLS + ABSTRACT_COLS + CLAIMS_COLS
                 + IPC3_COLS + IPC_STR_COLS + REL_COLS + ["domain_rel"])


def pick(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def clean_text(text) -> str:
    """Text normalisation (Chapter 3.2.2): Unicode NFKC, remove LaTeX-style math markup,
    control / replacement characters and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]{1,200}\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"[\u0000-\u0008\u000b-\u001f\u007f\ufffd\u200b-\u200f\ufeff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_claim(text: str, max_chars: int = 3000) -> str:
    """First claim = text before the start of claim 2 (heuristic, documented in Chapter 3)."""
    text = clean_text(text)
    if not text:
        return ""
    m = re.search(r"(?<![\d.])\b2\s*[.)]\s+(?=[A-Z])", text[3:])
    cut = (m.start() + 3) if m else len(text)
    return text[:min(cut, max_chars)].strip()


def ipc3_codes(row: dict, ipc3_col, ipc_str_col) -> list[str]:
    """IPC classes (3 chars, e.g. 'G06', 'H04') of a family."""
    raw = row.get(ipc3_col) if ipc3_col else None
    if isinstance(raw, (list, tuple)) or hasattr(raw, "tolist"):
        return sorted({str(c).strip()[:3] for c in list(raw) if str(c).strip()})
    if isinstance(raw, str) and raw.strip():
        return sorted({c[:3] for c in re.findall(r"[A-H]\d{2}", raw)})
    s = row.get(ipc_str_col) if ipc_str_col else None
    if isinstance(s, str):
        return sorted({c[:3] for c in re.findall(r"\b[A-H]\d{2}[A-Z]", s)} | {c[:3] for c in re.findall(r"\b[A-H]\d{2}\b", s)})
    return []


def build_corpus(queries, targets, relations, sections="GH", max_queries=0, seed=42):
    """Inputs are lists of dicts (queries, targets, relations).

    Returns (records, stats)."""
    q_id = pick(queries[0].keys(), Q_ID_COLS) if queries else None
    t_id = pick(targets[0].keys(), T_ID_COLS) if targets else None
    r_q = pick(relations[0].keys(), REL_Q_COLS) if relations else None
    r_t = pick(relations[0].keys(), REL_T_COLS) if relations else None
    r_s = pick(relations[0].keys(), REL_COLS) if relations else None
    if not (q_id and t_id and r_q and r_t and r_s):
        raise ValueError(f"Cannot find id/relevance columns. queries={list(queries[0]) if queries else None}, "
                         f"targets={list(targets[0]) if targets else None}, relations={list(relations[0]) if relations else None}")

    def fields(row, cols_title, cols_abs, cols_clm):
        title = row.get(pick(row.keys(), cols_title) or "", "") or ""
        abstract = row.get(pick(row.keys(), cols_abs) or "", "") or ""
        claims = row.get(pick(row.keys(), cols_clm) or "", "") or ""
        return clean_text(title), clean_text(abstract), first_claim(claims)

    def ipc_of(row):
        return ipc3_codes(row, pick(row.keys(), IPC3_COLS), pick(row.keys(), IPC_STR_COLS))

    kept_q = {}
    for row in queries:
        codes = ipc_of(row)
        if any(c[:1] in sections for c in codes):
            kept_q[str(row[q_id])] = (row, codes)
    ids = sorted(kept_q)
    if max_queries and len(ids) > max_queries:
        random.Random(seed).shuffle(ids)
        ids = sorted(ids[:max_queries])
    kept_q = {i: kept_q[i] for i in ids}

    positives: dict[str, list[str]] = {i: [] for i in kept_q}
    linked: set[str] = set()
    for rel in relations:
        qid, tid = str(rel[r_q]), str(rel[r_t])
        if qid in kept_q:
            linked.add(tid)
            try:
                positive = float(rel[r_s]) > 0
            except (TypeError, ValueError):
                positive = False
            if positive and tid != qid:
                positives[qid].append(tid)

    target_rows = {str(r[t_id]): r for r in targets}
    records: dict[str, dict] = {}
    for tid in sorted(linked):
        row = target_rows.get(tid)
        if row is None:
            continue
        title, abstract, claim = fields(row, TITLE_COLS, ABSTRACT_COLS, CLAIMS_COLS)
        records[tid] = {"patent_id": tid, "title": title, "abstract": abstract, "independent_claim": claim,
                        "cpc_codes": ipc_of(row), "citations": [], "is_query": False}
    for qid, (row, codes) in kept_q.items():
        title, abstract, claim = fields(row, TITLE_COLS, ABSTRACT_COLS, CLAIMS_COLS)
        rec = records.get(qid, {})
        rec.update({"patent_id": qid, "title": title, "abstract": abstract, "independent_claim": claim,
                    "cpc_codes": codes or rec.get("cpc_codes", []), "is_query": True})
        records[qid] = rec
    for qid, pos in positives.items():
        records[qid]["citations"] = sorted({p for p in pos if p in records})

    out = [r for r in records.values() if r["title"] or r["abstract"]]
    ids_out = {r["patent_id"] for r in out}
    for r in out:
        r["citations"] = [c for c in r["citations"] if c in ids_out]
    nq = sum(r["is_query"] for r in out)
    stats = {"patents": len(out), "queries": nq,
             "queries_with_positive": sum(1 for r in out if r["is_query"] and r["citations"]),
             "positive_links": sum(len(r["citations"]) for r in out)}
    return out, stats


def _rows(dsdict, keep_all=False):
    """All splits of a dataset concatenated, hyphens in column names turned into underscores.
    The huge description column is never loaded unless keep_all is set."""
    out = []
    for split in dsdict.keys():
        ds = dsdict[split]
        wanted = [c for c in ds.column_names if keep_all or c.replace("-", "_") in KNOWN_COLS]
        for row in ds.select_columns(wanted).to_list():
            out.append({k.replace("-", "_"): v for k, v in row.items()})
    return out


def load_tables(repo=REPO, inspect=False):
    """Download (online, cached by Hugging Face) and return queries, targets, relations as lists of dicts."""
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(repo)
    print("Configs on the Hub:", configs)

    def find(*keys, exclude=()):
        for c in configs:
            if c not in exclude and any(k in c.lower() for k in keys):
                return c
        return None

    cfg = {"queries": find("quer"), "targets": find("corpus", "target", "document")}
    cfg["relations"] = find("relation", "eval", "qrel", "relev") or find("default") \
        or next((c for c in configs if c not in cfg.values()), None)
    if not all(cfg.values()):
        raise SystemExit(f"Could not map the dataset configs {configs} to queries/targets/relations: {cfg}. "
                         f"Open https://huggingface.co/datasets/{repo} and edit the keywords in load_tables().")
    print("Config mapping:", cfg)
    tables = {}
    for name, c in cfg.items():
        dsdict = load_dataset(repo, c)
        rows = _rows(dsdict)
        print(f"  {name}: config={c!r}, splits={list(dsdict.keys())}, rows={len(rows)}")
        if rows:
            print("     columns used:", list(rows[0].keys()))
            if inspect:
                print("     first row :", {k: (str(v)[:70] + "...") if len(str(v)) > 70 else v for k, v in rows[0].items()})
        tables[name] = rows
    return tables["queries"], tables["targets"], tables["relations"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("patents_G_H.jsonl"))
    ap.add_argument("--sections", default="GH", help="IPC sections to keep for queries (default GH)")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all matching queries")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true", help="only print configs, columns and one sample row")
    args = ap.parse_args()

    queries, targets, relations = load_tables(inspect=args.inspect)
    if args.inspect:
        return
    records, stats = build_corpus(queries, targets, relations, args.sections, args.max_queries, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with_claim = sum(1 for r in records if r["independent_claim"])
    print("Wrote", args.out, stats, f"| with a claim: {with_claim}/{len(records)}")
    if with_claim == 0:
        print("[warn] no claims found: fine-tuning will use citation pairs only (check CLAIMS_COLS).")
    if stats["queries_with_positive"] == 0:
        raise SystemExit("No query has a positive relevance link: check the column mapping above.")


if __name__ == "__main__":
    main()
