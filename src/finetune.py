"""
finetune.py
-----------
Domain adaptation (Chapter 3). Fine-tunes a sentence-transformers model
(all-MiniLM-L6-v2 or PatentSBERTa) on (anchor, positive) pairs built from the
TRAINING split of the CPC/IPC G and H corpus, with in-batch negatives:

    anchor   = title + abstract
    positive = first (independent) claim of the same patent, and, with
               --use-citations, the title + abstract of a linked training patent
    negative = the positives of the other pairs in the batch (implicit)

Loss: MultipleNegativesRankingLoss (scale 20, i.e. temperature 0.05).
Optimiser: AdamW, lr 2e-5, linear warm-up over the first 10% of steps.

Usage:
    python finetune.py --corpus patents_G_H.jsonl --train-ids results/train_ids.txt \
        --base-model sentence-transformers/all-MiniLM-L6-v2 --out models/minilm-gh --use-citations
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger("finetune")


def load_pairs(corpus_path: Path, allowed_ids: set[str],
               use_citations: bool = False) -> list[tuple[str, str]]:
    """(anchor, positive) pairs from TRAINING patents only. A citation pair is
    used only when BOTH patents are in the training split, so no test query or
    test relevance link ever reaches fine-tuning."""
    records = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["patent_id"] in allowed_ids:
                records.append(rec)
    summary = {r["patent_id"]: f"{r['title']} {r['abstract']}".strip() for r in records}
    pairs = []
    for rec in records:
        anchor = summary[rec["patent_id"]]
        claim = (rec.get("independent_claim") or "").strip()
        if anchor and claim:
            pairs.append((anchor, claim))
        if use_citations:
            for cited in rec.get("citations", []):
                if cited in summary and cited != rec["patent_id"] and summary[cited]:
                    pairs.append((anchor, summary[cited]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--train-ids", type=Path, required=True,
                    help="text file, one patent_id per line (training split only)")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--use-citations", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ids = set(args.train_ids.read_text().split())
    pairs = load_pairs(args.corpus, train_ids, args.use_citations)
    logger.info("Base model: %s | training pairs: %d", args.base_model, len(pairs))
    if len(pairs) < args.batch_size:
        raise SystemExit("Not enough training pairs for the chosen batch size.")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length
    examples = [InputExample(texts=[a, p]) for a, p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        optimizer_params={"lr": args.lr},
        use_amp=torch.cuda.is_available(),
        show_progress_bar=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    (args.out / "training_info.json").write_text(json.dumps(
        {"base_model": args.base_model, "pairs": len(pairs), "epochs": args.epochs,
         "batch_size": args.batch_size, "lr": args.lr, "max_seq_length": args.max_seq_length,
         "use_citations": args.use_citations, "seed": args.seed}, indent=2))
    logger.info("Saved fine-tuned model to %s", args.out)


if __name__ == "__main__":
    main()
