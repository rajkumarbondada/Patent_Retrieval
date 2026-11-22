# Semantic Patent Retrieval: BM25 vs Base vs Fine-tuned Transformers

MSc in Artificial Intelligence, thesis project (student X24244732, programme MSCAIJAN26I).

This project compares five retrieval systems for prior-art search on the
[DAPFAM](https://huggingface.co/datasets/datalyes/DAPFAM_patent) patent benchmark,
restricted to query families in IPC sections **G** (Physics) and **H** (Electricity).

| System | Type |
|---|---|
| BM25 (k1 = 1.2, b = 0.75, Porter stemming) | lexical baseline |
| all-MiniLM-L6-v2 (base) | general-purpose sentence transformer |
| all-MiniLM-L6-v2 (fine-tuned) | general model, domain-adapted |
| PatentSBERTa (base) | patent-specific sentence transformer |
| PatentSBERTa (fine-tuned) | patent model, domain-adapted |

## Method

- **Data:** 27,853 patent families (732 G/H query families plus every target family linked to them). Relevance comes from DAPFAM citation links.
- **Split:** 30% of the query families (219 queries) are held out for testing. Every patent linked to a test query is removed from training, so no test relevance link reaches fine-tuning.
- **Fine-tuning:** Multiple Negatives Ranking Loss (scale 20) on (title + abstract → first claim) pairs and citation pairs from the training split. 1 epoch, batch size 32, learning rate 2e-5, 10% warm-up.
- **Search:** dense models use a FAISS `IndexIVFFlat` (cosine similarity on L2-normalised vectors).
- **Evaluation:** Precision@K, MAP@K and NDCG@K for K = 5, 10, 20. Paired Wilcoxon signed-rank tests with Holm correction.

## Results (219 test queries, K = 10)

| System | P@10 | MAP@10 | NDCG@10 |
|---|---|---|---|
| BM25 | 0.1603 | 0.0506 | 0.1791 |
| MiniLM (base) | 0.1836 | 0.0601 | 0.2031 |
| **MiniLM (fine-tuned)** | **0.1968** | **0.0672** | **0.2233** |
| PatentSBERTa (base) | 0.1580 | 0.0515 | 0.1801 |
| PatentSBERTa (fine-tuned) | 0.1822 | 0.0605 | 0.2040 |

Fine-tuning improves both transformers on every metric and K (relative gains of 6–23%).
Results for all values of K are in the notebook.

## Repository layout

```
Patent_Pipeline_Colab.ipynb   full pipeline with outputs (run in Google Colab)
src/prepare_dapfam.py         downloads DAPFAM and builds the G/H corpus
src/evaluation.py             P@K, AP/MAP, NDCG@K and Wilcoxon tests
src/finetune.py               fine-tunes a sentence-transformers model
src/run_experiment.py         train/test split, BM25, dense + FAISS retrieval, metrics
requirements.txt
```

## How to run

The easiest way is to open `Patent_Pipeline_Colab.ipynb` in Google Colab with a GPU runtime and run all cells.

To run locally instead (a GPU is recommended for fine-tuning):

```bash
pip install -r requirements.txt
cd src
python prepare_dapfam.py --out patents_G_H.jsonl
python run_experiment.py --corpus patents_G_H.jsonl --split-only --out results
python finetune.py --corpus patents_G_H.jsonl --train-ids results/train_ids.txt \
    --base-model sentence-transformers/all-MiniLM-L6-v2 --out models/minilm-gh --use-citations
python finetune.py --corpus patents_G_H.jsonl --train-ids results/train_ids.txt \
    --base-model AI-Growth-Lab/PatentSBERTa --out models/patentsberta-gh --use-citations
python run_experiment.py --corpus patents_G_H.jsonl --out results \
    --models "MiniLM (base)=sentence-transformers/all-MiniLM-L6-v2" \
             "MiniLM (fine-tuned)=models/minilm-gh" \
             "PatentSBERTa (base)=AI-Growth-Lab/PatentSBERTa" \
             "PatentSBERTa (fine-tuned)=models/patentsberta-gh" \
    --pairs "MiniLM (fine-tuned)|MiniLM (base)" \
            "PatentSBERTa (fine-tuned)|PatentSBERTa (base)"
```

The data is downloaded from the Hugging Face Hub, so you don't need a login or any manual download.

## Data and model credits

- DAPFAM benchmark: Ayaou, Cavallucci and Chibane (2025), `datalyes/DAPFAM_patent` on Hugging Face.
- `sentence-transformers/all-MiniLM-L6-v2` and `AI-Growth-Lab/PatentSBERTa` from the Hugging Face Hub.
