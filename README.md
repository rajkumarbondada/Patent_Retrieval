# Semantic Patent Retrieval: BM25 vs Base vs Fine-tuned Transformers

MSc in Artificial Intelligence, thesis project (programme MSCAIJAN26I).

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
- **Fine-tuning:** Multiple Negatives Ranking Loss (scale 20) on (title + abstract → first claim) pairs and citation pairs from the training split. 1 epoch, batch size 32, learning rate 2e-5, 10% warm-up, max length 256, mixed precision.
- **Search:** retrieval quality is scored with **exact** FAISS search (`IndexFlatIP`, cosine similarity on L2-normalised vectors), so the scores depend only on the embeddings. The approximate `IndexIVFFlat` index (256 cells, 16 searched per query) is evaluated separately as the scalable search component, by its recall of the exact top-K.
- **Evaluation:** Precision@K, MAP@K and NDCG@K for K = 5, 10, 20. Paired Wilcoxon signed-rank tests with Holm correction (72 tests).
- **Supplementary experiment:** both encoders fine-tuned for 3 epochs instead of 1, with everything else unchanged, evaluated as a separate experiment with its own Holm correction.

## Results (219 test queries, K = 10)

| System | P@10 | MAP@10 | NDCG@10 |
|---|---|---|---|
| BM25 | 0.1603 | 0.0506 | 0.1791 |
| MiniLM (base) | 0.1890 | 0.0622 | 0.2090 |
| **MiniLM (fine-tuned)** | **0.2005** | **0.0685** | **0.2265** |
| PatentSBERTa (base) | 0.1680 | 0.0549 | 0.1912 |
| PatentSBERTa (fine-tuned) | 0.1872 | 0.0625 | 0.2094 |

- Fine-tuned MiniLM is the best system on every metric and K (NDCG@10 26% above BM25) and the only one significantly better than BM25 at every K after Holm correction.
- Fine-tuning improves both transformers on every metric and K (relative gains of 4–19%), significantly at K = 20.
- Base PatentSBERTa is not significantly better than BM25, and PatentSBERTa stays below MiniLM after fine-tuning.
- Training for 3 epochs instead of 1 does not improve either model (no change significant, even before correction).
- The IVF index returns 92–98% of the exact top-20 results.

Results for all values of K, the significance tests and the timings are written to `results_final/` (and `results_3ep/`) by the notebook.

## Reproducibility

Re-running the notebook gives the same numbers:

- **Locked library versions:** the first run saves the exact versions to `requirements_lock.txt` on Google Drive, and later runs install those versions. The notebook warns if Colab's torch/numpy version or the GPU type differs from the locked run.
- **Deterministic fine-tuning:** all random number generators are seeded (seed 42), the training batches are shuffled with a seeded generator, and deterministic GPU kernels are used. Each model's `training_info.json` records the library versions and GPU.
- **Exact-search scoring:** scores do not depend on the IVF clustering, which can change when the embeddings change slightly.
- **Saved models are reused:** fine-tuning runs only if the model folder is missing, so do not delete `models/` on Drive unless you want to retrain.

Retraining on a different GPU type or library version can still change the fine-tuned models slightly.

## Repository layout

```
Patent_Pipeline_Colab.ipynb   full pipeline (run in Google Colab)
src/prepare_dapfam.py         downloads DAPFAM and builds the G/H corpus
src/evaluation.py             P@K, AP/MAP, NDCG@K and Wilcoxon tests
src/finetune.py               fine-tunes a sentence-transformers model (deterministic)
src/run_experiment.py         train/test split, BM25, dense retrieval (exact FAISS + IVF recall check), metrics, tests
requirements.txt
```

The scripts in `src/` are the same code that the notebook writes to disk with `%%writefile`.

## How to run

The easiest way is to open `Patent_Pipeline_Colab.ipynb` in Google Colab with a **T4 GPU** runtime and run all cells. Everything is saved to `MyDrive/patent_thesis`, including `results_final.zip` and `results_3ep.zip`. The 3-epoch section at the end is optional and takes the longest.

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
python run_experiment.py --corpus patents_G_H.jsonl --out results --index exact --ivf-check \
    --models "MiniLM (base)=sentence-transformers/all-MiniLM-L6-v2" \
             "MiniLM (fine-tuned)=models/minilm-gh" \
             "PatentSBERTa (base)=AI-Growth-Lab/PatentSBERTa" \
             "PatentSBERTa (fine-tuned)=models/patentsberta-gh" \
    --pairs "MiniLM (fine-tuned)|MiniLM (base)" \
            "PatentSBERTa (fine-tuned)|PatentSBERTa (base)" \
            "PatentSBERTa (base)|MiniLM (base)" \
            "PatentSBERTa (fine-tuned)|MiniLM (fine-tuned)"
```

The data is downloaded from the Hugging Face Hub, so you don't need a login or any manual download.

## Data and model credits

- DAPFAM benchmark: Ayaou, Cavallucci and Chibane (2025), `datalyes/DAPFAM_patent` on Hugging Face.
- `sentence-transformers/all-MiniLM-L6-v2` and `AI-Growth-Lab/PatentSBERTa` from the Hugging Face Hub.
