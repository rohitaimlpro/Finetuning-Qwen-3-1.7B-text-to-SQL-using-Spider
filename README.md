# Fine-tuning Qwen3-1.7B for Text-to-SQL (Spider)

Fine-tunes Qwen3-1.7B with QLoRA on the [Spider](https://yale-lily.github.io/spider) Text-to-SQL benchmark, following a full evaluation-driven ML workflow: **baseline first, error analysis to find the actual weakness, fine-tune to target it, re-measure to confirm it worked** — rather than fine-tuning blind and hoping for the best.

## Results

Evaluated on the full Spider dev set (1,034 examples), scored with the official [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) (Execution Accuracy + Exact Match).

| Metric (all 1,034) | Baseline | Fine-tuned | Δ |
|---|---:|---:|---:|
| Execution Accuracy | 58.7% | **67.3%** | **+8.6 pts** |
| Exact Match | 30.5% | **59.0%** | **+28.5 pts** |

By difficulty (Execution Accuracy / Exact Match):

| Difficulty | Baseline | Fine-tuned |
|---|---:|---:|
| easy (n=248) | 81.0% / 64.5% | 87.1% / 82.7% |
| medium (n=446) | 62.6% / 24.7% | 71.7% / 62.8% |
| hard (n=174) | 47.1% / 17.8% | 54.6% / 45.4% |
| extra (n=166) | 27.1% / 8.4% | 39.2% / 27.7% |

### What baseline error analysis found — and whether fine-tuning fixed it

A heuristic failure-bucket classifier (`src/error_analysis.py`, no SQL execution needed) was run on the baseline before deciding what to fine-tune for. It found the single biggest failure mode was **incorrect table/join selection** — the model either joined tables it didn't need, or missed one it did:

| Failure bucket | Baseline | Fine-tuned |
|---|---:|---:|
| Wrong table/join set (combined) | 35.0% | **23.2%** |
| — hallucinated/extra tables | 21.6% | 13.4% |
| — missing a needed table | 4.3% | 6.9% |
| — missing a needed join | 9.1% | 2.9% |
| Exact string match | 14.2% | 40.2% |

The targeted weakness shrank by ~12 points after fine-tuning — confirmation the fine-tune improved the specific thing it was meant to, not just surface-level metrics. (One honest nuance: "missing a needed table" ticked up slightly even as "extra tables" dropped sharply — a modest shift in error mode, not a clean win on every sub-metric.)

## Approach

1. **Baseline evaluation** — run the untouched base model against Spider's dev set, measure Execution Accuracy / Exact Match / latency / throughput before assuming fine-tuning is even necessary.
2. **Error analysis** — bucket *why* it fails (wrong tables, missing joins, aggregation mismatches, etc.) rather than jumping straight to fine-tuning.
3. **QLoRA fine-tuning** — via [Unsloth](https://github.com/unslothai/unsloth), on Spider's training set, using the identical schema/prompt formatting as evaluation so training and inference are consistent.
4. **Re-evaluation** — same scripts, same metrics, against the fine-tuned model, for a directly comparable before/after.
5. **Deployment** *(planned, not yet built)* — serve with vLLM behind a FastAPI endpoint, containerized, benchmarked for throughput/latency/concurrency.

## Training setup

- **Base model**: Qwen3-1.7B (4-bit, via Unsloth)
- **Method**: QLoRA, rank 16, targeting all attention + MLP projection layers
- **Data**: Spider `train_spider.json` + `train_others.json` (8,659 examples)
- **Hardware**: single Colab T4 GPU (free tier)
- **3 epochs**, effective batch size 8, learning rate 2e-4, 3,249 steps, ~2h53m wall-clock

## Repo structure

```
src/
  schema_utils.py        Spider tables.json -> CREATE TABLE schema text
  prompts.py              schema-aware prompt building (shared by eval + training)
  sql_safety.py           SELECT-only safety gate for generated SQL
  generate_predictions.py generation + latency/throughput/GPU-memory logging
  run_eval.py             wraps the official test-suite-sql-eval scorer
  error_analysis.py       heuristic failure-bucket classifier
  train_qlora.py          Unsloth QLoRA fine-tuning, resumable across Colab disconnects
stage1_baseline_colab.ipynb   baseline evaluation, run end-to-end on Colab
stage3_finetune_colab.ipynb   fine-tuning + re-evaluation, run end-to-end on Colab
```

`data/`, `results/`, and `third_party/` are intentionally not committed (datasets and model weights don't belong in git) — the notebooks fetch/build them from scratch.

## Reproducing this

Both notebooks are meant to be run top-to-bottom on Colab (T4 GPU runtime):

1. Upload this repo's contents to a Google Drive folder.
2. Open `stage1_baseline_colab.ipynb` — it downloads Spider, clones the eval suite, and runs the baseline.
3. Open `stage3_finetune_colab.ipynb` — it fine-tunes with QLoRA (checkpointing to Drive every 50 steps, so it survives a Colab disconnect) and re-runs the same evaluation against the result.

`pip install -r requirements.txt` for evaluation-only use; add `requirements-train.txt` for fine-tuning (Unsloth/TRL/bitsandbytes, GPU-only).

## Status / roadmap

- [x] Stage 1 — baseline evaluation
- [x] Error analysis
- [x] Stage 3 — QLoRA fine-tuning + re-evaluation
- [ ] Stage 2 — vLLM + FastAPI deployment, containerization, throughput benchmarking
- [ ] Publish fine-tuned weights to Hugging Face Hub

## Acknowledgments

- [Spider](https://yale-lily.github.io/spider) dataset (Yu et al.)
- [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) for Execution Accuracy / Exact Match scoring
- [Unsloth](https://github.com/unslothai/unsloth) for QLoRA fine-tuning
