# Model Card: Qwen3-1.7B Text-to-SQL (QLoRA fine-tune on Spider)

Follows the general structure of [Mitchell et al., 2019 — Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993), the convention used by Hugging Face and Google model cards.

## Model Details

- **Base model**: Qwen3-1.7B (Alibaba/Qwen)
- **Fine-tuning method**: QLoRA — rank 16, targeting all attention and MLP projection layers, via [Unsloth](https://github.com/unslothai/unsloth)
- **Task**: Text-to-SQL — natural-language question + database schema → SQLite `SELECT` query
- **Training compute**: 1× Colab T4 GPU, 3 epochs, ~2h53m
- **License**: inherits Qwen3's license terms; this repo's own code is MIT-licensed (see `LICENSE`)

## Intended Use

- **In scope**: generating read-only `SELECT` queries from natural-language questions against a known, provided database schema — a developer tool or an internal analyst-assistant, not a customer-facing autonomous agent.
- **Out of scope**: unsupervised execution against production databases with sensitive data, autonomous multi-step database administration, or any use where a hallucinated or subtly-wrong query could cause real harm without human review. This model has **not** been evaluated for those settings.

## Training Data & Procedure

Fine-tuned on [Spider](https://yale-lily.github.io/spider) (`train_spider.json` + `train_others.json`, 8,659 examples) — a cross-domain, human-annotated Text-to-SQL benchmark spanning 166 databases. Training and evaluation prompts are built identically (`src/prompts.py` + `src/schema_utils.py`), so there is no train/inference formatting mismatch.

## Evaluation Results

Full Spider dev set (1,034 examples, 20 databases **disjoint from training** — Spider is deliberately split this way to test generalization to unseen schemas), scored with the official [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval).

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Execution Accuracy | 58.7% | **67.3%** |
| Exact Match | 30.5% | **59.0%** |

Error analysis (`src/error_analysis.py`) found the baseline's dominant failure mode was incorrect table/join selection (35.0% of examples); fine-tuning reduced this to 23.2%. Full breakdown in `README.md`. **This 23.2% residual is the model's main known weakness** — it will sometimes still join tables it doesn't need or omit one it does, especially on multi-table "extra"-difficulty questions (39.2% execution accuracy vs. 87.1% on single-table "easy" questions).

## Risks & Mitigations

**SQL injection / stacked-query risk**: an LLM generating SQL that gets executed is a real attack surface — a compromised or confused generation could emit a destructive statement (`DROP`, `DELETE`, stacked queries via `;`). Mitigated by `src/sql_safety.py`: a token-level gate (via `sqlparse`, not naive string matching) that only allows exactly one `SELECT` statement through, walking actual SQL keyword tokens rather than searching raw text so legitimate queries containing words like "drop" in a string literal or column name aren't false-flagged.

**Red-teamed, not just asserted** — `src/redteam_sql_safety.py` runs 25 adversarial + legitimate cases against the gate directly. All 25 behaved as expected, including some genuinely tricky ones:

| Attack attempted | Result |
|---|---|
| Stacked query (`SELECT 1; DROP TABLE x`) | Rejected — split into 2 statements |
| Case-obfuscated stacked query | Rejected |
| CTE-prefixed DML smuggling (`WITH t AS (...) DELETE FROM x`) | Rejected — `sqlparse` correctly classifies the statement type as `DELETE` even with a `WITH` prefix, not fooled by the CTE prefix |
| Keyword split by a nested comment (`DR/**/OP TABLE x`) | Rejected — though *for a different reason than intended*: it fails to parse as recognizable SQL at all (`get_type()` returns `UNKNOWN`), and the gate fails closed on anything that isn't affirmatively `SELECT`. Worth being precise about: this wasn't "detected," it was caught by the gate's default-deny design. |
| Forbidden word inside a string literal (`WHERE message = 'DROP TABLE attempt'`) | Correctly **accepted** (no false positive) |
| Semicolon inside a string literal | Correctly **accepted** (not mistaken for a statement separator) |

**Known limitation, not fixed by this gate**: a `UNION`-based query pulling from a table outside the intended schema (e.g. `SELECT name FROM students UNION SELECT password FROM users`) is still a syntactically valid single `SELECT` and passes the gate — because the gate's scope is *"block destructive statements,"* not *"restrict which tables/schemas a query may touch."* In a real deployment against a sensitive multi-tenant database, this gate would need to be paired with application-layer authorization (e.g., a fixed, per-tenant allowlist of tables) — it is not a complete access-control solution on its own.

**Scope limitation of the red-team suite itself**: the above tests the safety *gate's* logic against hand-crafted SQL strings. It does not test whether adversarial *natural-language phrasing* in the input question can induce the live model to want to emit harmful SQL before the gate even runs — that requires running the actual model (Colab/GPU) and is a documented follow-up, not covered here.

**Hallucination risk**: the residual 23.2% wrong-table/join rate (above) is this model's primary hallucination-adjacent failure mode. It is measured and disclosed, not hidden — see the error-analysis breakdown in `README.md`.

## Explainability & Limitations

Classic tabular-model explainability tools (**SHAP**, **LIME**, feature importance) are built around models with fixed, structured input features and a single scalar/class output — they don't map cleanly onto an autoregressive LLM generating a variable-length token sequence, and applying them here isn't standard or well-established practice. **Attention weights are also not a valid substitute**: a substantial body of NLP interpretability research (e.g., Jain & Wallace, 2019, *"Attention is not Explanation"*) has shown attention scores don't reliably correspond to the actual causal reasons behind a model's output.

What this project uses instead: **behavioral/error-based analysis** (`src/error_analysis.py`) — bucketing *what kind* of mistake the model makes (wrong table, missing join, wrong aggregation) across a large evaluation set, rather than trying to explain any single prediction internally. This is a pragmatic, defensible stand-in for interpretability at the scale this project operates, not a claim of true mechanistic explainability.

## Governance & Regulatory Alignment

This is a portfolio-scale project, not a certified compliance artifact — the following is an honest note on which practices here parallel established frameworks, not a compliance claim.

- **NIST AI Risk Management Framework**: this model card + the evaluation pipeline (`generate_predictions.py`, `run_eval.py`) + the red-team suite are a lightweight, practical parallel to the RMF's **Map** (identify risks — error analysis) and **Measure** (evaluate against metrics — EX/EM, red-team pass rate) functions. There is no formal **Govern** (organizational policy/approval workflow) or **Manage** (ongoing risk response) function implemented here — those require an organizational context this solo project doesn't have.
- **EU AI Act**: a Text-to-SQL developer assistant of this kind would likely fall under limited-risk obligations (primarily transparency — users should know SQL is AI-generated) rather than the Act's high-risk categories (biometrics, credit scoring, critical infrastructure, etc.). This document itself, plus the model being clearly labeled as an AI-generated-SQL tool, is the practical transparency measure here.
- **Responsible AI principles**: disclosed limitations over silent gaps, a measured (not assumed) safety mitigation, and honest scope boundaries throughout this document are the concrete expression of that principle at this project's scale.

## Model Card Authors

Maintained alongside the rest of this repository; see `README.md` for the full pipeline and reproduction steps.
