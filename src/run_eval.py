"""Run the official Spider test-suite evaluation (Exact Match + Execution
Accuracy, broken down by difficulty) against a pred/gold pair produced by
generate_predictions.py.

Requires the eval suite to be vendored first:
    git clone https://github.com/taoyds/test-suite-sql-eval third_party/test_suite_sql_eval

Usage:
    python src/run_eval.py \
        --gold results/baseline_qwen3_1p7b/gold.txt \
        --pred results/baseline_qwen3_1p7b/pred.txt \
        --db data/spider/database \
        --table data/spider/tables.json \
        --output_dir results/baseline_qwen3_1p7b
"""
import argparse
import os
import subprocess
import sys

EVAL_SUITE_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "test_suite_sql_eval")


def ensure_nltk_data():
    """The vendored eval suite tokenizes gold/predicted SQL with nltk's
    word_tokenize, which needs punkt_tab downloaded first -- not bundled
    with nltk itself, and easy to miss until evaluation.py crashes on it."""
    import nltk

    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--db", required=True, help="Path to Spider's database/ folder (sqlite files)")
    parser.add_argument("--table", required=True, help="Path to tables.json")
    parser.add_argument("--etype", default="all", choices=["all", "exec", "match"])
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_nltk_data()
    eval_script = os.path.join(EVAL_SUITE_DIR, "evaluation.py")
    if not os.path.exists(eval_script):
        sys.exit(
            "Eval suite not found. Run:\n"
            "  git clone https://github.com/taoyds/test-suite-sql-eval "
            f"{EVAL_SUITE_DIR}"
        )

    cmd = [
        sys.executable, eval_script,
        "--gold", args.gold,
        "--pred", args.pred,
        "--db", args.db,
        "--table", args.table,
        "--etype", args.etype,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    log_path = os.path.join(args.output_dir, "eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nFull breakdown (by difficulty) saved to {log_path}")


if __name__ == "__main__":
    main()
