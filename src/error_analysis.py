"""Bucket failure types from a generate_predictions.py debug file.

Static/heuristic analysis only (table/join/clause comparison via sqlparse) --
no SQL execution needed, so this runs anywhere predictions_debug.jsonl is,
without needing the Spider database files or tables.json.

Usage:
    python src/error_analysis.py \
        --debug_file results/baseline_qwen3_1p7b/predictions_debug.jsonl \
        --output_dir results/baseline_qwen3_1p7b
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import sqlparse

AGG_FUNCS = {"count", "avg", "sum", "max", "min"}
SET_OPS = {"intersect", "except", "union"}

BUCKET_ORDER = [
    "unsafe_generation",
    "syntax_error",
    "hallucinated_table",
    "missing_table",
    "missing_set_operation",
    "missing_join",
    "missing_group_by",
    "missing_having",
    "missing_order_by",
    "aggregation_mismatch",
    "subquery_vs_join_style",
    "other_non_exact_match",
    "exact_match",
]


def normalize(sql):
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def extract_features(sql):
    sql_lower = sql.lower()
    parsed = sqlparse.parse(sql)
    is_valid_select = bool(parsed) and parsed[0].get_type() == "SELECT"

    tables = set(re.findall(r"\bfrom\s+([a-zA-Z_]\w*)", sql_lower))
    tables |= set(re.findall(r"\bjoin\s+([a-zA-Z_]\w*)", sql_lower))

    return {
        "is_valid_select": is_valid_select,
        "tables": tables,
        "num_joins": len(re.findall(r"\bjoin\b", sql_lower)),
        "has_group_by": "group by" in sql_lower,
        "has_having": "having" in sql_lower,
        "has_order_by": "order by" in sql_lower,
        "has_subquery": sql_lower.count("select") > 1,
        "set_ops": {op for op in SET_OPS if re.search(rf"\b{op}\b", sql_lower)},
        "agg_funcs": {fn for fn in AGG_FUNCS if re.search(rf"\b{fn}\s*\(", sql_lower)},
    }


def classify(record):
    """Return the single failure bucket that best explains why predicted_sql
    doesn't match gold_sql. Checks run in order of how decisive/severe the
    mismatch is -- e.g. a hallucinated table matters more than a missing
    ORDER BY, so it's reported first even if both are true."""
    if not record["was_safe"]:
        return "unsafe_generation"

    gold_sql = record["gold_sql"]
    pred_sql = record["predicted_sql"]

    if normalize(gold_sql) == normalize(pred_sql):
        return "exact_match"

    gold_feat = extract_features(gold_sql)
    pred_feat = extract_features(pred_sql)

    if not pred_feat["is_valid_select"]:
        return "syntax_error"

    if pred_feat["tables"] - gold_feat["tables"]:
        return "hallucinated_table"
    if gold_feat["tables"] - pred_feat["tables"]:
        return "missing_table"

    if gold_feat["set_ops"] and not pred_feat["set_ops"]:
        return "missing_set_operation"

    if gold_feat["num_joins"] > pred_feat["num_joins"]:
        return "missing_join"

    if gold_feat["has_group_by"] and not pred_feat["has_group_by"]:
        return "missing_group_by"

    if gold_feat["has_having"] and not pred_feat["has_having"]:
        return "missing_having"

    if gold_feat["has_order_by"] and not pred_feat["has_order_by"]:
        return "missing_order_by"

    if gold_feat["agg_funcs"] != pred_feat["agg_funcs"]:
        return "aggregation_mismatch"

    if gold_feat["has_subquery"] != pred_feat["has_subquery"]:
        return "subquery_vs_join_style"

    return "other_non_exact_match"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_file", required=True, help="predictions_debug.jsonl from generate_predictions.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_bucket", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    with open(args.debug_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    buckets = defaultdict(list)
    for record in records:
        bucket = classify(record)
        buckets[bucket].append(record)

    total = len(records)
    counts = {b: len(buckets[b]) for b in BUCKET_ORDER if b in buckets}
    summary = {
        "total_examples": total,
        "buckets": {
            b: {"count": c, "pct": round(100 * c / total, 1)}
            for b, c in sorted(counts.items(), key=lambda kv: -kv[1])
        },
    }

    with open(os.path.join(args.output_dir, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [f"# Error analysis ({total} examples)\n"]
    for bucket, stats in summary["buckets"].items():
        lines.append(f"## {bucket} -- {stats['count']} ({stats['pct']}%)\n")
        for record in buckets[bucket][: args.samples_per_bucket]:
            lines.append(f"- **Q**: {record['question']}")
            lines.append(f"  - gold: `{record['gold_sql']}`")
            lines.append(f"  - pred: `{record['predicted_sql']}`")
        lines.append("")

    report_path = os.path.join(args.output_dir, "error_analysis_samples.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, indent=2))
    print(f"\nSample report saved to {report_path}")


if __name__ == "__main__":
    main()
