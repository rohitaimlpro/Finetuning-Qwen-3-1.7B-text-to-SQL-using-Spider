"""Stage 1 baseline (and later fine-tuned) generation over the Spider dev set.

Loads a causal LM with `transformers`, generates SQL for every dev-set
question with a schema-aware prompt, applies the SELECT-only safety gate,
and records the outputs plus latency/throughput/GPU-memory metrics.

Usage:
    python src/generate_predictions.py \
        --model_name Qwen/Qwen3-1.7B \
        --data_dir data/spider \
        --output_dir results/baseline_qwen3_1p7b

Re-run with --model_name pointing at a fine-tuned checkpoint later to produce
a directly comparable results/ folder.
"""
import argparse
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_prompt
from schema_utils import build_schema_str, load_tables
from sql_safety import sanitize_prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--data_dir", default="data/spider", help="Root folder containing dev.json / tables.json")
    parser.add_argument("--split_file", default="dev.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples (quick smoke test)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, args.split_file), "r", encoding="utf-8") as f:
        examples = json.load(f)
    if args.limit:
        examples = examples[: args.limit]

    tables_by_db = load_tables(os.path.join(args.data_dir, "tables.json"))
    schema_cache = {db_id: build_schema_str(entry) for db_id, entry in tables_by_db.items()}

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # device_map="auto" lets accelerate offload to disk when it judges RAM
    # insufficient -- catastrophic for throughput. Only use it when there's a
    # GPU to place weights on; plain CPU runs load fully into RAM instead.
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map=device_map
    )
    if device_map is None:
        model = model.to("cpu")
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pred_lines = []
    gold_lines = []
    debug_records = []
    total_gen_tokens = 0
    total_gen_time = 0.0
    unsafe_count = 0

    for example in tqdm(examples, desc="Generating"):
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["query"]
        schema_str = schema_cache[db_id]

        prompt = build_prompt(tokenizer, question, schema_str, db_id)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        new_tokens = output_ids.shape[1] - input_len
        raw_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        predicted_sql, was_safe = sanitize_prediction(raw_text)
        if not was_safe:
            unsafe_count += 1

        total_gen_tokens += new_tokens
        total_gen_time += elapsed

        pred_lines.append(predicted_sql.replace("\n", " ").strip())
        gold_lines.append(f"{gold_sql}\t{db_id}")
        debug_records.append(
            {
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "raw_output": raw_text,
                "was_safe": was_safe,
                "latency_sec": elapsed,
                "new_tokens": new_tokens,
            }
        )

    with open(os.path.join(args.output_dir, "pred.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines) + "\n")
    with open(os.path.join(args.output_dir, "gold.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(gold_lines) + "\n")
    with open(os.path.join(args.output_dir, "predictions_debug.jsonl"), "w", encoding="utf-8") as f:
        for record in debug_records:
            f.write(json.dumps(record) + "\n")

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
    )
    perf_metrics = {
        "model_name": args.model_name,
        "num_examples": len(examples),
        "unsafe_predictions": unsafe_count,
        "total_generation_time_sec": total_gen_time,
        "avg_latency_sec": total_gen_time / len(examples),
        "tokens_per_sec": total_gen_tokens / total_gen_time if total_gen_time > 0 else None,
        "peak_gpu_memory_gb": peak_mem_gb,
    }
    with open(os.path.join(args.output_dir, "perf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(perf_metrics, f, indent=2)

    print(json.dumps(perf_metrics, indent=2))


if __name__ == "__main__":
    main()
