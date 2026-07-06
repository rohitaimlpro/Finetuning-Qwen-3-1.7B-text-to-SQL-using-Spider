"""QLoRA fine-tuning of Qwen3 on Spider Text-to-SQL via Unsloth, with
Colab-safe resumability.

Checkpoints (adapter weights + optimizer/scheduler/RNG state) save every
--save_steps steps directly under --output_dir. Point --output_dir at a
Drive path on Colab: if it already contains a checkpoint, training resumes
from the latest one automatically. Safe to just re-run this same command
after a Colab disconnect -- no flag needed to trigger resume.

Usage:
    python src/train_qlora.py \
        --model_name unsloth/Qwen3-1.7B \
        --data_dir data/spider \
        --output_dir results/qlora_qwen3_1p7b \
        --num_train_epochs 3

Note: this imports `unsloth` first (before transformers/trl) because its
patches need to apply before those libraries are touched.
"""
import argparse
import glob
import json
import os

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from prompts import build_messages
from schema_utils import build_schema_str, load_tables

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/Qwen3-1.7B")
    parser.add_argument("--data_dir", default="data/spider")
    parser.add_argument("--train_files", nargs="+", default=["train_spider.json", "train_others.json"])
    parser.add_argument("--output_dir", required=True, help="Point this at a Drive path so checkpoints survive a runtime reset")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Only train on the first N examples (smoke test)")
    return parser.parse_args()


def load_training_examples(data_dir, train_files):
    examples = []
    for filename in train_files:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            examples.extend(json.load(f))
    return examples


def build_dataset(tokenizer, examples, tables_by_db):
    schema_cache = {db_id: build_schema_str(entry) for db_id, entry in tables_by_db.items()}
    texts = []
    for ex in examples:
        db_id = ex["db_id"]
        messages = build_messages(ex["question"], schema_cache[db_id], db_id)
        messages.append({"role": "assistant", "content": ex["query"]})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[-1]))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Training's completion is tracked by final_adapter/ and merged_model/,
    # not just checkpoint-*/ -- those are safe to delete once training is
    # done (e.g. to free disk space) without losing the "already trained"
    # signal and accidentally retraining from scratch.
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    merged_dir = os.path.join(args.output_dir, "merged_model")

    if os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"{merged_dir} already exists -- training already fully complete. Nothing to do.")
        return

    already_trained = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

    if already_trained:
        print(f"Found completed adapter at {adapter_dir} -- skipping training, retrying merge only.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=TARGET_MODULES,
            lora_alpha=args.lora_r,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )

        tables_by_db = load_tables(os.path.join(args.data_dir, "tables.json"))
        examples = load_training_examples(args.data_dir, args.train_files)
        if args.limit:
            examples = examples[: args.limit]
        train_dataset = build_dataset(tokenizer, examples, tables_by_db)

        sft_config = SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
        )

        resume_path = find_latest_checkpoint(args.output_dir)
        if resume_path:
            print(f"Found existing checkpoint -- resuming from {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)

        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Adapter saved to {adapter_dir}")

    # Merge LoRA into the base model so generate_predictions.py can load it
    # with plain AutoModelForCausalLM.from_pretrained, same as the baseline.
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
