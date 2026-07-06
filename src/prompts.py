"""Build text-to-SQL prompts using the model's chat template."""

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a question, "
    "generate a single syntactically correct SQLite query that answers the question. "
    "Output only the SQL query with no explanation, no markdown formatting, and no code fences."
)


def build_messages(question, schema_str, db_id):
    user_content = (
        f"Database: {db_id}\n\n"
        f"Schema:\n{schema_str}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt(tokenizer, question, schema_str, db_id):
    messages = build_messages(question, schema_str, db_id)
    # Qwen3 supports an enable_thinking switch; disabled for eval so generation
    # is fast, deterministic, and directly comparable pre/post fine-tuning.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
