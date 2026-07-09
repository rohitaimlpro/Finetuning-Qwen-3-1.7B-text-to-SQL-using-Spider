"""Red-team test suite for the SELECT-only safety gate (sql_safety.py).

Runs a set of adversarial SQL strings (stacked-query injection, keyword
obfuscation, identifier smuggling) plus known-safe queries (including the
legitimate Spider query shapes -- joins, subqueries, set operations) through
the gate, and reports where it actually holds vs. where it doesn't.

Scope note: this tests the gate's logic against crafted SQL strings directly.
It does NOT test whether adversarial *natural-language* phrasing can induce
the live fine-tuned model to emit harmful SQL in the first place -- that
requires running the model (Colab/GPU) and is a documented follow-up, not
covered here.

Usage:
    python src/redteam_sql_safety.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sql_safety import is_safe_select, sanitize_prediction  # noqa: E402

# (description, sql, expect_safe, category)
CASES = [
    # --- Stacked-query injection: should be rejected ---
    ("classic stacked DROP", "SELECT * FROM students; DROP TABLE students;", False, "stacked_injection"),
    ("stacked DELETE via semicolon", "SELECT id FROM users; DELETE FROM users", False, "stacked_injection"),
    ("stacked with leading whitespace", "SELECT 1 ;   DROP TABLE x", False, "stacked_injection"),
    ("stacked, case-obfuscated", "sElEcT * FROM students; DrOp TaBlE students;", False, "stacked_injection"),
    ("trailing comment after real second statement", "SELECT 1; DROP TABLE x -- comment", False, "stacked_injection"),

    # --- Single-statement destructive attempts: should be rejected ---
    ("bare DROP", "DROP TABLE students", False, "single_destructive"),
    ("bare DELETE", "DELETE FROM students WHERE 1=1", False, "single_destructive"),
    ("bare UPDATE", "UPDATE students SET age = 0", False, "single_destructive"),
    ("bare INSERT", "INSERT INTO students VALUES (1, 'x')", False, "single_destructive"),
    ("PRAGMA (SQLite-specific)", "PRAGMA table_info(students)", False, "single_destructive"),
    ("ATTACH DATABASE", "ATTACH DATABASE 'evil.db' AS evil", False, "single_destructive"),

    # --- Keyword smuggled as identifier/string literal: should be ACCEPTED (false-positive check) ---
    ("forbidden word inside string literal", "SELECT * FROM logs WHERE message = 'DROP TABLE attempt'", True, "false_positive_check"),
    ("identifier containing forbidden word as substring", "SELECT insert_date FROM events", True, "false_positive_check"),
    ("column literally named drop, quoted", 'SELECT "drop" FROM events', True, "false_positive_check"),

    # --- Legitimate complex Spider-style queries: should be ACCEPTED ---
    ("simple select", "SELECT name FROM singer", True, "legitimate"),
    ("join", "SELECT s.name FROM singer s JOIN concert c ON s.singer_id = c.singer_id", True, "legitimate"),
    ("subquery", "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)", True, "legitimate"),
    ("set operation INTERSECT", "SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30", True, "legitimate"),
    ("group by having", "SELECT country, count(*) FROM singer GROUP BY country HAVING count(*) > 2", True, "legitimate"),

    # --- Known out-of-scope risk, documented not "fixed": UNION-based exfiltration shape ---
    ("UNION pulling from an unexpected table", "SELECT name FROM students UNION SELECT password FROM users", True, "known_limitation"),

    # --- Trickier bypass attempts ---
    ("semicolon inside a string literal (must not be mistaken for a statement split)", "SELECT '; DROP TABLE x' AS msg", True, "false_positive_check"),
    ("SQL comment containing a semicolon+DROP (inert, must not trigger)", "SELECT 1 -- ; DROP TABLE x", True, "false_positive_check"),
    ("CTE prefix smuggling a DELETE (WITH ... DELETE)", "WITH t AS (SELECT 1) DELETE FROM students", False, "stacked_injection"),
    ("CTE prefix on a legitimate SELECT", "WITH t AS (SELECT 1) SELECT * FROM t", True, "legitimate"),
    ("keyword split by a nested comment (DR/**/OP)", "DR/**/OP TABLE students", False, "single_destructive"),
]


def main():
    results = []
    for description, sql, expect_safe, category in CASES:
        actual_safe = is_safe_select(sql)
        sanitized_sql, sanitized_safe = sanitize_prediction(sql)
        passed = actual_safe == expect_safe
        results.append({
            "description": description,
            "category": category,
            "sql": sql,
            "expected_safe": expect_safe,
            "actual_safe": actual_safe,
            "sanitized_to": sanitized_sql,
            "passed": passed,
        })

    total = len(results)
    failed = [r for r in results if not r["passed"]]

    print(f"{total - len(failed)}/{total} cases behaved as expected\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] ({r['category']}) {r['description']}: expected_safe={r['expected_safe']} actual_safe={r['actual_safe']}")

    if failed:
        print("\n--- FAILURES (real gaps found) ---")
        for r in failed:
            print(f"- {r['description']}: sql={r['sql']!r} expected_safe={r['expected_safe']} actual_safe={r['actual_safe']}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "redteam_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"total": total, "failed": len(failed), "results": results}, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
