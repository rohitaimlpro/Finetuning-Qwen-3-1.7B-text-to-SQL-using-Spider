"""Reject any generated SQL that isn't a single read-only SELECT statement."""
import re

import sqlparse

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "truncate",
    "replace", "attach", "detach", "pragma", "vacuum", "create",
}


def extract_sql(raw_text):
    """Strip code fences / stray prose the model may still emit."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    return text.strip().rstrip(";").strip()


def _has_forbidden_keyword(stmt):
    """Walk actual SQL keyword tokens (not string literals or identifiers)
    looking for forbidden statements, including inside subqueries."""
    for token in stmt.flatten():
        if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML):
            if token.value.lower() in FORBIDDEN_KEYWORDS:
                return True
    return False


def is_safe_select(sql):
    """True if sql is exactly one SELECT statement with no forbidden keywords."""
    if not sql:
        return False

    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        return False

    parsed = sqlparse.parse(statements[0])
    if not parsed or parsed[0].get_type() != "SELECT":
        return False

    if _has_forbidden_keyword(parsed[0]):
        return False

    return True


def sanitize_prediction(raw_text, fallback="SELECT 1"):
    """Return (sql_to_use, was_safe). Unsafe/invalid output is swapped for a
    harmless placeholder so downstream execution-accuracy scoring never runs
    an unvetted statement against the database."""
    sql = extract_sql(raw_text)
    if is_safe_select(sql):
        return sql, True
    return fallback, False
