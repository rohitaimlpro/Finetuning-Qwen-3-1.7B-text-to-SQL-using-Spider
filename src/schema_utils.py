"""Turn Spider's tables.json entries into readable CREATE TABLE schema text."""
import json


def load_tables(tables_json_path):
    """Return {db_id: table_entry} for every database in tables.json."""
    with open(tables_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["db_id"]: entry for entry in entries}


def build_schema_str(table_entry):
    """Render a Spider tables.json entry as CREATE TABLE statements.

    table_entry fields (Spider format):
      table_names_original: [str]
      column_names_original: [[table_idx, col_name]]  (table_idx -1 is the "*" column)
      column_types: [str]
      primary_keys: [col_idx]
      foreign_keys: [[col_idx, ref_col_idx]]
    """
    table_names = table_entry["table_names_original"]
    columns = table_entry["column_names_original"]
    col_types = table_entry["column_types"]
    primary_keys = set(table_entry.get("primary_keys", []))

    columns_by_table = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, col_name) in enumerate(columns):
        if table_idx == -1:
            continue
        is_pk = col_idx in primary_keys
        columns_by_table[table_idx].append((col_name, col_types[col_idx], is_pk))

    statements = []
    for table_idx, table_name in enumerate(table_names):
        lines = []
        for col_name, col_type, is_pk in columns_by_table[table_idx]:
            suffix = " PRIMARY KEY" if is_pk else ""
            lines.append(f"  {col_name} {col_type.upper()}{suffix}")
        body = ",\n".join(lines)
        statements.append(f"CREATE TABLE {table_name} (\n{body}\n);")

    foreign_keys = table_entry.get("foreign_keys", [])
    if foreign_keys:
        fk_lines = []
        for col_idx, ref_col_idx in foreign_keys:
            src_table = table_names[columns[col_idx][0]]
            src_col = columns[col_idx][1]
            ref_table = table_names[columns[ref_col_idx][0]]
            ref_col = columns[ref_col_idx][1]
            fk_lines.append(f"-- FOREIGN KEY: {src_table}.{src_col} -> {ref_table}.{ref_col}")
        statements.append("\n".join(fk_lines))

    return "\n\n".join(statements)
