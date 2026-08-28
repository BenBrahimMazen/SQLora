"""Normalize SQL strings for exact-match comparison.

Pipeline position: used by src/evaluate.py to compute the Exact Match (EM)
metric over predictions, in parallel with execution accuracy
(src/execution_eval.py). Standalone CLI for quick checks.

This is a pragmatic normalizer for the Spider setting, NOT a full SQL parser.
It canonicalizes surface forms that are semantically irrelevant in SQLite:

    - whitespace / newlines (collapsed to single spaces)
    - casing (everything lowercased)
    - trailing semicolons
    - quote style ('x' vs "x" string literals -> 'x')
    - operator spacing ("a=b" vs "a = b"; "> =" -> ">=")
    - "<>" unified to "!="
    - comma / parenthesis spacing
    - column order in the top-level SELECT list ("SELECT b, a" == "SELECT a, b")
      — result-set semantics don't depend on projection order

Known limitations (documented, acceptable for this project's purposes):
    - values are compared case-sensitively after lowercasing the whole query,
      so 'Hello' vs 'hello' predictions count as mismatches;
    - a keyword appearing inside a string literal can confuse the SELECT-list
      sorter (it bails and leaves the query unchanged rather than corrupt it).

Stdlib only — covered by tests/test_sql_normalization.py in offline CI.
"""

from __future__ import annotations

import argparse
import re


def _collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def _unify_quotes(s: str) -> str:
    """Convert double-quoted string literals to single-quoted ones.

    SQLite tolerates "value" for string literals (legacy quirk), so gold and
    predicted queries may legitimately differ in quote style. Identifiers that
    happen to be double-quoted are also rewritten — a rare, documented
    limitation (Spider queries quote identifiers almost never).
    """
    return re.sub(r'"([^"]*)"', r"'\1'", s)


def _unify_operators(s: str) -> str:
    # Merge accidentally-split multi-char operators first.
    s = re.sub(r">\s*=", ">=", s)
    s = re.sub(r"<\s*=", "<=", s)
    s = re.sub(r"!\s*=", "!=", s)
    s = re.sub(r"<\s*>", "!=", s)  # <> and != mean the same in SQLite
    # Exactly one space around comparison operators.
    s = re.sub(r"\s*(>=|<=|!=|=|>|<)\s*", r" \1 ", s)
    return s


def _unify_commas_and_parens(s: str) -> str:
    s = re.sub(r"\s*,\s*", ", ", s)
    # No whitespace touching either side of a paren ("count (*)" == "count(*)").
    # The output is for comparison only, not re-execution.
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _find_keyword(s: str, pos: int, keyword: str) -> bool:
    """True if `keyword` (lowercase) occurs at pos with word boundaries."""
    end = pos + len(keyword)
    if s[pos:end] != keyword:
        return False
    before_ok = pos == 0 or not _is_word_char(s[pos - 1])
    after_ok = end >= len(s) or not _is_word_char(s[end])
    return before_ok and after_ok


def _sort_select_items(s: str) -> str:
    """Sort the top-level SELECT list alphabetically ("select b, a" -> "select a, b").

    Only the first SELECT clause is touched. The scanner tracks parenthesis
    depth and skips string literals, and bails (returns the input unchanged)
    on anything it cannot parse confidently — never corrupts the query.
    Skipped when the list contains a bare ``*`` or ``table.*`` projection.
    """
    m = re.search(r"\bselect\b", s)
    if not m:
        return s
    head_end = m.end()

    # Consume an optional DISTINCT / ALL so it stays prefix of the list.
    prefix = ""
    mdist = re.match(r"\s+(distinct|all)\b", s[head_end:])
    if mdist:
        prefix = mdist.group(1)
        head_end += mdist.end()

    items: list[tuple[int, int]] = []
    item_start = head_end
    depth = 0
    from_pos = len(s)
    j = head_end
    while j < len(s):
        c = s[j]
        if c == "'":  # skip string literal (handles doubled '' escapes)
            j += 1
            while j < len(s):
                if s[j] == "'":
                    if j + 1 < len(s) and s[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            j += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return s  # unbalanced — bail
            depth -= 1
        elif depth == 0 and c == ",":
            items.append((item_start, j))
            item_start = j + 1
        elif depth == 0 and _find_keyword(s, j, "from"):
            from_pos = j
            break
        j += 1
    if depth != 0:
        return s

    items.append((item_start, from_pos))
    texts = [s[a:b].strip() for a, b in items]
    if len(texts) < 2:
        return s
    # '*' and 't.*' projections: leave the list order alone.
    if any(t == "*" or t.endswith(".*") for t in texts):
        return s
    sorted_texts = sorted(texts)

    head = s[: m.end()] + (f" {prefix}" if prefix else "")
    tail = s[from_pos:] if from_pos < len(s) else ""
    rebuilt = f"{head} {', '.join(sorted_texts)}"
    return f"{rebuilt} {tail}" if tail else rebuilt


def normalize_sql(sql: str) -> str:
    """Normalize a SQL string for exact-match comparison (see module docstring)."""
    s = (sql or "").strip()
    s = re.sub(r";\s*$", "", s)          # trailing semicolon
    s = _collapse_whitespace(s)
    s = s.lower()
    s = _unify_quotes(s)
    s = _unify_operators(s)
    s = _unify_commas_and_parens(s)
    s = _sort_select_items(s)
    return s.strip()


def sql_exact_match(predicted: str, gold: str) -> bool:
    """Exact-match comparison after normalization. Empty predictions never match."""
    if not (predicted or "").strip():
        return False
    return normalize_sql(predicted) == normalize_sql(gold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize SQL strings / compare two queries for exact match."
    )
    parser.add_argument("sql", nargs="+", help="One or two SQL strings to normalize/compare")
    args = parser.parse_args()
    normalized = [normalize_sql(s) for s in args.sql]
    for original, norm in zip(args.sql, normalized):
        print(f"  in : {original}")
        print(f"  out: {norm}")
    if len(args.sql) == 2:
        print("  exact match:", normalized[0] == normalized[1])


if __name__ == "__main__":
    main()
