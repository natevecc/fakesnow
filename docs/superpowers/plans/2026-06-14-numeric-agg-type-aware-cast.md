# Type-Aware Numeric-Aggregate Cast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `numeric_agg_implicit_cast` cast only *text* columns to DOUBLE for numeric aggregates (SUM/AVG/…), leaving numeric columns untouched — so `SUM(varchar)` works while `SUM(bigint)`/`SUM(double)` keep their native type and wire-shape.

**Architecture:** The transform gains an optional live DuckDB connection. For a bare-column aggregate arg it resolves the column's type via `DESCRIBE` of the enclosing SELECT's own FROM/JOIN base tables: text → wrap in `TRY_CAST(... AS DOUBLE)`; numeric → leave alone; unresolvable (no connection, CTE/derived source, unknown table) → fall back to casting (value-correct, matches legacy behavior). Wired into the cursor exactly like `create_table_as` (lambda passing `self._duck_conn`).

**Tech Stack:** Python 3.13, sqlglot 30.6.x (`exp`, `.transform`, `find_ancestor`, `find_all`), DuckDB `DESCRIBE`.

**Background / why:** Commit `0da090f` excluded all bare `exp.Column` args from casting to stop demoting `SUM(bigint)`→DOUBLE (breaks the Node-SDK integer wire-shape / GAP-2), but that broke `SUM(varchar)` (DuckDB: `No function matches sum(VARCHAR)`). No single cast target is lossless (DOUBLE demotes ints + loses huge-int precision; DECIMAL truncates floats), so the cast must be type-aware. Real Snowflake: SUM/AVG of NUMBER→NUMBER (fixed), of FLOAT/VARCHAR→FLOAT (real) — this design matches that.

---

## Verified decisions (spiked against sqlglot 30.6.1.dev63 + live DuckDB)

- `agg.find_ancestor(exp.Select)` and `find_all` work inside `.transform()`.
- This sqlglot version stores FROM under arg key **`from_`** (not `from`) — so do **not** hardcode the key; iterate `select.args.values()` for `exp.From`/`exp.Join` nodes.
- Scope FROM/JOIN tables to those whose `find_ancestor(exp.Select) is select` — excludes derived-subquery-internal tables (prevents WHERE-subquery name-collision false matches).
- `DESCRIBE` must use the **bare table identifier without its alias** (`DESCRIBE t_int AS ti` fails) — rebuild `exp.Table(this=, db=, catalog=)` dropping the alias.
- All of these verified correct: varchar→cast, bigint/integer/double→skip, qualified col (`ti.amount`)→skip, join→skip, derived subquery→cast (fallback), unknown table→cast (fallback), schema-qualified+quoted (`gold."Mixed"`)→cast, no-connection→cast (fallback).
- **The existing unit test `test_numeric_agg_implicit_cast` passes unchanged** because it calls the transform with no connection → fallback cast → its pre-`0da090f` expectations hold.

**Known v1 limitation (acceptable, documented):** an integer column sourced through a CTE/derived table falls back to casting (→ DOUBLE/real), since `DESCRIBE <cte_name>` isn't a catalog object. Result *values* stay correct; only the wire-shape of integer aggregates over CTE-derived columns is affected. No current test/consumer depends on that specific shape.

---

## File structure

- **Modify** `fakesnow/transforms/transforms.py` — change `numeric_agg_implicit_cast` signature + cast decision; add three private helpers + one constant. One focused area; the alias-naming logic (`_numeric_agg_col_name`) is unchanged.
- **Modify** `fakesnow/cursor.py:323` — pass `self._duck_conn` via lambda.
- **Modify** `tests/test_transforms.py` — add a type-aware unit test (the existing `test_numeric_agg_implicit_cast` stays green unchanged).
- **Modify** `tests/test_fakes.py` — the existing `test_numeric_aggs_varchar_implicit_cast` now passes; add a `SUM(bigint)` integer-shape regression test.

Test command (venv only — bare `python` is a broken pyenv shim):
```
.venv/bin/python -m pytest tests/test_transforms.py tests/test_fakes.py -v
```

---

## Task 1: Type-aware cast in the transform

**Files:**
- Modify: `fakesnow/transforms/transforms.py`
- Test: `tests/test_transforms.py`

- [ ] **Step 1: Write the failing test** (type-aware behavior, exercised with a live DuckDB connection)

Add to `tests/test_transforms.py`:

```python
def test_numeric_agg_implicit_cast_type_aware() -> None:
    import duckdb

    from fakesnow.transforms.transforms import numeric_agg_implicit_cast

    con = duckdb.connect()
    con.execute("CREATE TABLE t_txt (amount VARCHAR)")
    con.execute("CREATE TABLE t_num (n BIGINT, m INTEGER, f DOUBLE)")

    def rendered(sql: str) -> str:
        return (
            sqlglot.parse_one(sql, read="snowflake")
            .transform(lambda e: numeric_agg_implicit_cast(e, con))
            .sql(dialect="duckdb")
        )

    # text column -> cast so DuckDB can aggregate it
    assert "TRY_CAST(amount AS DOUBLE)" in rendered("SELECT SUM(amount) FROM t_txt")
    # numeric columns -> left untouched (preserve native type / wire-shape)
    assert rendered("SELECT SUM(n) FROM t_num") == "SELECT SUM(n) FROM t_num"
    assert rendered("SELECT AVG(m) FROM t_num") == "SELECT AVG(m) FROM t_num"
    assert rendered("SELECT SUM(f) FROM t_num") == "SELECT SUM(f) FROM t_num"
    # qualified numeric column -> skip
    assert rendered("SELECT SUM(t.n) FROM t_num t") == "SELECT SUM(t.n) FROM t_num AS t"
    # unknown/unresolvable -> safe fallback cast
    assert "TRY_CAST(x AS DOUBLE)" in rendered("SELECT SUM(x) FROM nope")

    # no connection -> legacy fallback (cast all bare columns)
    assert (
        sqlglot.parse_one("SELECT SUM(amount) FROM t").transform(numeric_agg_implicit_cast).sql()
        == 'SELECT SUM(TRY_CAST(amount AS DOUBLE)) AS "SUM(AMOUNT)" FROM t'
    )
```

- [ ] **Step 2: Run it — expect FAIL** (`numeric_agg_implicit_cast` currently takes one arg; the `lambda e: ...(e, con)` call raises `TypeError`, and bare-column casts are currently skipped).

Run: `.venv/bin/python -m pytest tests/test_transforms.py::test_numeric_agg_implicit_cast_type_aware -v`

- [ ] **Step 3: Implement.** In `fakesnow/transforms/transforms.py`:

(a) Confirm `DuckDBPyConnection` is importable in this file (it's used by `create_table_as`). If not already imported, add `from duckdb import DuckDBPyConnection` (match however `create_table_as` references the type).

(b) Add this constant near `_NUMERIC_ONLY_AGGS`:

```python
# DuckDB type-name prefixes treated as text; numeric aggregates must cast these.
_TEXT_COLUMN_PREFIXES = ("VARCHAR", "CHAR", "TEXT", "STRING", "BPCHAR")
```

(c) Add these helpers (above `numeric_agg_implicit_cast`):

```python
def _agg_from_join_tables(select: exp.Select) -> list[exp.Table]:
    """The select's OWN FROM + JOIN base tables (version-agnostic arg key;
    excludes tables nested inside derived subqueries)."""
    tables: list[exp.Table] = []
    for value in select.args.values():
        for node in value if isinstance(value, list) else [value]:
            if isinstance(node, (exp.From, exp.Join)):
                for table in node.find_all(exp.Table):
                    if table.find_ancestor(exp.Select) is select:
                        tables.append(table)
    return tables


def _describe_table_columns(duck_conn: DuckDBPyConnection, table: exp.Table) -> dict[str, str]:
    """Map lowercased column name -> DuckDB type for `table`, via DESCRIBE on the
    bare identifier (alias stripped). Returns {} on any failure (fail-open)."""
    reference = exp.Table(this=table.this, db=table.args.get("db"), catalog=table.args.get("catalog"))
    try:
        rows = duck_conn.execute(f"DESCRIBE {reference.sql(dialect='duckdb')}").fetchall()
    except Exception:
        return {}
    return {row[0].lower(): row[1] for row in rows}


def _numeric_agg_column_needs_cast(
    duck_conn: DuckDBPyConnection | None, agg: exp.Expression, column: exp.Column
) -> bool:
    """True if `column` is a text type (cast needed) OR cannot be resolved to a
    known numeric base-table column (safe fallback). False only when it resolves
    to a known numeric column, in which case the native type is preserved."""
    if duck_conn is None:
        return True
    select = agg.find_ancestor(exp.Select)
    if select is None:
        return True
    qualifier = column.table
    name = column.name.lower()
    for table in _agg_from_join_tables(select):
        if qualifier and table.alias_or_name.lower() != qualifier.lower():
            continue
        columns = _describe_table_columns(duck_conn, table)
        if name in columns:
            return columns[name].upper().startswith(_TEXT_COLUMN_PREFIXES)
    return True
```

(d) Change the `numeric_agg_implicit_cast` signature and cast decision. New version:

```python
def numeric_agg_implicit_cast(expression: Expr, duck_conn: DuckDBPyConnection | None = None) -> Expr:
    """Wrap text arguments to numeric aggregate functions with TRY_CAST(... AS DOUBLE).

    Snowflake implicitly casts VARCHAR to numeric for aggregates like SUM(), AVG(),
    MEDIAN(). DuckDB rejects SUM(VARCHAR), so text columns are cast. Numeric columns
    are left untouched so their native type and Snowflake wire-shape are preserved
    (casting a BIGINT to DOUBLE would demote it to a float). When the argument's type
    cannot be resolved (no connection, CTE/derived source, unknown table) the cast is
    applied as a safe, value-correct fallback.

    Example:
        >>> import sqlglot
        >>> sqlglot.parse_one("SELECT SUM(amount) FROM t").transform(numeric_agg_implicit_cast).sql()
        'SELECT SUM(TRY_CAST(amount AS DOUBLE)) AS "SUM(AMOUNT)" FROM t'
    """
    if isinstance(expression, _NUMERIC_ONLY_AGGS):
        arg = expression.this
        col_name = None if isinstance(expression.parent, exp.Alias) else _numeric_agg_col_name(expression, arg)
        if not isinstance(arg, (exp.Cast, exp.TryCast)):
            should_cast = (
                _numeric_agg_column_needs_cast(duck_conn, expression, arg)
                if isinstance(arg, exp.Column)
                else True
            )
            if should_cast:
                expression.set(
                    "this",
                    exp.TryCast(this=arg, to=exp.DataType(this=exp.DataType.Type.DOUBLE)),
                )
        if col_name and isinstance(expression.parent, exp.Select):
            return exp.alias_(expression, col_name, quoted=True)
    return expression
```

- [ ] **Step 4: Run both the new test and the existing unit test.**

Run: `.venv/bin/python -m pytest tests/test_transforms.py::test_numeric_agg_implicit_cast_type_aware tests/test_transforms.py::test_numeric_agg_implicit_cast -v`
Expected: BOTH PASS. (The existing `test_numeric_agg_implicit_cast` must pass unchanged — it calls the transform with no connection, hitting the fallback.)

- [ ] **Step 5: Commit**

```bash
git add fakesnow/transforms/transforms.py tests/test_transforms.py
git commit -m "feat(agg): type-aware numeric-aggregate cast (cast text columns only)"
```

---

## Task 2: Wire the connection into the cursor

**Files:**
- Modify: `fakesnow/cursor.py` (line 323)
- Test: `tests/test_fakes.py`

- [ ] **Step 1: Confirm the failing integration test.** The existing `tests/test_fakes.py::test_numeric_aggs_varchar_implicit_cast` fails at HEAD (`No function matches sum(VARCHAR)`), because Task 1's transform only casts text columns *when given a connection* — and the cursor doesn't pass one yet.

Run: `.venv/bin/python -m pytest tests/test_fakes.py::test_numeric_aggs_varchar_implicit_cast -v`
Expected: FAIL (BinderException surfaced as ProgrammingError).

- [ ] **Step 2: Wire it in.** In `fakesnow/cursor.py`, change line 323 from:

```python
            .transform(transforms.numeric_agg_implicit_cast)
```

to:

```python
            .transform(lambda e: transforms.numeric_agg_implicit_cast(e, self._duck_conn))
```

- [ ] **Step 3: Run the varchar integration test.**

Run: `.venv/bin/python -m pytest tests/test_fakes.py::test_numeric_aggs_varchar_implicit_cast -v`
Expected: PASS (`SUM(AMOUNT) == 600.0`, etc.).

- [ ] **Step 4: Add the integer-shape regression test.** Add to `tests/test_fakes.py` (near the varchar test; match its existing fixture style — `dcur` DictCursor):

```python
def test_numeric_aggs_integer_preserves_fixed_shape(dcur: snowflake.connector.cursor.DictCursor):
    """SUM over an integer column must NOT be demoted to a float: it keeps the
    fixed/NUMBER wire-shape (scale 0) and full integer precision."""
    dcur.execute("CREATE TABLE t_int_agg (n BIGINT)")
    dcur.execute("INSERT INTO t_int_agg VALUES (4611686018427387904), (1), (2)")  # 2**62 + small
    dcur.execute("SELECT SUM(n) AS total FROM t_int_agg")
    row = dcur.fetchone()
    assert row is not None
    # exact integer value, no float rounding
    assert row["TOTAL"] == 4611686018427387907
    # fixed (NUMBER) wire type, scale 0 — type_code 0 is FIXED in the snowflake connector
    assert dcur.description[0].type_code == 0
    assert dcur.description[0].scale == 0
```

- [ ] **Step 5: Run it.**

Run: `.venv/bin/python -m pytest tests/test_fakes.py::test_numeric_aggs_integer_preserves_fixed_shape -v`
Expected: PASS. (If `description[0].scale` is `None` rather than `0` for this connector/path, assert `in (0, None)` — but verify the exact value first; do NOT weaken without checking.)

- [ ] **Step 6: Commit**

```bash
git add fakesnow/cursor.py tests/test_fakes.py
git commit -m "feat(agg): pass duck_conn so numeric-agg cast is type-aware in execute"
```

---

## Task 3: Full-suite regression + lint/types

**Files:** none (verify; fix-ups only if attributable to this change)

- [ ] **Step 1: Full suite.** Run `.venv/bin/python -m pytest -q`.
Expected: the two numeric_agg tests now PASS. Remaining failures should be only the unrelated pre-existing ones not touched here (e.g. `test_cli.py::test_run_server`, an environment/CLI issue). Confirm **zero new** failures vs. that baseline and that BOTH numeric_agg tests are green.

- [ ] **Step 2: Lint + types on changed files.**

Run: `.venv/bin/python -m ruff check fakesnow/transforms/transforms.py fakesnow/cursor.py tests/test_transforms.py tests/test_fakes.py`
Run: `.venv/bin/python -m pyright fakesnow/transforms/transforms.py fakesnow/cursor.py`
Expected: clean on these files (ignore pre-existing findings elsewhere). Fix any new finding attributable to this change.

- [ ] **Step 3: Commit any fix-ups**

```bash
git add -A
git commit -m "chore(agg): lint and type fixes"
```

---

## Self-Review

**Spec coverage:** text→cast (Task 1 test + varchar integration), numeric→skip (Task 1 test), integer wire-shape preserved (Task 2 regression test), fallback/no-connection (Task 1 test), cursor wiring (Task 2). ✓
**Placeholder scan:** none — all code complete; the only conditional is the documented `scale in (0, None)` contingency, gated on verifying the real value first. ✓
**Type/name consistency:** `numeric_agg_implicit_cast(expression, duck_conn=None)`; helpers `_agg_from_join_tables`, `_describe_table_columns`, `_numeric_agg_column_needs_cast`; constant `_TEXT_COLUMN_PREFIXES`. Cursor lambda matches the `create_table_as` precedent. ✓
