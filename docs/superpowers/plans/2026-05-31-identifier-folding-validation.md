# Identifier-Folding Validation Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fakesnow raise a Snowflake-style error when a query references an identifier (CTE/table name or column/alias) with quoting that mismatches its definition under Snowflake folding — a class of bug DuckDB silently resolves.

**Architecture:** A *validation pass* that runs on a **copy** of the parsed Snowflake AST and only ever *raises* — it never mutates the AST that becomes DuckDB SQL, so query results are unchanged. It uses sqlglot's own Snowflake-folding resolution (`normalize_identifiers` + `traverse_scope` + `qualify`) to detect mismatches, because DuckDB itself is case-insensitive even for quoted identifiers and will not raise. Hooked into `cursor.execute` per exploded statement, before transform.

**Tech Stack:** Python 3.13, sqlglot 30.6.x optimizer (`normalize_identifiers`, `qualify`, `scope`), `snowflake.connector.errors.ProgrammingError`, pytest 9.

**Spec:** `docs/superpowers/specs/2026-05-31-identifier-folding-validation-design.md`

---

## Decisions locked during planning

These resolve the spec's open items. They were verified empirically against the installed `sqlglot 30.6.1.dev63` and the fakesnow source.

1. **Public API:** `fakesnow.transforms.identifier_folding.check_folding(expression, *, quoted_identifiers_ignore_case=False) -> None`. Raises on a positive mismatch; otherwise returns `None`.

2. **Error class & codes** (mirroring fakesnow's existing DuckDB→Snowflake conversions in `cursor.py:412-430` and real Snowflake wording):
   - Table/CTE mismatch → `ProgrammingError(msg=f"SQL compilation error:\nObject '{ref}' does not exist or not authorized.", errno=2003, sqlstate="42S02")` — same sqlstate fakesnow already uses for a missing table (`test_sqlstate` asserts `"42S02"`).
   - Column mismatch → `ProgrammingError(msg=f"SQL compilation error: invalid identifier '{ref}'", errno=904, sqlstate="42000")` — Snowflake's real "invalid identifier" code.

3. **Two qualify message formats** (both must be handled — verified): an *unqualified* unresolved column raises `Column 'X' could not be resolved. Line: N, Col: M`; a *qualified* reference (e.g. `ee.period` — the real numerator pattern) raises `Unknown column: X`. The regex matches both.

4. **`QUOTED_IDENTIFIERS_IGNORE_CASE`:** There is **no session-parameter store reachable from the cursor** (verified: `FakeSnowflakeConnection.__init__` keeps no session-param dict; `SET QUOTED_IDENTIFIERS_IGNORE_CASE=TRUE` currently raises `NotImplementedError` in `transforms.alter_session`). The parameter's default is `FALSE`, which means checks-active — exactly what we want, and the only reachable state today. **Decision (YAGNI):** implement the behavior via a `quoted_identifiers_ignore_case` keyword arg (default `False`) that short-circuits the checks, and have the cursor pass the default. Do **not** build speculative connection session-state plumbing; the arg is the clean extension point for when someone implements the `SET …=TRUE` path. This is faithful to Snowflake today (the param can't be TRUE) and leaves a single obvious wire-up point later.

5. **Statement gate (perf + correctness):** only statements containing a `SELECT` can have an intra-query folding mismatch. Skip anything else (`expression.find(exp.Select) is None`) cheaply — covers DDL/DML/session statements and bounds per-query overhead.

6. **Fail-open:** the pass wraps analysis in `try/except`; any internal error (parse quirk, unsupported construct) is logged at debug and swallowed. It re-raises only the `ProgrammingError` it deliberately throws. This structurally bounds false positives on unseen queries.

---

## File structure

- **Create** `fakesnow/transforms/identifier_folding.py` — the entire validation pass. Self-contained: public `check_folding`, two private detections (`_check_table_folding`, `_check_column_folding`), a column-inventory helper (`_available_columns`), and two error raisers. One clear responsibility; no dependency on the rest of `transforms`.
- **Modify** `fakesnow/cursor.py` — add one import (next to line 29) and one call inside the `execute` loop (after line 180). Nothing else changes.
- **Create** `tests/test_identifier_folding.py` — unit tests against `check_folding` directly (fast, no DB) plus integration tests through the `cur` fixture.

Test command (bare `python` is a broken pyenv shim — always use the venv):

```bash
.venv/bin/python -m pytest tests/test_identifier_folding.py -v
```

---

## Task 1: Validation module + Detection 1 (table/CTE folding)

**Files:**
- Create: `fakesnow/transforms/identifier_folding.py`
- Test: `tests/test_identifier_folding.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_identifier_folding.py`:

```python
from __future__ import annotations

import pytest
import snowflake.connector.errors
from sqlglot import parse_one

from fakesnow.transforms import identifier_folding


def _check(sql: str) -> None:
    identifier_folding.check_folding(parse_one(sql, read="snowflake"))


@pytest.mark.parametrize(
    "sql",
    [
        "WITH ts AS (SELECT 1 AS a) SELECT * FROM ts",
        'WITH "ts" AS (SELECT 1 AS a) SELECT * FROM "ts"',
    ],
)
def test_table_consistent_quoting_passes(sql: str) -> None:
    _check(sql)  # no raise


@pytest.mark.parametrize(
    "sql",
    [
        'WITH ts AS (SELECT 1 AS a) SELECT * FROM "ts"',
        'WITH "ts" AS (SELECT 1 AS a) SELECT * FROM ts',
    ],
)
def test_table_folding_mismatch_raises(sql: str) -> None:
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        _check(sql)


def test_check_does_not_mutate_input() -> None:
    expression = parse_one("WITH ts AS (SELECT 1 AS a) SELECT * FROM ts", read="snowflake")
    before = expression.sql(dialect="snowflake")
    identifier_folding.check_folding(expression)
    assert expression.sql(dialect="snowflake") == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fakesnow.transforms.identifier_folding'`.

- [ ] **Step 3: Write the module (Detection 1 only)**

Create `fakesnow/transforms/identifier_folding.py`:

```python
"""Snowflake identifier case-folding validation.

Real Snowflake folds unquoted identifiers to UPPERCASE and treats double-quoted
identifiers as case-sensitive, so an object defined one way and referenced with
mismatched quoting errors on Snowflake. DuckDB matches identifiers
case-insensitively, so fakesnow would otherwise resolve such a query silently.

This pass detects those intra-query folding mismatches on a *copy* of the parsed
Snowflake AST and raises the Snowflake-style error. It never mutates the
expression that becomes DuckDB SQL, so query results are unchanged --- the only
new behavior is an error on genuinely-mismatched queries.

It raises only when a reference matches an in-scope name case-insensitively but
not exactly (the precise Snowflake-only failure): DuckDB *would* resolve it, but
Snowflake would reject it. A reference with no case-insensitive match falls
through to DuckDB, which raises its own not-found error.
"""

from __future__ import annotations

import logging

import snowflake.connector.errors
from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope

logger = logging.getLogger(__name__)


def check_folding(
    expression: exp.Expression,
    *,
    quoted_identifiers_ignore_case: bool = False,
) -> None:
    """Raise a Snowflake-style ProgrammingError on an identifier-folding mismatch.

    Operates on a copy of `expression`; never mutates it. Fails open: if the
    checker cannot analyze the statement it logs and returns, never blocking a
    query because of its own limitation.
    """
    if quoted_identifiers_ignore_case:
        return
    try:
        ast = expression.copy()
        normalize_identifiers(ast, dialect="snowflake")
        _check_table_folding(ast)
    except snowflake.connector.errors.ProgrammingError:
        raise
    except Exception as e:  # fail-open: never block a query on a checker limitation
        logger.debug("identifier-folding check skipped: %s", e)


def _check_table_folding(ast: exp.Expression) -> None:
    """Detection 1: a table reference collides with an in-scope CTE/derived
    source name case-insensitively but not exactly."""
    for scope in traverse_scope(ast):
        defined = {cte.alias_or_name for cte in scope.ctes}
        for name, source in scope.sources.items():
            if not isinstance(source, exp.Table):
                defined.add(name)
        for name, source in scope.sources.items():
            if isinstance(source, exp.Table):
                ref = source.name
                for known in defined:
                    if ref != known and ref.lower() == known.lower():
                        _raise_object_not_found(ref)


def _raise_object_not_found(ref: str) -> None:
    raise snowflake.connector.errors.ProgrammingError(
        msg=f"SQL compilation error:\nObject '{ref}' does not exist or not authorized.",
        errno=2003,
        sqlstate="42S02",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: PASS (5 tests: 2 pass-cases, 2 raise-cases, 1 no-mutate).

- [ ] **Step 5: Commit**

```bash
git add fakesnow/transforms/identifier_folding.py tests/test_identifier_folding.py
git commit -m "feat(folding): detect table/CTE identifier-folding mismatches"
```

---

## Task 2: Detection 2 (column folding)

**Files:**
- Modify: `fakesnow/transforms/identifier_folding.py`
- Test: `tests/test_identifier_folding.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identifier_folding.py`:

```python
@pytest.mark.parametrize(
    "sql",
    [
        "WITH t AS (SELECT 1 AS period) SELECT period FROM t",
        "WITH ee AS (SELECT 1 AS period) SELECT ee.period FROM ee",
        "SELECT x.period FROM (SELECT 1 AS period) x",
    ],
)
def test_column_consistent_quoting_passes(sql: str) -> None:
    _check(sql)  # no raise


@pytest.mark.parametrize(
    "sql",
    [
        'WITH t AS (SELECT 1 AS period) SELECT "period" FROM t',
        'WITH t AS (SELECT 1 AS "period") SELECT period FROM t',
        'WITH ee AS (SELECT 1 AS "period") SELECT ee.period FROM ee',
        'SELECT x.period FROM (SELECT 1 AS "period") x',
        'WITH ee AS (SELECT 1 AS "period", 5 AS event_encounter_count) '
        "SELECT ee.period, ee.event_encounter_count FROM ee",
    ],
)
def test_column_folding_mismatch_raises(sql: str) -> None:
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        _check(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT foo, bar FROM some_unknown_gold_table",  # unknown base table: columns assumed to exist
        "WITH t AS (SELECT 1 AS a) SELECT xyz FROM t",  # genuine typo, no case-insensitive match
        "WITH ee AS (SELECT 1 AS period) SELECT ee.nope FROM ee",
        "SELECT g.foo FROM gold.t g",
    ],
)
def test_genuine_unknown_columns_fall_through(sql: str) -> None:
    _check(sql)  # no raise; left for DuckDB to handle
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: FAIL — the `test_column_folding_mismatch_raises` cases do **not** raise yet (column detection not implemented).

- [ ] **Step 3: Add column detection to the module**

In `fakesnow/transforms/identifier_folding.py`, add `import re` and the qualify/error imports to the import block, the compiled regex, the call into `check_folding`, and the new functions.

Change the import block to add these three lines:

```python
import re
```

```python
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify
```

Add the module-level regex below `logger = logging.getLogger(__name__)`:

```python
# qualify reports an unresolved column two ways: unqualified refs as
# "Column 'X' could not be resolved", qualified refs (a.b) as "Unknown column: X".
_UNRESOLVED_COLUMN = re.compile(r"Column '([^']+)' could not be resolved|Unknown column: (\S+)")
```

Add the call inside `check_folding`'s `try`, right after `_check_table_folding(ast)`:

```python
        _check_table_folding(ast)
        _check_column_folding(ast)
```

Add the new functions (after `_check_table_folding`):

```python
def _check_column_folding(ast: exp.Expression) -> None:
    """Detection 2: qualify reports an unresolved column that collides with an
    available column on a known source case-insensitively but not exactly."""
    try:
        qualify(ast.copy(), dialect="snowflake", validate_qualify_columns=True)
    except OptimizeError as e:
        match = _UNRESOLVED_COLUMN.search(str(e))
        if not match:
            return  # not a column-resolution failure; leave to DuckDB
        missing = match.group(1) or match.group(2)
        for columns in _available_columns(ast).values():
            for column in columns:
                if missing != column and missing.lower() == column.lower():
                    _raise_invalid_identifier(missing)


def _available_columns(ast: exp.Expression) -> dict[str, set[str]]:
    """Columns exposed by each non-base-table source (CTE/derived), keyed by source name.
    Base tables are skipped: their columns are unknown, so qualify treats them as permissive
    and a missing column there is not a folding collision."""
    columns: dict[str, set[str]] = {}
    for scope in traverse_scope(ast):
        for name, source in scope.sources.items():
            if isinstance(source, exp.Table):
                continue
            inner = getattr(source, "expression", None)
            if isinstance(inner, exp.Select):
                columns.setdefault(name, set()).update(p.alias_or_name for p in inner.selects)
    return columns


def _raise_invalid_identifier(ref: str) -> None:
    raise snowflake.connector.errors.ProgrammingError(
        msg=f"SQL compilation error: invalid identifier '{ref}'",
        errno=904,
        sqlstate="42000",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: PASS (all Task 1 + Task 2 tests; mismatch cases raise, consistent/typo/unknown-base-table cases do not).

- [ ] **Step 5: Commit**

```bash
git add fakesnow/transforms/identifier_folding.py tests/test_identifier_folding.py
git commit -m "feat(folding): detect column/alias identifier-folding mismatches"
```

---

## Task 3: Statement gate, ignore-case skip, fail-open

**Files:**
- Modify: `fakesnow/transforms/identifier_folding.py`
- Test: `tests/test_identifier_folding.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identifier_folding.py`:

```python
@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE foo (id INT)",
        "INSERT INTO foo VALUES (1)",
        "SET my_var = 1",
    ],
)
def test_non_select_statements_skipped(sql: str) -> None:
    _check(sql)  # no analysis, no raise


def test_ignore_case_flag_skips_checks() -> None:
    sql = 'WITH ts AS (SELECT 1 AS a) SELECT * FROM "ts"'  # normally raises (Task 1)
    identifier_folding.check_folding(
        parse_one(sql, read="snowflake"), quoted_identifiers_ignore_case=True
    )  # no raise


def test_simple_select_does_not_raise() -> None:
    _check("SELECT 1")  # nothing to resolve; must not raise from the checker
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py::test_non_select_statements_skipped -v`
Expected: at least one case errors or behaves unexpectedly without the `SELECT` gate (e.g. scope analysis on `SET`/`CREATE`). If they already pass via fail-open, the gate still belongs in for perf — proceed to Step 3.

- [ ] **Step 3: Add the SELECT gate**

In `check_folding`, add the gate as the first statement inside the `try`, before `ast = expression.copy()`:

```python
    try:
        # Only statements that resolve identifiers (contain a SELECT) can have a
        # folding mismatch; skip DDL/DML/session statements cheaply.
        if expression.find(exp.Select) is None:
            return
        ast = expression.copy()
```

(The `quoted_identifiers_ignore_case` short-circuit and the fail-open `except` are already present from Task 1.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: PASS (all tests across Tasks 1-3).

- [ ] **Step 5: Commit**

```bash
git add fakesnow/transforms/identifier_folding.py tests/test_identifier_folding.py
git commit -m "feat(folding): gate to SELECT statements, honor ignore-case, fail open"
```

---

## Task 4: Wire the pass into the cursor

**Files:**
- Modify: `fakesnow/cursor.py` (import near line 29; call inside `execute` loop after line 180)
- Test: `tests/test_identifier_folding.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_identifier_folding.py`:

```python
import snowflake.connector.cursor


def test_table_folding_raises_through_cursor(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        cur.execute('WITH ts AS (SELECT 1 AS a) SELECT * FROM "ts"')
    assert cur.sqlstate == "42S02"


def test_column_folding_raises_through_cursor(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        cur.execute('WITH t AS (SELECT 1 AS period) SELECT "period" FROM t')
    assert cur.sqlstate == "42000"


def test_consistent_query_returns_rows(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("WITH ts AS (SELECT 1 AS a) SELECT a FROM ts")
    assert cur.fetchall() == [(1,)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -k through_cursor -v`
Expected: FAIL — the mismatch queries do not raise yet (the pass is not wired into `execute`).

- [ ] **Step 3: Wire it in**

In `fakesnow/cursor.py`, add the import after line 29 (`import fakesnow.transforms as transforms`):

```python
from fakesnow.transforms import identifier_folding
```

Then add the call inside the `execute` loop (currently lines 180-182). Change:

```python
            for exp in self._transform_explode(expression):
                transformed = self._transform(exp, params)
                self._execute(transformed, params)
```

to:

```python
            for exp in self._transform_explode(expression):
                identifier_folding.check_folding(exp)
                transformed = self._transform(exp, params)
                self._execute(transformed, params)
```

(The existing `except snowflake.connector.errors.ProgrammingError` handler at lines 188-190 already captures `e.sqlstate` into `self._sqlstate` and re-raises, so `cur.sqlstate` is populated.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: PASS (all unit + integration tests).

Also run through the server path to confirm parity:

Run: `TEST_SERVER=1 .venv/bin/python -m pytest tests/test_identifier_folding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fakesnow/cursor.py tests/test_identifier_folding.py
git commit -m "feat(folding): enforce identifier-folding validation in cursor.execute"
```

---

## Task 5: Acceptance suite — spec verification cases + no-false-positive corpus

**Files:**
- Test: `tests/test_identifier_folding.py`

- [ ] **Step 1: Write the acceptance tests**

Append to `tests/test_identifier_folding.py`. These cover the spec's verification list — Snowflake-specific constructs and realistic shapes that must **pass** (proving no false positive), plus the canonical regressions that must **fail**.

```python
@pytest.mark.parametrize(
    "sql",
    [
        # information_schema folds to uppercase and resolves
        "SELECT table_name FROM information_schema.tables",
        # qualified base-table name; unknown columns on unknown schema stay permissive
        'SELECT "PHRASE_ENCOUNTER_V2"."col" FROM "DEVCLIENT"."GOLD"."PHRASE_ENCOUNTER_V2"',
        # window / qualify construct over a CTE, consistent quoting
        "WITH t AS (SELECT 1 AS id, 2 AS n) "
        "SELECT id, n FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY n) = 1",
        # multi-CTE join, all unquoted (the consistent numerator shape)
        "WITH denom AS (SELECT 1 AS period, 10 AS cnt), "
        "num AS (SELECT 1 AS period, 3 AS event_encounter_count) "
        "SELECT d.period, n.event_encounter_count FROM denom d JOIN num n ON d.period = n.period",
    ],
)
def test_snowflake_constructs_pass(sql: str) -> None:
    _check(sql)  # no false positive


@pytest.mark.parametrize(
    "sql",
    [
        # canonical CTE mismatch, both directions
        'WITH time_series AS (SELECT 1 AS d) SELECT * FROM "time_series"',
        'WITH "time_series" AS (SELECT 1 AS d) SELECT * FROM time_series',
        # numerator cross-fragment alias mismatch (the real motivating case)
        'WITH num AS (SELECT 1 AS "period", 3 AS event_encounter_count) '
        "SELECT num.period, num.event_encounter_count FROM num",
    ],
)
def test_canonical_regressions_raise(sql: str) -> None:
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        _check(sql)
```

- [ ] **Step 2: Run the acceptance tests**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -k "constructs_pass or canonical_regressions" -v`
Expected: PASS. If any `test_snowflake_constructs_pass` case raises, that is a false positive — triage before proceeding (the fail-open design means most analysis gaps should *not* raise; a raise means a real collision was detected, so confirm the construct truly is consistent).

- [ ] **Step 3: (Optional) Real production-query corpus replay**

If the real composed SQL from the mainapp-api builders is available (regenerate via the `populationAnalyticsQuery` builder per the spec's feasibility notes, substituting bind placeholders with literals), replay each statement through `check_folding` and assert **zero** raises:

```bash
.venv/bin/python - <<'PY'
import glob
from sqlglot import parse_one
from fakesnow.transforms import identifier_folding

for path in glob.glob("/tmp/real_*query*.parsesafe.sql"):
    with open(path) as f:
        sql = f.read()
    for stmt in sql.split(";"):
        if stmt.strip():
            identifier_folding.check_folding(parse_one(stmt, read="snowflake"))
    print("OK", path)
PY
```

Expected: `OK` for every file (no `ProgrammingError`). This is a manual gate, not a committed test (the corpus is environment-specific).

- [ ] **Step 4: Commit**

```bash
git add tests/test_identifier_folding.py
git commit -m "test(folding): acceptance suite for Snowflake constructs and canonical regressions"
```

---

## Task 6: Full-suite regression + lint/types (Phase A finalize)

**Files:** none (verification only; fix-ups if needed)

- [ ] **Step 1: Run the entire existing test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS with **zero new failures** vs. baseline. The validation pass only adds errors on genuinely-mismatched queries; any *existing* fakesnow test that newly fails indicates either a latent folding bug in a fixture query (fix the fixture) or a checker false positive (fix the checker — most likely tighten a detection or let it fall through). Investigate every new failure; do not suppress.

- [ ] **Step 2: Run pre-commit (lint, format, types)**

Run: `.venv/bin/python -m pre_commit run --all-files` (or `uv run pre-commit run --all-files`)
Expected: PASS. Fix any ruff/pyright findings in the new module/tests.

- [ ] **Step 3: Commit any fix-ups**

```bash
git add -A
git commit -m "chore(folding): lint and type fixes"
```

> **STOP — approval gate.** Phase A (implementation) is complete and committed locally on `snowflake-integration-compat-strict`. **Do not push and do not build/push an image without explicit human approval** (per repo policy: fork pushes and Docker Hub pushes require human sign-off). Report status and wait.

---

## Task 7: Release the image so mainapp-api consumes it (Phase B — approval-gated, operational)

This is what actually makes the change available to mainapp-api. The coupling is a **Docker image** pinned by digest, not a package. There is **no CI automation** that builds/pushes the image — it is manual and needs human-owned Docker Hub credentials.

- [ ] **Step 1: Confirm the build entry point**

Run: `git -C /Users/nathanvecchiarelli/projects/natevecc/fakesnow show HEAD:Dockerfile | head -40` and check `docker-compose.yml` (`natevecc/fakesnow:dev`) and any `Makefile` image target.
Expected: a `python:3.13-slim` image running `uvicorn fakesnow.server:app --host 0.0.0.0 --port 8000`, multi-arch capable.

- [ ] **Step 2: Build and push the multi-arch image (requires Docker Hub login)**

```bash
docker login                      # human credentials for the natevecc/ namespace
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t natevecc/fakesnow:snowflake-integration-compat-strict \
  --push \
  /Users/nathanvecchiarelli/projects/natevecc/fakesnow
```

- [ ] **Step 3: Capture the new manifest-list digest**

```bash
docker buildx imagetools inspect natevecc/fakesnow:snowflake-integration-compat-strict
```

Copy the top-level `Digest: sha256:…` (the manifest **list** digest, multi-arch — not a per-platform digest).

- [ ] **Step 4: Bump the digest in mainapp-api**

Edit `/Users/nathanvecchiarelli/projects/phrase/mainapp-api/.worktrees/fakesnow-mainline/docker/fakesnow-gold/Dockerfile:17`:

```
ARG BASE=natevecc/fakesnow@sha256:<new-manifest-list-digest>
```

Commit in the mainapp-api worktree (its own repo/PR — e.g. PR #1421), run mainapp-api's fakesnow-backed integration tests against the new image, and confirm green. This unblocks removing the downstream workarounds (the 4 raw-SQL folding asserts; the `.or(z.coerce.number())` cleanup is the separate, already-shipped GAP-2).

---

## Task 8: Upstream to tekumara/fakesnow (Phase C — optional, decoupled)

The validation pass is generic Snowflake fidelity and uses only stock sqlglot optimizer APIs (no dependency on the custom `natevecc/sqlglot` pin), so it is a clean upstream candidate. Independent of Phase B; do anytime after Phase A.

- [ ] **Step 1: Create a focused branch off upstream**

```bash
git -C /Users/nathanvecchiarelli/projects/natevecc/fakesnow fetch upstream
git -C /Users/nathanvecchiarelli/projects/natevecc/fakesnow switch -c folding-validation-upstream upstream/main
```

- [ ] **Step 2: Cherry-pick only the folding commits**

```bash
git cherry-pick <Task1-sha> <Task2-sha> <Task3-sha> <Task4-sha> <Task5-sha> <Task6-sha>
```

Resolve any conflicts (the cursor hook is the only contended spot). Confirm the module needs nothing from the fork's sqlglot pin: `git -C /Users/nathanvecchiarelli/projects/natevecc/fakesnow grep -n "natevecc/sqlglot" pyproject.toml` is unrelated to this feature.

- [ ] **Step 3: Verify on stock upstream**

Run: `.venv/bin/python -m pytest tests/test_identifier_folding.py -v` on the upstream branch (after `uv sync`). Expected: PASS. If anything depends on fork-only behavior, reduce the change until it passes on stock upstream.

- [ ] **Step 4: Open the PR (requires human approval to push)**

Push `folding-validation-upstream` to `origin` (the fork) and open a PR against `tekumara/fakesnow:main`, following upstream conventions (conventional-commit titles for release-please, pass `ci.yml`). Describe the Snowflake folding semantics, the detect-and-raise approach, and the zero-false-positive design. **Do not push without explicit human approval.**

---

## Self-Review

**Spec coverage:**
- Goal "mismatch raises a Snowflake-style error" → Tasks 1, 2, 4. ✓
- "Consistent quoting continues to pass" → Task 1/2 pass-cases, Task 5. ✓
- "Zero false positives on the real query surface (incl. unknown gold tables)" → `test_genuine_unknown_columns_fall_through`, `test_snowflake_constructs_pass`, Task 5 optional corpus replay, Task 6 full-suite. ✓
- "SQL emitted to DuckDB unchanged" → copy-only design + `test_check_does_not_mutate_input`. ✓
- Hook after parse, not in transform chain → Task 4 (call in loop before `_transform`). ✓
- `normalize_identifiers` first → in `check_folding`. ✓
- Detection 1 (scope collision) → Task 1. Detection 2 (qualify) → Task 2. Shared trigger rule (case-insensitive-but-not-exact) → both `_check_table_folding` and `_check_column_folding`. ✓
- Error behavior (ProgrammingError family) → Decisions §2, Tasks 1/2 raisers, Task 4 sqlstate asserts. ✓
- `QUOTED_IDENTIFIERS_IGNORE_CASE` awareness → Decisions §4 + `quoted_identifiers_ignore_case` arg + Task 3 test. ✓ (Connection-state plumbing deferred as YAGNI — documented.)
- Fail-open conservatism → Task 1 `except Exception` + Task 3 tests. ✓
- Edge cases (qualified names, information_schema) → Task 5. ✓
- Out-of-scope (base-table columns) → `_available_columns` skips `exp.Table`; `test_genuine_unknown_columns_fall_through` proves base-table columns stay permissive. ✓
- Verification layers (regression suite, corpus replay, differential) → Tasks 5/6 (differential vs real Snowflake noted as optional, needs creds). ✓
- Downstream impact / lifecycle (image, upstream) → Tasks 7/8. ✓

**Placeholder scan:** No TBD/TODO/"add error handling"; all code blocks complete; the only `<…>` placeholders are git SHAs and the new image digest in operational Tasks 7-8, which are runtime values by nature. ✓

**Type/name consistency:** Public function is `check_folding` everywhere (module, cursor call, tests). Private helpers `_check_table_folding`, `_check_column_folding`, `_available_columns`, `_raise_object_not_found`, `_raise_invalid_identifier` defined in Tasks 1-2 and referenced consistently. Regex `_UNRESOLVED_COLUMN` with two alternation groups; `missing = match.group(1) or match.group(2)`. ✓
