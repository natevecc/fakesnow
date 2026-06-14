# fakesnow identifier-folding validation pass — design

- **Date:** 2026-05-31
- **Status:** Approved (pending spec review)
- **Repo:** fakesnow fork (`snowflake-integration-compat-strict` branch)
- **Effort:** M

## Problem

Real Snowflake folds **unquoted** identifiers to UPPERCASE and treats **double-quoted** identifiers as case-sensitive (stored verbatim). So `time_series` (unquoted → `TIME_SERIES`) and `"time_series"` (quoted → literal `time_series`) are *different* identifiers, and a query that defines a name one way but references it the other way **errors on real Snowflake** ("invalid identifier" / "object does not exist").

fakesnow runs on DuckDB, which matches identifiers **case-insensitively**. So these mismatches resolve silently — a query that is broken on real Snowflake passes fakesnow's integration tests. mainapp-api currently compensates with brittle raw-SQL string-match unit tests (see [Downstream impact](#downstream-impact)). The goal is to catch these mismatches in the test backend so that compensating assertions can be replaced by real behavioral coverage, and future regressions are caught at integration time.

## Goals

- A query that defines an identifier (CTE/table name, or column/alias) one way and references it with mismatched quoting **raises a Snowflake-style error** through fakesnow.
- Consistent quoting continues to pass.
- **Zero false positives** on the real mainapp-api query surface (including queries that touch unknown-schema gold tables).
- The SQL emitted to DuckDB is **unchanged** — only new errors are added, never altered results.

## Non-goals

- Folding-mismatch detection on **base-table columns** (e.g. `SELECT "patient_id" FROM gold.patients` where the stored column is `PATIENT_ID`). This needs live catalog introspection and is deferred; none of the current downstream workarounds concern base-table columns. See [Out of scope](#out-of-scope).
- Rewriting/normalizing the SQL emitted to DuckDB.
- Full reimplementation of Snowflake's binder.

## Verified background

Snowflake identifier semantics confirmed against official docs (`docs.snowflake.com/en/sql-reference/identifiers-syntax`):

- Unquoted → UPPERCASE; quoted → case-sensitive verbatim; an unquoted identifier equals a capitalized double-quoted one. The rule applies uniformly to tables, views, CTEs, columns, and aliases.
- A quoted-lowercase reference to an unquoted-defined (folded-uppercase) object **errors** — demonstrated: `with "sam" as (select 1) select * from sam;` → "Object 'SAM' does not exist".
- Session/account parameter **`QUOTED_IDENTIFIERS_IGNORE_CASE`** (default **FALSE**) makes double-quoted identifiers fold to uppercase when TRUE, negating the case-sensitivity. The design must honor it.

Why the obvious approaches don't work (all verified empirically against DuckDB v1.5.2, the version fakesnow ships):

- **DuckDB binder won't raise.** DuckDB is case-insensitive *even for quoted identifiers*: `WITH TIME_SERIES AS (...) SELECT * FROM "time_series"` returns rows. So folding the SQL and letting DuckDB choke does not work.
- **No DuckDB setting changes this.** Of 157 settings, only `default_collation` (string *values*) and `preserve_identifier_case` (stored/displayed casing, not *matching*) are relevant; the mismatch passes with `preserve_identifier_case` either true or false. Case-insensitive identifier resolution is a deliberate DuckDB design choice.
- **`normalize_identifiers` alone is insufficient.** It correctly folds unquoted→UPPER and keeps quoted verbatim, but the result still resolves in DuckDB. It is necessary (to get Snowflake-canonical forms for analysis) but not sufficient.

The viable mechanism: use sqlglot's own resolution, which *does* apply Snowflake folding correctly, to **detect** the mismatch and raise — without relying on DuckDB.

## Design

A **validation pass** (not a SQL rewrite). It runs on a **copy** of the parsed Snowflake AST and only ever *raises*; it never mutates the AST that becomes DuckDB SQL. Consequently existing query *results* cannot change — the only new behavior is an error on genuinely-mismatched queries.

### Hook point

New module `fakesnow/transforms/identifier_folding.py`, invoked from `cursor.py` immediately after `parse_one(command, read="snowflake")` (~cursor.py:177), in the same spirit as the existing `checks.py` catalog checks — extending correct quoted/unquoted comparison from catalog-existence to intra-query resolution. It does not join the `.transform()` chain, because it does not transform the emitted SQL.

Both detections run after applying `normalize_identifiers(ast_copy, dialect="snowflake")` so identifiers are in Snowflake-canonical form (unquoted→UPPER, quoted verbatim).

### Trigger rule (shared by both detections)

Raise **if and only if** a reference matches an available in-scope name **case-insensitively but not exactly** under Snowflake folding. This is the precise Snowflake-only failure: condition (a) — a case-insensitive match exists — means DuckDB *would* have resolved it, so we are only ever overriding cases DuckDB silently accepts; condition (b) — no exact match — means real Snowflake *would* reject it. A reference with **no** case-insensitive match at all is left alone: it falls through to DuckDB, which raises its own "not found" error (genuine absence/typo — already correct parity, and no concern of ours). This bounds false positives structurally: anything sqlglot cannot model produces no case-insensitive match, so the pass stays silent.

### Detection 1 — table/CTE name mismatch (scope collision)

Walk scopes (`sqlglot.optimizer.scope.traverse_scope`). A reference that fails to bind under Snowflake folding appears as an unbound `exp.Table` source. Flag when such a `Table` source's name matches an in-scope CTE/derived-source name **case-insensitively but not exactly** (e.g. `time_series` vs `TIME_SERIES`). That collision is precisely the Snowflake-only failure: it would have resolved in DuckDB (case-insensitive) but not in Snowflake.

`qualify` alone does **not** catch this class (it silently aliases the unbound table), which is why this dedicated check exists.

### Detection 2 — column reference mismatch (qualify validation)

Realize the [trigger rule](#trigger-rule-shared-by-both-detections) with a **dual-qualify** check that uses each engine's own resolver as the oracle, rather than a hand-rolled column comparison. On a copy, run `qualify(..., dialect="snowflake", validate_qualify_columns=True)` (case-sensitive). If it resolves, there is no mismatch. If it raises an unresolved-column `OptimizeError`, run `qualify(..., dialect="duckdb", validate_qualify_columns=True)` (case-insensitive) on a copy:

- **DuckDB also rejects it** → the column is genuinely absent (typo, or absent from the referenced source) → fall through to DuckDB.
- **DuckDB resolves it** → Snowflake rejects but DuckDB silently runs it → the folding mismatch → raise.

This is exactly the trigger rule: condition (b) "no exact Snowflake match" is the Snowflake-qualify failure; condition (a) "DuckDB would resolve it" is the DuckDB-qualify success. Because DuckDB's resolver scopes each reference to the source it actually targets, a sibling source's same-named quoted column cannot contaminate the check (the earlier global-union approach raised a false positive there). Columns on unknown-schema base tables stay permissive in *both* engines, so Snowflake-qualify succeeds and legitimate gold-table column references never reach the DuckDB check — confirmed against the real population query. A non-column `OptimizeError` (some unrelated qualify failure) is left to DuckDB. The gate also correctly suppresses non-folding dialect rejections (e.g. ambiguous columns in a `NATURAL`/comma join): if DuckDB rejects for the same structural reason, no folding error is raised.

### Error behavior

Raise the Snowflake-style error mainapp-api would see from real Snowflake — `ProgrammingError`-class, message in the "invalid identifier" / "does not exist" family. The exact fakesnow error class to reuse is an [open item](#open-items).

### `QUOTED_IDENTIFIERS_IGNORE_CASE` awareness

Honor the session parameter (fakesnow already reads/echoes session params — cf. `JS_TREAT_INTEGER_AS_BIGINT` in `server.py`). Default `FALSE` → checks active. If a session sets it `TRUE`, fold quoted identifiers too (or skip the checks) so no mismatch fires — matching Snowflake exactly and providing a per-session kill switch.

### Fail-open conservatism

If the validation pass itself cannot analyze a query (parse quirk, unsupported construct, internal error), it **logs and falls through** — it never blocks a query because of its own limitation. It raises only on a *positive* mismatch detection. This structurally bounds false positives on queries not yet seen.

### Edge cases (verified handled by `normalize_identifiers`)

- **Qualified names** `db.sch.tbl.col` → uppercased component-wise; resolves normally.
- **`information_schema`** → folds to `INFORMATION_SCHEMA`; DuckDB resolves case-insensitively.
- **Gold-schema DDL** is already uppercase, so folding is a no-op there.

## Out of scope

Base-table **column** folding mismatches require knowing each gold table's columns. The faithful way to do that is to introspect DuckDB's live catalog (accurate by construction) and feed the schema to `qualify`. This is deferred ("Option 2′") because no current downstream workaround concerns base-table columns, and it adds a per-query introspection cost plus a residual casing caveat (DuckDB discards whether a DDL identifier was quoted, so it cannot perfectly reconstruct Snowflake's stored casing for non-uppercase objects).

## Verification

1. **Fork-side regression suite** — `(query → pass | fail)` pairs covering: the canonical `time_series` CTE mismatch (both directions), consistent quoting, qualified names, `information_schema`, the numerator column-alias mismatch (`AS "period"` vs unquoted reference), and Snowflake-specific constructs that must *pass* (proving no false positive). Fast; runs every commit.
2. **Production-query corpus replay** — run the SQL mainapp-api's v3 builders emit through the pass; expect **zero** new failures. Any new failure is triaged as a real latent bug or a checker false positive.
3. **Differential vs real Snowflake** (optional, the only true oracle) — run should-pass/should-fail cases against a real Snowflake account; assert accept/reject parity. Requires credentials.

## Downstream impact

This is **cleanup/preventive** — investigation found **no live mismatch** in mainapp-api's current queries (quoting is enforced by construction via `helpers/cohort-columns.ts`). Once this pass is live (and the fakesnow cutover lands), the following raw-SQL workaround asserts — each self-documented as compensating for fakesnow's case-insensitivity — can be replaced by behavioral integration coverage:

- `population-analytics-query.test.ts:146` — `not.toMatch(/"time_series"/)` (table/CTE; Detection 1)
- `population-analytics-query.test.ts:115` — §17 guard `toMatch(/AS\s+"period"/)` (column alias; Detection 2)
- `population-numerator.fragment.test.ts:34` — `toMatch(/\bAS period\b/)` (column alias; Detection 2)
- `alert-population-numerator.fragment.test.ts:14` — `toMatch(/\.period AS period\b/)` (column alias; Detection 2)

This work and those tests currently live in mainapp-api worktrees, pre-production; removals are gated on the fakesnow integration cutover going live.

## Open items

- fakesnow's exact `ProgrammingError`-equivalent class to raise (match what surfaces for DuckDB binder errors today).
- Per-query overhead of running `normalize_identifiers` + `qualify` + scope walk on every statement; measure and decide whether to gate behind a flag if material.
- Confirm mainapp-api's Snowflake account uses the default `QUOTED_IDENTIFIERS_IGNORE_CASE = FALSE`.

## Feasibility evidence

Validated against the real composed population query (and alert variant), placeholders substituted with literals:

| Check | Correct query | Seeded regression |
|-------|---------------|-------------------|
| Detection 1 (table/CTE) | no collision (pass) | catches `FROM "time_series"` collision |
| Detection 2 (columns) | pass — no false positive on unknown gold table | raises `Unknown column: PERIOD` |
