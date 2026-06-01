from __future__ import annotations

import pytest
import snowflake.connector.cursor
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


@pytest.mark.parametrize(
    "sql",
    [
        'WITH foo AS (SELECT 1 AS a) SELECT * FROM gold."foo"',  # qualified ref cannot name a CTE
        'SELECT * FROM (SELECT 1 AS a) AS "tbl", tbl',  # derived alias is not a CTE
        "WITH x AS (SELECT 1 AS a) SELECT * FROM x JOIN other ON x.a = other.a",  # distinct base table
    ],
)
def test_table_no_false_positive_on_non_cte_collisions(sql: str) -> None:
    _check(sql)  # no raise


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
    sql = 'WITH ts AS (SELECT 1 AS a) SELECT * FROM "ts"'  # mismatched quoting; would raise without the flag
    identifier_folding.check_folding(parse_one(sql, read="snowflake"), quoted_identifiers_ignore_case=True)  # no raise


def test_simple_select_does_not_raise() -> None:
    _check("SELECT 1")  # nothing to resolve; must not raise from the checker


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


@pytest.mark.parametrize(
    "sql",
    [
        # genuine missing column (DuckDB errors too); a sibling CTE's quoted column must
        # not contaminate the check for a reference targeting a different source
        'WITH t1 AS (SELECT 1 AS "period"), t2 AS (SELECT 2 AS x) SELECT t2.PERIOD FROM t2',
        'WITH t1 AS (SELECT 1 AS "period"), t2 AS (SELECT 2 AS x) SELECT period FROM t2',
    ],
)
def test_column_no_false_positive_across_sources(sql: str) -> None:
    _check(sql)  # no raise; the missing column is genuinely absent from the referenced source
