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
