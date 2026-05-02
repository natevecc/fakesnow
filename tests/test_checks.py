import sqlglot

from fakesnow.checks import is_unqualified_table_expression


def test_check_unqualified_select() -> None:
    assert is_unqualified_table_expression(sqlglot.parse_one("SELECT * FROM customers")) == (True, True)

    assert is_unqualified_table_expression(sqlglot.parse_one("SELECT * FROM jaffles.customers")) == (True, False)

    assert is_unqualified_table_expression(sqlglot.parse_one("SELECT * FROM marts.jaffles.customers")) == (False, False)


def test_check_unqualified_create_table() -> None:
    assert is_unqualified_table_expression(sqlglot.parse_one("CREATE TABLE customers (ID INT)")) == (True, True)

    assert is_unqualified_table_expression(sqlglot.parse_one("CREATE TABLE jaffles.customers (ID INT)")) == (
        True,
        False,
    )


def test_check_unqualified_drop_table() -> None:
    assert is_unqualified_table_expression(sqlglot.parse_one("DROP TABLE customers")) == (True, True)

    assert is_unqualified_table_expression(sqlglot.parse_one("DROP TABLE jaffles.customers")) == (
        True,
        False,
    )


def test_check_unqualified_schema() -> None:
    # assert is_unqualified_table_expression(sqlglot.parse_one("CREATE SCHEMA jaffles")) == (True, False)

    # assert is_unqualified_table_expression(sqlglot.parse_one("CREATE SCHEMA marts.jaffles")) ==  (False, False)

    assert is_unqualified_table_expression(sqlglot.parse_one("USE SCHEMA jaffles")) == (True, False)

    assert is_unqualified_table_expression(sqlglot.parse_one("USE SCHEMA marts.jaffles")) == (False, False)


def test_check_unqualified_database() -> None:
    assert is_unqualified_table_expression(sqlglot.parse_one("CREATE DATABASE marts")) == (False, False)

    assert is_unqualified_table_expression(sqlglot.parse_one("USE DATABASE marts")) == (False, False)


def test_check_unqualified_select_cte_uses_base_table() -> None:
    expression = sqlglot.parse_one("WITH cte AS (SELECT * FROM marts.jaffles.customers) SELECT * FROM cte")

    assert is_unqualified_table_expression(expression) == (False, False)


def test_check_cte_with_mixed_qualification_uses_all_base_tables() -> None:
    expression = sqlglot.parse_one(
        "WITH cte AS (SELECT * FROM marts.jaffles.customers JOIN orders ON TRUE) SELECT * FROM cte"
    )

    assert is_unqualified_table_expression(expression) == (True, True)


def test_check_table_function_does_not_require_current_database() -> None:
    expression = sqlglot.parse_one("SELECT * FROM table(flatten([1, 2]))", read="snowflake")

    assert is_unqualified_table_expression(expression) == (False, False)


def test_check_create_view_with_inner_join_qualified_sources() -> None:
    # Regression: dbt-snowflake emits CREATE VIEW ... AS SELECT ... INNER JOIN ...
    # The right-hand-side of a JOIN has parent.kind = 'INNER' (the join type),
    # which is a string but is not a DDL kind. Previously this raised
    # AssertionError("Unexpected parent kind: INNER") in _missing_qualifiers,
    # surfacing as a 500 that dbt-snowflake retries indefinitely.
    sql = (
        "create or replace view TXAUS.gold.inpatient_orders as ("
        "select orders.*, enc_type.encounter_type "
        "from TXAUS.gold.orders_deduped orders "
        "inner join TXAUS.gold.most_recent_inpatient_encounter_type enc_type "
        "on enc_type.encntr_id = orders.encntr_id"
        ")"
    )
    expression = sqlglot.parse_one(sql, read="snowflake")

    assert is_unqualified_table_expression(expression) == (False, False)


def test_check_select_with_unqualified_join_target() -> None:
    # The joined-side table participates in qualifier checks too: if it lacks
    # a database/schema it should be reported, not crash.
    expression = sqlglot.parse_one(
        "SELECT * FROM marts.jaffles.customers c LEFT JOIN orders o ON c.id = o.customer_id",
        read="snowflake",
    )

    assert is_unqualified_table_expression(expression) == (True, True)


def test_check_select_with_partially_qualified_join_target() -> None:
    # Pins the (no_database=True, no_schema=False) edge of the JOIN-side
    # qualifier check. Guards against a future "simplification" of the
    # exp.Join branch that always returns (False, False) or (True, True).
    expression = sqlglot.parse_one(
        "SELECT * FROM marts.jaffles.customers c JOIN jaffles.orders o ON c.id = o.customer_id",
        read="snowflake",
    )

    assert is_unqualified_table_expression(expression) == (True, False)


def test_check_outer_and_cross_join_string_parent_kind() -> None:
    # FULL OUTER and CROSS JOIN both produce parent.kind as a non-DDL string
    # ("OUTER", "CROSS"), the same crash class as the INNER repro. Pin them
    # explicitly to document the bug class is not INNER-specific.
    for sql in [
        "SELECT * FROM a.b.c x FULL OUTER JOIN a.b.d y ON 1=1",
        "SELECT * FROM a.b.c x CROSS JOIN a.b.d y",
    ]:
        assert is_unqualified_table_expression(sqlglot.parse_one(sql, read="snowflake")) == (False, False)


