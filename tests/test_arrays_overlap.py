"""Tests for ARRAYS_OVERLAP semantics matching real Snowflake VARIANT typed equality.

Snowflake's VARIANT stores both the value AND the runtime type. ARRAYS_OVERLAP
compares elements with strict typed equality (no implicit string<->number coercion)
and NULL-safe equality (NULL elements match NULL elements).

References:
- https://docs.snowflake.com/en/sql-reference/functions/arrays_overlap
- https://docs.snowflake.com/en/sql-reference/data-types-semistructured
"""

from __future__ import annotations

import snowflake.connector


def test_heterogeneous_variant_types_do_not_overlap(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """ARRAYS_OVERLAP(['123'], [123]) is FALSE — VARIANT stores type alongside value."""
    cur.execute("SELECT ARRAYS_OVERLAP(PARSE_JSON('[\"123\"]'), PARSE_JSON('[123]'))")
    assert cur.fetchone() == (False,)


def test_heterogeneous_disjoint_values_do_not_overlap(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("SELECT ARRAYS_OVERLAP(PARSE_JSON('[\"M\"]'), PARSE_JSON('[1]'))")
    assert cur.fetchone() == (False,)


def test_homogeneous_int_match_overlaps(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("SELECT ARRAYS_OVERLAP(PARSE_JSON('[123]'), PARSE_JSON('[123]'))")
    assert cur.fetchone() == (True,)


def test_homogeneous_string_match_overlaps(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("SELECT ARRAYS_OVERLAP(PARSE_JSON('[\"abc\"]'), PARSE_JSON('[\"abc\"]'))")
    assert cur.fetchone() == (True,)


def test_mixed_array_one_typed_match_overlaps(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """Mixed-type left array overlaps right iff at least one element matches both value AND type."""
    cur.execute("SELECT ARRAYS_OVERLAP(PARSE_JSON('[1, \"1\"]'), PARSE_JSON('[\"1\"]'))")
    assert cur.fetchone() == (True,)


def test_null_null_overlaps_nullsafe(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """Snowflake ARRAYS_OVERLAP is NULL-safe: [NULL] vs [NULL] is TRUE."""
    cur.execute("SELECT ARRAYS_OVERLAP(PARSE_JSON('[null]'), PARSE_JSON('[null]'))")
    assert cur.fetchone() == (True,)


def test_null_safe_mixed(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """Example 5 from the Snowflake docs: [1, 2, NULL] overlaps [3, NULL, 5] via the shared NULL."""
    cur.execute(
        "SELECT ARRAYS_OVERLAP(ARRAY_CONSTRUCT(1, 2, NULL), ARRAY_CONSTRUCT(3, NULL, 5))"
    )
    assert cur.fetchone() == (True,)


def test_typed_int_array_construct_overlap(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("SELECT ARRAYS_OVERLAP(ARRAY_CONSTRUCT(1, 2, 3), ARRAY_CONSTRUCT(3, 4))")
    assert cur.fetchone() == (True,)


def test_typed_int_array_construct_no_overlap(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("SELECT ARRAYS_OVERLAP(ARRAY_CONSTRUCT(1, 2), ARRAY_CONSTRUCT(3, 4))")
    assert cur.fetchone() == (False,)


def test_empty_arrays_do_not_overlap(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur.execute("SELECT ARRAYS_OVERLAP(ARRAY_CONSTRUCT(), ARRAY_CONSTRUCT())")
    assert cur.fetchone() == (False,)


def test_null_array_argument_returns_null(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """ARRAYS_OVERLAP with a NULL array argument returns NULL per Snowflake docs."""
    cur.execute("SELECT ARRAYS_OVERLAP(NULL, ARRAY_CONSTRUCT(1))")
    assert cur.fetchone() == (None,)


def test_arrays_overlap_in_where_clause(cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """ARRAYS_OVERLAP used as a predicate (real-world cohort filter shape)."""
    cur.execute(
        """
        WITH t AS (
            SELECT ARRAY_CONSTRUCT(1, 2, 3) AS ids
            UNION ALL
            SELECT ARRAY_CONSTRUCT(4, 5, 6)
        )
        SELECT COUNT(*) FROM t WHERE ARRAYS_OVERLAP(ids, ARRAY_CONSTRUCT(3, 99))
        """
    )
    assert cur.fetchone() == (1,)
