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
import re

import snowflake.connector.errors
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

logger = logging.getLogger(__name__)

# qualify reports an unresolved column two ways: unqualified refs as
# "Column 'X' could not be resolved", qualified refs (a.b) as "Unknown column: X".
_UNRESOLVED_COLUMN = re.compile(r"Column '([^']+)' could not be resolved|Unknown column: (\S+)")


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
        _check_column_folding(ast)
    except snowflake.connector.errors.ProgrammingError:
        raise
    except Exception as e:  # fail-open: never block a query on a checker limitation
        logger.debug("identifier-folding check skipped: %s", e)


def _check_table_folding(ast: exp.Expression) -> None:
    """Detection 1: a bare (unqualified) table reference collides with an
    in-scope CTE name case-insensitively but not exactly. CTEs are the only
    construct referenced by bare name, so the check keys strictly off CTE names;
    qualified references (which cannot name a CTE) and empty names are skipped."""
    for scope in traverse_scope(ast):
        cte_names = {cte.alias_or_name for cte in scope.ctes}
        if not cte_names:
            continue
        for source in scope.sources.values():
            if not isinstance(source, exp.Table):
                continue
            if source.db or source.catalog:
                continue  # a qualified reference cannot name a CTE
            ref = source.name
            if not ref:
                continue
            for cte_name in cte_names:
                if ref != cte_name and ref.lower() == cte_name.lower():
                    _raise_object_not_found(ref)


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
                # skip unnamed projections (e.g. SELECT *), whose alias_or_name is empty
                columns.setdefault(name, set()).update(
                    p.alias_or_name for p in inner.selects if p.alias_or_name
                )
    return columns


def _raise_object_not_found(ref: str) -> None:
    raise snowflake.connector.errors.ProgrammingError(
        msg=f"SQL compilation error:\nObject '{ref}' does not exist or not authorized.",
        errno=2003,
        sqlstate="42S02",
    )


def _raise_invalid_identifier(ref: str) -> None:
    raise snowflake.connector.errors.ProgrammingError(
        msg=f"SQL compilation error: invalid identifier '{ref}'",
        errno=904,
        sqlstate="42000",
    )
