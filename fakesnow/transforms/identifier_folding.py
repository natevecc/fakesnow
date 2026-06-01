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


def _raise_object_not_found(ref: str) -> None:
    raise snowflake.connector.errors.ProgrammingError(
        msg=f"SQL compilation error:\nObject '{ref}' does not exist or not authorized.",
        errno=2003,
        sqlstate="42S02",
    )
