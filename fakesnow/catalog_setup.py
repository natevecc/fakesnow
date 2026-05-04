"""Shared post-attach setup for newly attached DuckDB catalogs.

Whenever a fresh catalog is attached to fakesnow's DuckDB connection
(via FakeSnow startup, FakeSnowflakeConnection construction, or a SQL
`CREATE DATABASE` issued through a cursor) we must install the same
per-catalog scaffolding: the info-schema extension views and the
`_fs_*` macros that fakesnow's transforms emit unqualified calls to.

Centralising this in one helper makes the asymmetry impossible: every
attach site goes through `post_attach_setup` and gets identical
treatment.
"""

from __future__ import annotations

from duckdb import DuckDBPyConnection

from fakesnow import info_schema, macros


def post_attach_setup(duck_conn: DuckDBPyConnection, catalog_name: str) -> None:
    """Install per-catalog info-schema views and `_fs_*` macros for a
    newly-attached catalog.

    Called from every site that attaches a catalog: FakeSnow startup
    (re-attaching db_path files), FakeSnowflakeConnection construction
    (when a connection's database doesn't yet exist), and the cursor's
    SQL `CREATE DATABASE` handler.
    """
    duck_conn.execute(info_schema.per_db_creation_sql(catalog_name))
    duck_conn.execute(macros.creation_sql(catalog_name))
