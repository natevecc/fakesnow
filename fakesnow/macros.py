from string import Template

# emulate the Snowflake FLATTEN function for ARRAYs and OBJECTTs
# see https://docs.snowflake.com/en/sql-reference/functions/flatten.html
FS_FLATTEN = Template(
    """
CREATE OR REPLACE MACRO ${catalog}._fs_flatten(input) AS TABLE
    SELECT
        -- SEQ: hash of input gives same value for all rows from same input, close enough to Snowflake's SEQ
        hash(TO_JSON(input))::UBIGINT AS SEQ,
        e.k AS KEY,
        COALESCE(e.k, '[' || (row_number() OVER () - 1) || ']') AS PATH,
        CASE WHEN e.k IS NOT NULL THEN NULL ELSE (row_number() OVER () - 1)::BIGINT END AS INDEX,
        e.v AS VALUE,
        TO_JSON(input) AS THIS
    FROM (
        SELECT UNNEST(
            CASE WHEN json_type(TO_JSON(input)) = 'OBJECT'
                 THEN list_transform(
                    json_keys(TO_JSON(input)),
                    x -> struct_pack(k := x, v := CAST(TO_JSON(input) -> x AS JSON))
                 )
                 ELSE list_transform(
                    CAST(TO_JSON(input) AS JSON[]),
                    x -> struct_pack(k := NULL::VARCHAR, v := x)
                 )
            END, recursive := true
        )
    ) AS e(k, v)
    """
)

# use json_group_object instead of json_object because it filters out keys that are null
# see https://github.com/duckdb/duckdb/issues/19357
FS_OBJECT_CONSTRUCT = Template(
    """
CREATE OR REPLACE MACRO ${catalog}._fs_object_construct(keys, vals, keep_nulls) AS (
    WITH kv AS (
        SELECT
            key,
            list_extract(vals, idx) AS value
        FROM UNNEST(keys) WITH ORDINALITY AS u(key, idx)
        ORDER BY idx
    )
    SELECT json_group_object(key, value) AS obj
    FROM kv
    WHERE keep_nulls OR value IS NOT NULL
);
"""
)

FS_TO_TIMESTAMP = Template(
    """
CREATE OR REPLACE MACRO ${catalog}._fs_to_timestamp(val, scale) AS (
    CASE
        WHEN try_cast(val AS BIGINT) IS NOT NULL
            THEN
                CASE
                    WHEN scale = 0 THEN cast(to_timestamp(val::BIGINT) as TIMESTAMP)
                    WHEN scale = 3 THEN cast(to_timestamp(val::BIGINT / 1000) as TIMESTAMP)
                    WHEN scale = 6 THEN cast(to_timestamp(val::BIGINT / 1000000) as TIMESTAMP)
                    WHEN scale = 9 THEN cast(to_timestamp(val::BIGINT / 1000000000) as TIMESTAMP)
                    ELSE NULL
                END
        ELSE CAST(val AS TIMESTAMP)
    END
);
"""
)

# Snowflake's CURRENT_WAREHOUSE() scalar — DuckDB has no native equivalent and
# fakesnow does not (yet) track the per-session warehouse on the connection. We
# register an unqualified macro per-attached-catalog so an unqualified call
# resolves while the session has any of fakesnow's databases on its search path
# (the same per-catalog registration approach the _fs_* macros use). Returns a
# constant 'COMPUTE_WH' to satisfy dbt-snowflake's pre-DDL probe for
# `dynamic_table` materializations (16 cerner_transforms models depend on
# this). See docs/decisions/2026-04-26-fakesnow-fifth-fix.md
# (b7-current-warehouse).
#
# NOTE on naming divergence: unlike `_fs_*` macros which are internal
# implementation helpers invoked by sqlglot transforms in
# transforms/transforms.py, this macro takes its user-facing Snowflake name
# (`current_warehouse`) directly so unmodified Snowflake SQL resolves without
# a transform pass. Future refactors that grep `_fs_*` to enumerate
# fakesnow's DuckDB shims should be aware of this deliberate exception.
#
# Known limitations (deferred to a follow-up that may touch conn.py):
#   - Returns the constant 'COMPUTE_WH' regardless of what was passed to
#     `snowflake.connector.connect(warehouse=...)`. Code that branches on
#     warehouse identity will see 'COMPUTE_WH' for every session.
#   - Resolves only when the session schema is set to a fakesnow-attached
#     catalog. `connect()` with no database leaves the session in DuckDB's
#     default `memory` catalog, where this macro is not registered, so
#     `current_warehouse()` raises the original Catalog Error in that path.
#     Real Snowflake returns the warehouse (or NULL) regardless of database.
# TODO(phrase-fork): on upgrade, accept `warehouse` kwarg in conn.py, store
# `self.warehouse = (kwarg or 'COMPUTE_WH').upper()`, set a duckdb session
# variable via parameterised `execute("SET VARIABLE fakesnow_warehouse = ?", [...])`,
# and rewrite the macro body to
# `COALESCE(GETVARIABLE('fakesnow_warehouse'), 'COMPUTE_WH')`.
FS_CURRENT_WAREHOUSE = Template(
    """
CREATE OR REPLACE MACRO ${catalog}.current_warehouse() AS 'COMPUTE_WH';
"""
)


def creation_sql(catalog: str) -> str:
    return f"""
        {FS_FLATTEN.substitute(catalog=catalog)};
        {FS_OBJECT_CONSTRUCT.substitute(catalog=catalog)};
        {FS_TO_TIMESTAMP.substitute(catalog=catalog)};
        {FS_CURRENT_WAREHOUSE.substitute(catalog=catalog)};
    """
