from __future__ import annotations

import datetime
import gzip
import json
import logging
import os
import secrets
from base64 import b64encode
from dataclasses import dataclass
from typing import Any

import snowflake.connector.errors
from sqlglot import parse_one
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from fakesnow.arrow import to_ipc, to_sf
from fakesnow.converter import from_binding
from fakesnow.expr import normalise_ident
from fakesnow.fakes import FakeSnowflakeConnection
from fakesnow.instance import FakeSnow
from fakesnow.rowtype import describe_as_rowtype

# Configure parent logger so all fakesnow.* loggers inherit handlers and level
fakesnow_logger = logging.getLogger("fakesnow")
fakesnow_logger.handlers = logging.getLogger("uvicorn").handlers
if os.environ.get("LOG_LEVEL", "").lower() == "debug":
    fakesnow_logger.setLevel(logging.DEBUG)
else:
    fakesnow_logger.setLevel(logging.INFO)

logger = logging.getLogger("fakesnow.server")


class SafeJSONResponse(JSONResponse):
    """JSONResponse that handles non-serializable types like datetime."""

    def render(self, content: Any) -> bytes:
        return json.dumps(content, default=str).encode("utf-8")

logger.info("Creating shared in-memory database for session")
_shared_db_path = os.environ.get("FAKESNOW_SHARED_DB_PATH")
shared_fs = FakeSnow(db_path=_shared_db_path) if _shared_db_path else FakeSnow()
sessions: dict[str, FakeSnowflakeConnection] = {}
# Per-session SDK-driver toggles captured at login for Node SDK BigInt
# normalization. The Node SDK selects `convertRawBigInt` only when
# `JS_TREAT_INTEGER_AS_BIGINT` is present in the per-query `data.parameters`
# array. The flag is supplied by the client at login under SESSION_PARAMETERS
# but real Snowflake echoes it back on every query response. We mirror that
# here without modifying FakeSnowflakeConnection. Keys are the same login
# tokens used in `sessions`; entries are removed by `session(delete=true)`.
_session_params_by_token: dict[str, dict[str, Any]] = {}


# Snowflake parameter names whose values the SDK reads from `data.parameters`
# on every query response. We echo whatever the client sent at login for these
# (defaulting to False / unset) so SDK behavior matches real Snowflake.
_ECHOED_SESSION_PARAMS = ("JS_TREAT_INTEGER_AS_BIGINT",)


def _build_query_parameters(token: str | None) -> list[dict[str, Any]]:
    """Build the `data.parameters` echo array for a query response.

    Always emits the canonical Snowflake parameters fakesnow has historically
    returned (TIMEZONE), then appends any SDK-driver toggles in
    `_ECHOED_SESSION_PARAMS` that the client actually supplied at login. We
    echo the value verbatim rather than coercing to bool: real Snowflake
    preserves the client's value type, and a forced `bool(...)` on a string
    like "false" would invert the intent (any non-empty string is truthy).

    Parameters the client did not supply are simply not echoed -- before
    BigInt normalization was added, fakesnow only echoed TIMEZONE, so this
    preserves backward compatibility for clients that never opted into
    JS_TREAT_INTEGER_AS_BIGINT.
    """
    params: list[dict[str, Any]] = [{"name": "TIMEZONE", "value": "Etc/UTC"}]
    sp = _session_params_by_token.get(token, {}) if token else {}
    for name in _ECHOED_SESSION_PARAMS:
        if name in sp:
            params.append({"name": name, "value": sp[name]})
    return params


def _stringify_fixed_ints(
    rowset_json: list[list[Any]], rowtype: list[dict[str, Any]]
) -> list[list[Any]]:
    """Stringify integer columns in `rowset_json` per the Snowflake JSON wire
    contract.

    Real Snowflake JSON-encodes `fixed`/scale-0 columns as strings (e.g. `"42"`,
    not bare `42`) so that the Node SDK can call `bigInt(rawColumnValue)` on a
    precision-safe input. Without this, ints > 2^53 lose precision during
    JSON.parse on the Node side -- before the SDK ever sees the value -- and
    the `JS_TREAT_INTEGER_AS_BIGINT` path is unreachable.

    Narrow predicate (`scale == 0` only) by design: matches the bug, leaves
    DECIMAL(N,M) untouched, minimizes blast radius for Python-SDK JSON-path
    consumers.
    """
    fixed_int_cols = [
        i
        for i, c in enumerate(rowtype)
        if c.get("type") == "fixed" and c.get("scale") == 0
    ]
    if not fixed_int_cols:
        return rowset_json
    for row in rowset_json:
        for i in fixed_int_cols:
            if row[i] is not None:
                row[i] = str(row[i])
    return rowset_json


_EPOCH_DATE = datetime.date(1970, 1, 1)


def _stringify_booleans(
    rowset_json: list[list[Any]], rowtype: list[dict[str, Any]]
) -> list[list[Any]]:
    """Encode boolean columns as "TRUE"/"FALSE" strings per the Snowflake JSON
    wire contract used by the Node SDK (see column.js convertRawBoolean).

    Without this, the Node SDK trips on native JS `false` values: the third
    branch of its boolean test calls `.toUpperCase()` on the raw value, which
    is fine for strings but throws TypeError for booleans.
    """
    bool_cols = [i for i, c in enumerate(rowtype) if c.get("type") == "boolean"]
    if not bool_cols:
        return rowset_json
    for row in rowset_json:
        for i in bool_cols:
            value = row[i]
            if value is None:
                continue
            row[i] = "TRUE" if value else "FALSE"
    return rowset_json


def _normalize_temporal_for_json_rowset(
    rowset_json: list[list[Any]], rowtype: list[dict[str, Any]]
) -> list[list[Any]]:
    """Encode date/time/timestamp columns in `rowset_json` per the Snowflake JSON
    wire contract used by the Node SDK.

    Without this, the Node SDK's `convertRawDate` / `convertRawTimestampNtz`
    parse the raw column value via `Number(...)`/`BigNumber(...)`. fakesnow's
    JSON rowset previously emitted `datetime.date` / `datetime.datetime` which
    the response serializer rendered as ISO strings (e.g. "2025-01-01"). The
    Node SDK then coerces the ISO string to NaN, multiplies by 86400 (DATE) or
    a scale factor (TIMESTAMP), and the resulting NaN epoch silently becomes
    1970-01-01. (The Python SDK is unaffected; it consumes `rowsetBase64`,
    not `rowset`.)

    Wire format expected by the Node SDK (see column.js `convertRawDate`,
    `convertRawTimestampNtz` in snowflake-sdk):
      - DATE         : days since Unix epoch as a string, e.g. "20089"
      - TIMESTAMP_NTZ: fractional epoch seconds as a string with `scale`
                       digits of fractional precision, e.g.
                       "1735693261.000000000" (scale=9). This matches the
                       format real Snowflake emits.

    Other temporal types (TIME, TIMESTAMP_LTZ, TIMESTAMP_TZ) are not produced
    by the failing test path and are intentionally left to a follow-up.
    """
    temporal_cols = [
        (i, c.get("type"), c.get("scale") or 0)
        for i, c in enumerate(rowtype)
        if c.get("type") in ("date", "timestamp_ntz")
    ]
    if not temporal_cols:
        return rowset_json
    for row in rowset_json:
        for i, sf_type, scale in temporal_cols:
            value = row[i]
            if value is None:
                continue
            if sf_type == "date" and isinstance(value, datetime.date):
                row[i] = str((value - _EPOCH_DATE).days)
            elif sf_type == "timestamp_ntz" and isinstance(value, datetime.datetime):
                # Use a naive datetime as epoch reference -- arrow_table.to_pylist()
                # yields naive datetimes for TIMESTAMP_NTZ.
                delta = value - datetime.datetime(1970, 1, 1)
                seconds = delta.days * 86400 + delta.seconds
                micros = delta.microseconds
                # Format as fractional seconds with `scale` digits of precision.
                # scale=9 (default for fakesnow) means nanosecond precision; we
                # only have microsecond precision from Python datetime, so the
                # last 3 digits are always 0.
                if scale == 0:
                    row[i] = str(seconds)
                else:
                    # Right-pad microseconds to `scale` digits.
                    frac = f"{micros:06d}".ljust(scale, "0")[:scale]
                    row[i] = f"{seconds}.{frac}"
    return rowset_json


@dataclass
class ServerError(Exception):
    status_code: int
    code: str
    message: str


async def login_request(request: Request) -> JSONResponse:
    database = (d := request.query_params.get("databaseName")) and normalise_ident(d)
    schema = (s := request.query_params.get("schemaName")) and normalise_ident(s)
    body = await request.body()
    if request.headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    body_json = json.loads(body)
    session_params: dict[str, Any] = body_json["data"]["SESSION_PARAMETERS"]
    nop_regexes = session_params.get("nop_regexes")
    autocommit = session_params.get("AUTOCOMMIT", True)

    # Session parameters take precedence over environment variable for db path, this allow you to have some sessions
    # share a database and others use isolated databases
    db_path = session_params.get("FAKESNOW_DB_PATH") or os.environ.get("FAKESNOW_DB_PATH")
    if db_path is None:
        # Use the shared in-memory database. This is shared across all sessions and is cleared when the server restarts.
        # Useful for sharing data between sessions without needing to manage database files.
        logger.info("Using shared in-memory database for session")
        fs = shared_fs
    elif db_path == ":isolated:":
        # Explicitly setting FAKESNOW_DB_PATH = ":isolated:", creates a new isolated database in memory for every login.
        # Connection close is triggered by the context manager when hitting FakeSnowflakeConnection.__exit__()
        # If used outside of a context manager, users will need to manually close the connection when they're done with
        # it to release resources.
        logger.info("Using isolated in-memory database for session")
        fs = FakeSnow()
    else:
        # Use the set value for db_path. This instructs fakesnow to persist databases to the filesystem, making it
        # persistent across server restarts.
        logger.info(f"Using persistent database at {db_path} for session")
        fs = FakeSnow(db_path=db_path)
    token = secrets.token_urlsafe(32)
    # Forward the full session_params dict so connection-level fakesnow-specific
    # toggles reach FakeSnowflakeConnection without requiring a server.py edit per
    # new opt-in. AUTOCOMMIT and nop_regexes are also extracted into named kwargs
    # above; the named kwargs win because FakeSnowflakeConnection reads them from
    # kwargs directly. Other Snowflake-canonical params in the dict (TIMEZONE,
    # STATEMENT_TIMEOUT_IN_SECONDS, etc.) are inert -- FakeSnowflakeConnection
    # ignores keys it doesn't know.
    logger.info(
        f"[LOGIN] database={database} schema={schema} autocommit={autocommit} "
        f"nop_regexes={nop_regexes}"
    )
    sessions[token] = fs.connect(
        database,
        schema,
        nop_regexes=nop_regexes,
        autocommit=autocommit,
        session_parameters=session_params,
    )
    # Stash the raw SESSION_PARAMETERS dict so per-query responses can echo
    # SDK-driver toggles like JS_TREAT_INTEGER_AS_BIGINT back to the client
    # (real Snowflake echoes these on every query). See
    # `_build_query_parameters` for which keys are forwarded.
    _session_params_by_token[token] = session_params
    response = {
        "data": {
            "token": token,
            "parameters": [
                {"name": "AUTOCOMMIT", "value": autocommit},
                {"name": "CLIENT_SESSION_KEEP_ALIVE_HEARTBEAT_FREQUENCY", "value": 3600},
            ],
            "sessionInfo": {
                "databaseName": database,
                "schemaName": schema,
            },
        },
        "success": True,
    }
    return SafeJSONResponse(response)


async def query_request(request: Request) -> JSONResponse:
    try:
        conn = to_conn(to_token(request))
        request_id = request.query_params.get("requestId", "unknown")
        logger.debug(f"[QUERY_REQUEST] START requestId={request_id} host={request.client.host if request.client else 'unknown'}")

        body = await request.body()
        if request.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)

        body_json = json.loads(body)

        sql_text = body_json["sqlText"]
        logger.debug(f"[QUERY_REQUEST] SQL: {sql_text}")

        if bindings := body_json.get("bindings"):
            # Convert parameters like {'1': {'type': 'FIXED', 'value': '10'}, ...} to tuple (10, ...)
            params = tuple(from_binding(bindings[str(pos)]) for pos in range(1, len(bindings) + 1))
            logger.debug(f"[QUERY_REQUEST] Bindings: {params}")
        else:
            params = None

        expr = parse_one(sql_text, read="snowflake")

        try:
            # only a single sql statement is sent at a time by the python snowflake connector
            logger.debug(f"[QUERY_REQUEST] Executing SQL with params={params}")
            cur = await run_in_threadpool(conn.cursor().execute, sql_text, binding_params=params, server=True)
            logger.info(f"[QUERY_REQUEST] SQL execution completed, queryId={cur.sfqid}, rowcount={cur._rowcount}")  # noqa: SLF001

            rowtype = describe_as_rowtype(cur._describe_last_sql())  # noqa: SLF001

            expr = cur._last_transformed  # noqa: SLF001
            assert expr
            if put_stage_data := expr.args.get("put_stage_data"):
                # this is a PUT command, so return the stage data
                logger.info("[QUERY_REQUEST] PUT command detected, returning stage data")
                return SafeJSONResponse(
                    {
                        "data": put_stage_data,
                        "success": True,
                    }
                )

        except snowflake.connector.errors.ProgrammingError as e:
            logger.error(f"[QUERY_REQUEST] ProgrammingError: {sql_text=} errno={e.errno} {e.msg}")
            code = f"{e.errno:06d}"
            response = {
                "data": {
                    "errorCode": code,
                    "sqlState": e.sqlstate,
                },
                "code": code,
                "message": e.msg,
                "success": False,
            }
            logger.error(f"[QUERY_REQUEST] Returning error response: code={code}")
            return SafeJSONResponse(response)
        except Exception as e:
            # we have a bug or use of an unsupported feature
            msg = f"{sql_text=} {params=} Unhandled exception"
            logger.error(f"[QUERY_REQUEST] {msg}", exc_info=e)
            # my guess at mimicking a 500 error as per https://docs.snowflake.com/en/developer-guide/sql-api/reference
            # and https://github.com/snowflakedb/gosnowflake/blob/8ed4c75ffd707dd712ad843f40189843ace683c4/restful.go#L318
            raise ServerError(status_code=500, code="261000", message=msg) from None

        if cur._arrow_table:  # noqa: SLF001
            batch_bytes = to_ipc(to_sf(cur._arrow_table, rowtype))  # noqa: SLF001
            rowset_b64 = b64encode(batch_bytes).decode("utf-8")
            # Convert arrow table to array of arrays for Node.js SDK
            # SDK expects [[val1, val2], [val1, val2], ...], not [{col: val}, ...]
            rowset_json = [list(row.values()) for row in cur._arrow_table.to_pylist()]  # noqa: SLF001
            # Stringify scale-0 fixed numerics so the Node SDK can call
            # bigInt(rawColumnValue) without losing precision past 2^53.
            rowset_json = _stringify_fixed_ints(rowset_json, rowtype)
            # Encode date/timestamp values as numeric strings the Node SDK's
            # convertRawDate/convertRawTimestampNtz expect (see helper docstring).
            rowset_json = _normalize_temporal_for_json_rowset(rowset_json, rowtype)
            rowset_json = _stringify_booleans(rowset_json, rowtype)
            logger.debug(f"[QUERY_REQUEST] Arrow table: {len(rowset_json)} rows, rowset_b64 length={len(rowset_b64)}")
        else:
            rowset_b64 = ""
            rowset_json = []
            logger.debug("[QUERY_REQUEST] No arrow table, empty result")

        # Cache the result data (limit to 50 most recent)
        cache_data = {
            "parameters": _build_query_parameters(to_token(request)),
            "rowtype": rowtype,
            "rowsetBase64": rowset_b64,  # For Python SDK
            "rowset": rowset_json,  # For Node.js SDK
            "total": cur._rowcount,  # noqa: SLF001
            "returned": cur._rowcount,  # noqa: SLF001  # Node.js SDK needs this
            "queryId": cur.sfqid,
            "queryResultFormat": "arrow",
            "version": 1,  # Node.js SDK requires version field
            "chunks": [],  # Node.js SDK expects chunks field (empty list, not None)
            "finalDatabaseName": conn.database,
            "finalSchemaName": conn.schema,
        }

        # Store in cache, maintaining max 50 entries (LRU)
        # Store internal tuple format expected by result_scan() and get_results_from_sfqid()
        sfqid = cur.sfqid
        if sfqid is None:
            raise ServerError(status_code=500, code="261001", message="Missing query id after execution")
        conn.results_cache[sfqid] = (
            cur._arrow_table,  # noqa: SLF001
            cur._rowcount,  # noqa: SLF001
            cur._last_sql,  # noqa: SLF001
            cur._last_params,  # noqa: SLF001
            cur._last_transformed,  # noqa: SLF001
            rowtype,  # rowtype needed for get_cached_query_result()
        )
        if len(conn.results_cache) > 50:
            conn.results_cache.popitem(last=False)  # Remove oldest item
        logger.debug(f"[QUERY_REQUEST] Cached result for queryId={cur.sfqid}, cache size={len(conn.results_cache)}")

        # Return cache_data with both rowset and rowsetBase64
        response = {
            "data": cache_data,
            "code": "0",  # 0 = Success, results ready immediately
            "success": True,
        }
        logger.debug(f"[QUERY_REQUEST] END requestId={request_id} queryId={cur.sfqid} rows={cur._rowcount} status=success code=0")  # noqa: SLF001
        return SafeJSONResponse(response)

    except ServerError as e:
        logger.error(f"[QUERY_REQUEST] ServerError: code={e.code} message={e.message}")
        return SafeJSONResponse(
            {"data": None, "code": e.code, "message": e.message, "success": False, "headers": None},
            status_code=e.status_code,
        )

async def get_cached_query_result(request: Request) -> JSONResponse:
    try:
        token = to_token(request)
        conn = to_conn(token)

        # Extract query_id from path: /queries/{query_id}/result
        query_id = request.path_params.get("query_id")
        request_guid = request.query_params.get("request_guid", "unknown")
        disable_offline_chunks = request.query_params.get("disableOfflineChunks", "unknown")

        logger.info(f"[GET_RESULT] START query_id={query_id} request_guid={request_guid} disableOfflineChunks={disable_offline_chunks} client={request.client.host if request.client else 'unknown'}")

        if not query_id:
            logger.error("[GET_RESULT] Missing query_id in request path")
            raise ServerError(status_code=400, code="002003", message="Missing query_id in request path")

        # Retrieve from cache
        cached_tuple = conn.results_cache.get(query_id)

        if not cached_tuple:
            logger.error(f"[GET_RESULT] Query results not found for query_id={query_id}, cache keys: {list(conn.results_cache.keys())}")
            raise ServerError(status_code=404, code="000604", message=f"Query results not found for query_id: {query_id}")

        # Unpack the cached tuple format (arrow_table, rowcount, last_sql, last_params, last_transformed, rowtype)
        arrow_table, rowcount, _, _, _, rowtype = cached_tuple

        # Reconstruct response data from cached tuple
        if arrow_table:
            batch_bytes = to_ipc(to_sf(arrow_table, rowtype))
            rowset_b64 = b64encode(batch_bytes).decode("utf-8")
            rowset_json = [list(row.values()) for row in arrow_table.to_pylist()]
            # Stringify scale-0 fixed numerics for the Node SDK BigInt path
            # (mirrors the fresh-query path in `query_request`).
            rowset_json = _stringify_fixed_ints(rowset_json, rowtype)
            # Encode date/timestamp values for the Node SDK
            # (mirrors the fresh-query path in `query_request`).
            rowset_json = _normalize_temporal_for_json_rowset(rowset_json, rowtype)
            rowset_json = _stringify_booleans(rowset_json, rowtype)
        else:
            rowtype = []
            rowset_b64 = ""
            rowset_json = []

        has_rowset_b64 = bool(rowset_b64)
        rowset_count = len(rowset_json)

        logger.debug(f"[GET_RESULT] Found cached result: rowtype={rowtype}, rows={rowset_count}/{rowcount}, has_rowset_b64={has_rowset_b64}")

        cached_result = {
            "parameters": _build_query_parameters(token),
            "rowtype": rowtype,
            "rowsetBase64": rowset_b64,
            "rowset": rowset_json,
            "total": rowcount,
            "returned": rowcount,
            "queryId": query_id,
            "queryResultFormat": "arrow",
            "version": 1,
            "chunks": [],
            "finalDatabaseName": conn.database,
            "finalSchemaName": conn.schema,
        }

        # For GET /queries/{query_id}/result endpoint:
        # Return cached result with both rowset (JSON) and rowsetBase64 (Arrow)
        response = {
            "data": cached_result,
            "code": "0",  # 0 = Success, results ready immediately
            "success": True,
        }
        logger.debug(f"[GET_RESULT] END query_id={query_id} status=success code=0 rows={rowset_count}")
        return SafeJSONResponse(response)

    except ServerError as e:
        logger.error(f"[GET_RESULT] ServerError: code={e.code} message={e.message}")
        return SafeJSONResponse(
            {"data": None, "code": e.code, "message": e.message, "success": False, "headers": None},
            status_code=e.status_code,
        )


def to_token(request: Request) -> str:
    if not (auth := request.headers.get("Authorization")):
        logger.error("[AUTH] Authorization header not found")
        raise ServerError(status_code=401, code="390101", message="Authorization header not found in the request data.")

    token = auth[17:-1]
    logger.debug("[AUTH] Token extracted from Authorization header")
    return token


def to_conn(token: str) -> FakeSnowflakeConnection:
    if not (conn := sessions.get(token)):
        logger.error(f"[AUTH] Session not found for token, available sessions: {len(sessions)}")
        raise ServerError(status_code=401, code="390104", message="User must login again to access the service.")

    logger.debug(f"[AUTH] Session found, database={conn.database} schema={conn.schema}")
    return conn


async def session(request: Request) -> JSONResponse:
    try:
        token = to_token(request)
        _ = to_conn(token)

        if bool(request.query_params.get("delete")):
            logger.info("[SESSION] DELETE session")
            try:
                sessions[token]._duck_conn.close()  # Close the duckdb connection to release resources
            finally:
                sessions.pop(token, None)
                # Drop the session-params shadow entry (used to echo SDK
                # toggles like JS_TREAT_INTEGER_AS_BIGINT on every query) so
                # we don't leak per-token state across the server lifetime.
                _session_params_by_token.pop(token, None)
        else:
            logger.debug("[SESSION] HEARTBEAT")

        return SafeJSONResponse(
            {"data": None, "code": None, "message": None, "success": True},
        )

    except ServerError as e:
        logger.error(f"[SESSION] ServerError: code={e.code} message={e.message}")
        return SafeJSONResponse(
            {"data": None, "code": e.code, "message": e.message, "success": False, "headers": None},
            status_code=e.status_code,
        )


async def health(request: Request) -> JSONResponse:
    try:
        cur = shared_fs.duck_conn.execute("SELECT 1")
        cur.fetchone()
        return SafeJSONResponse({"status": "ok"})
    except Exception as e:
        return SafeJSONResponse({"status": "degraded", "error": str(e)[:200]}, status_code=503)


def monitoring_query(request: Request) -> JSONResponse:
    try:
        token = to_token(request)
        conn = to_conn(token)

        sfqid = request.path_params["sfqid"]
        if not conn.results_cache.get(sfqid):
            logger.debug(f"[MONITORING] query {sfqid} not found in cache")
            return SafeJSONResponse({"data": {"queries": []}, "success": True})

        logger.debug(f"[MONITORING] query {sfqid} status=SUCCESS")
        return SafeJSONResponse({"data": {"queries": [{"status": "SUCCESS"}]}, "success": True})
    except ServerError as e:
        logger.error(f"[MONITORING] ServerError: code={e.code} message={e.message}")
        return SafeJSONResponse(
            {"data": None, "code": e.code, "message": e.message, "success": False, "headers": None},
            status_code=e.status_code,
        )


routes = [
    Route(
        "/session/v1/login-request",
        login_request,
        methods=["POST"],
    ),
    Route("/session", session, methods=["POST"]),
    Route(
        "/queries/v1/query-request",
        query_request,
        methods=["POST"],
    ),
    Route(
        "/queries/{query_id}/result",
        get_cached_query_result,
        methods=["GET"],
    ),
    Route("/queries/v1/abort-request", lambda _: SafeJSONResponse({"success": True}), methods=["POST"]),
    Route("/monitoring/queries/{sfqid}", monitoring_query, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
]

app = Starlette(debug=False, routes=routes)
