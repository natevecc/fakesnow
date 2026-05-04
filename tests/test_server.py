# ruff: noqa: E501

import datetime
import os
import tempfile
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytz
import requests
import snowflake.connector
from dirty_equals import IsDatetime, IsUUID
from pandas.testing import assert_frame_equal
from snowflake.connector.cursor import ResultMetadata

from tests.utils import indent


def test_server_abort_request(server: dict) -> None:
    # network_timeout is set to 0 to trigger an abort
    with snowflake.connector.connect(**server | {"network_timeout": 0}) as conn1, conn1.cursor() as cur:
        cur.execute("select 'will abort'")


def test_server_binding_qmark(server: dict):
    with (
        snowflake.connector.connect(**server, database="db1", schema="schema1", paramstyle="qmark") as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            # test most important types from
            # https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#data-type-mappings-for-qmark-and-numeric-bindings
            """
            create or replace table example (
                XINT INT, XDECIMAL DECIMAL(10,2), XFLOAT REAL,
                XSTR TEXT, XUNICODE TEXT, XBYTES BINARY, XBYTEARRAY BINARY, XBOOL BOOLEAN,
                XDATE DATE, XTIME TIME, XDATETIME TIMESTAMP_NTZ
            )
            """
        )
        cur.execute(
            "insert into example values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                10.23,
                1.23456,
                "Jenny",
                "Jenny",
                b"Jenny",
                bytearray(b"Jenny"),
                True,
                datetime.date(2023, 1, 1),
                datetime.time(1, 2, 3, 123456),
                datetime.datetime(2023, 1, 2, 3, 4, 5, 123456),
            ),
        )
        cur.execute("select * from example")
        assert cur.fetchall() == [
            (
                1,
                Decimal("10.23"),
                1.23456,
                "Jenny",
                "Jenny",
                bytearray(b"Jenny"),
                bytearray(b"Jenny"),
                True,
                datetime.date(2023, 1, 1),
                datetime.time(1, 2, 3, 123456),
                datetime.datetime(2023, 1, 2, 3, 4, 5, 123456),
            )
        ]

        cur.execute("select * from example where xint = ?", (1,))


def test_server_connect(sconn: snowflake.connector.SnowflakeConnection) -> None:
    conn = sconn

    assert conn.database == "DB1"
    assert conn.schema == "SCHEMA1"

    conn.execute_string("create database db2; use database db2; create schema schema2; use schema schema2;")
    assert conn.database == "DB2"
    assert conn.schema == "SCHEMA2"


def test_server_connect_autocommit_false(server: dict) -> None:
    # connect with autocommit=False
    with (
        snowflake.connector.connect(**server | {"autocommit": False}, database="db1", schema="schema1") as conn,
        conn.cursor() as cur,
    ):
        cur.execute("create table test_commit (i int)")
        cur.execute("insert into test_commit values (1)")

        # verify row is present in current transaction
        cur.execute("select count(*) from test_commit")
        row = cur.fetchone()
        assert row and row[0] == 1

        # rollback
        conn.rollback()

        # verify row is gone
        cur.execute("select count(*) from test_commit")
        row = cur.fetchone()
        assert row and row[0] == 0

        # insert again
        cur.execute("insert into test_commit values (1)")
        conn.commit()

        # verify row is present
        cur.execute("select count(*) from test_commit")
        row = cur.fetchone()
        assert row and row[0] == 1


def test_server_client_session_keep_alive(server: dict) -> None:
    with snowflake.connector.connect(**server | {"client_session_keep_alive": True}):
        # shouldn't error
        pass


def test_server_close(server: dict) -> None:
    conn = snowflake.connector.connect(**server)

    # conn.close() ignores errors so we call the endpoint directly
    assert conn.rest and conn.rest.token
    response = requests.post(
        f"http://{server['host']}:{server['port']}/session?delete=true",
        headers={"Authorization": f'Snowflake Token="{conn.rest.token}"'},
        timeout=5,
        json={},
    )
    assert response.status_code == 200
    assert response.json()["success"]


def test_server_errors(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur = scur
    with pytest.raises(snowflake.connector.errors.ProgrammingError) as excinfo:
        cur.execute("select * from this_table_does_not_exist")

    assert excinfo.value.errno == 2003
    assert excinfo.value.sqlstate == "42S02"
    assert excinfo.value.msg
    assert "THIS_TABLE_DOES_NOT_EXIST" in excinfo.value.msg


def test_server_fetch_pandas_all(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur = scur

    cur.execute("select * from values (1, 'Salted'), (2, 'Caramel') as t(id, flavour)")

    expected_df = pd.DataFrame(
        [
            # TODO: snowflake returns int8
            (np.int32(1), "Salted"),
            (np.int32(2), "Caramel"),
        ],
        columns=["ID", "FLAVOUR"],
    )

    assert_frame_equal(cur.fetch_pandas_all(), expected_df)


def test_server_no_gzip(server: dict) -> None:
    # mimic the go snowflake connector which does not gzip requests
    headers = {
        "Accept": "application/snowflake",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": "Go/1.13.0 (darwin-arm64) gc/go1.23.6",
        "Client_App_Id": "Go",
        "Client_App_Version": "1.13.0",
    }

    login_payload = {
        "data": {
            "CLIENT_APP_ID": "Go",
            "CLIENT_APP_VERSION": "1.13.0",
            "SVN_REVISION": "",
            "ACCOUNT_NAME": "fakesnow",
            "LOGIN_NAME": "fake",
            "PASSWORD": "snow",
            "SESSION_PARAMETERS": {"CLIENT_VALIDATE_DEFAULT_PARAMETERS": True},
            "CLIENT_ENVIRONMENT": {
                "APPLICATION": "Go",
                "OS": "darwin",
                "OS_VERSION": "gc-arm64",
                "OCSP_MODE": "FAIL_OPEN",
                "GO_VERSION": "go1.23.6",
            },
        }
    }

    response = requests.post(
        f"http://{server['host']}:{server['port']}/session/v1/login-request",
        headers=headers,
        json=login_payload,
        timeout=5,
    )
    assert response.status_code == 200
    assert response.json()["success"]
    token = response.json()["data"]["token"]

    payload = {
        "sqlText": "SELECT current_timestamp() as TIME, current_user() as USER, current_role() as ROLE;",
        "asyncExec": False,
        "sequenceId": 1,
        "isInternal": False,
        "queryContextDTO": {},
    }

    response = requests.post(
        f"http://{server['host']}:{server['port']}/queries/v1/query-request?requestId=uuid1&request_guid=uuid2",
        headers=headers | {"Authorization": f'Snowflake Token="{token}"'},
        json=payload,
        timeout=5,
    )
    assert response.status_code == 200
    assert response.json()["success"]


def test_server_nop_regexes(server: dict) -> None:
    with snowflake.connector.connect(**server | {"session_parameters": {"nop_regexes": ["^CALL.*"]}}) as conn:
        cur = conn.cursor()
        cur.execute("call this_procedure_does_not_exist('foo', 'bar');")
        assert cur.fetchall() == [("Statement executed successfully.",)]


def test_server_put_list(sdcur: snowflake.connector.cursor.DictCursor) -> None:
    dcur = sdcur

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as temp_file:
        data = "1,2\n"
        temp_file.write(data)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file_basename = os.path.basename(temp_file_path)

        dcur.execute("CREATE STAGE stage1")
        dcur.execute(f"PUT 'file://{temp_file_path}' @stage1")
        assert dcur.fetchall() == [
            {
                "source": temp_file_basename,
                "target": f"{temp_file_basename}.gz",
                "source_size": len(data),
                "target_size": 42,  # GZIP compressed size
                "source_compression": "NONE",
                "target_compression": "GZIP",
                "status": "UPLOADED",
                "message": "",
            }
        ]

        dcur.execute("LIST @stage1")
        results = dcur.fetchall()
        assert len(results) == 1
        assert results[0] == {
            "name": f"stage1/{temp_file_basename}.gz",
            "size": 42,
            "md5": "29498d110c32a756df8109e70d22fa36",
            "last_modified": IsDatetime(
                # string in RFC 7231 date format (e.g. 'Sat, 31 May 2025 08:50:51 GMT')
                format_string="%a, %d %b %Y %H:%M:%S GMT"
            ),
        }

        # fully qualified stage name quoted
        dcur.execute('CREATE STAGE db1.schema1."stage2"')
        dcur.execute(f"PUT 'file://{temp_file_path}' @db1.schema1.\"stage2\"")


def test_server_put_qmark_quoted(server: dict) -> None:
    with (
        snowflake.connector.connect(
            **server,
            database="db1",
            schema="schema1",
            paramstyle="qmark",
        ) as conn,
        conn.cursor(snowflake.connector.cursor.DictCursor) as dcur,
        tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as temp_file,
    ):
        data = "1,2\n"
        temp_file.write(data)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file_basename = os.path.basename(temp_file_path)

        # quoted to mimic write_pandas
        dcur.execute("CREATE STAGE identifier(?)", ('"stage1"',))
        dcur.execute(f"PUT 'file://{temp_file_path}' ?", ('@"stage1"',))
        assert dcur.fetchall() == [
            {
                "source": temp_file_basename,
                "target": f"{temp_file_basename}.gz",
                "source_size": len(data),
                "target_size": 42,  # GZIP compressed size
                "source_compression": "NONE",
                "target_compression": "GZIP",
                "status": "UPLOADED",
                "message": "",
            }
        ]

        # fully qualified stage name quoted
        dcur.execute("CREATE STAGE identifier(?)", ('db1.schema1."stage2"',))
        dcur.execute(f"PUT 'file://{temp_file_path}' ?", ('@db1.schema1."stage2"',))


def test_server_put_non_existent_stage(sdcur: snowflake.connector.cursor.DictCursor) -> None:
    dcur = sdcur

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as temp_file:
        temp_file_path = temp_file.name

        with pytest.raises(snowflake.connector.errors.ProgrammingError) as excinfo:
            dcur.execute(f"PUT 'file://{temp_file_path}' @foobar")

        assert (
            str(excinfo.value)
            == "002003 (02000): SQL compilation error:\nStage 'DB1.SCHEMA1.FOOBAR' does not exist or not authorized."
        )


def test_server_response_params(server: dict) -> None:
    # mimic the jdbc driver
    headers = {
        "client_app_id": "JDBC",
        "client_app_version": "3.22.0",
        "accept": "application/json",
        "accept-encoding": "",
        "user-agent": "JDBC/3.22.0 (Mac OS X 15.3.1) JAVA/21.0.5",
    }

    login_payload = {
        "data": {
            "ACCOUNT_NAME": "127",
            "CLIENT_APP_ID": "JDBC",
            "CLIENT_APP_VERSION": "3.22.0",
            "CLIENT_ENVIRONMENT": {
                "tracing": "INFO",
                "OS": "Mac OS X",
                "OCSP_MODE": "FAIL_OPEN",
                "JAVA_VM": "OpenJDK 64-Bit Server VM",
                "APPLICATION": "DBeaver_DBeaver",
                "JDBC_JAR_NAME": "snowflake-jdbc-3.22.0",
                "password": "****",
                "database": "TEST_DB",
                "application": "DBeaver_DBeaver",
                "OS_VERSION": "15.3.1",
                "serverURL": "http://127.0.0.1:8000/",
                "JAVA_VERSION": "21.0.5",
                "user": "fake",
                "account": "127",
                "JAVA_RUNTIME": "OpenJDK Runtime Environment",
            },
            "EXT_AUTHN_DUO_METHOD": "push",
            "LOGIN_NAME": "fake",
            "PASSWORD": "snow",
            "SESSION_PARAMETERS": {},
        },
        "inFlightCtx": None,
    }

    response = requests.post(
        f"http://{server['host']}:{server['port']}/session/v1/login-request",
        headers=headers,
        json=login_payload,
        timeout=5,
    )
    assert response.status_code == 200
    assert response.json()["success"]

    # expected by the JDBC driver
    assert {"name": "AUTOCOMMIT", "value": True} in response.json()["data"]["parameters"]

    # expected by the .NET connector
    assert "sessionInfo" in response.json()["data"]

    # set autocommit=False
    response = requests.post(
        f"http://{server['host']}:{server['port']}/session/v1/login-request",
        headers=headers,
        json=login_payload | {"data": {"SESSION_PARAMETERS": {"AUTOCOMMIT": False}}},
        timeout=5,
    )
    assert response.status_code == 200
    assert response.json()["success"]

    assert {"name": "AUTOCOMMIT", "value": False} in response.json()["data"]["parameters"]


def test_server_rowcount(scur: snowflake.connector.cursor.SnowflakeCursor):
    cur = scur

    cur.execute("select * from values ('Salted'), ('Caramel')")
    assert cur.rowcount == 2


def test_server_sfid(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur = scur
    assert not cur.sfqid
    cur.execute("select 1")
    assert cur.sfqid == IsUUID()


def test_server_types_no_result_set(sconn: snowflake.connector.SnowflakeConnection) -> None:
    cur = sconn.cursor()
    cur.execute(
        """
        create or replace table example (
            XBOOLEAN BOOLEAN, XINT INT, XFLOAT FLOAT, XDECIMAL DECIMAL(10,2),
            XVARCHAR VARCHAR, XVARCHAR20 VARCHAR(20),
            XDATE DATE, XTIME TIME, XTIMESTAMP TIMESTAMP_TZ, XTIMESTAMP_NTZ TIMESTAMP_NTZ,
            XBINARY BINARY, /* XARRAY ARRAY, XOBJECT OBJECT, */ XVARIANT VARIANT
        )
        """
    )
    cur.execute("select * from example")
    # fmt: off
    assert cur.description == [
        ResultMetadata(name='XBOOLEAN', type_code=13, display_size=None, internal_size=None, precision=None, scale=None, is_nullable=True),
        # TODO: is_nullable should be False
        ResultMetadata(name='XINT', type_code=0, display_size=None, internal_size=None, precision=38, scale=0, is_nullable=True),
        ResultMetadata(name='XFLOAT', type_code=1, display_size=None, internal_size=None, precision=None, scale=None, is_nullable=True),
        ResultMetadata(name="XDECIMAL", type_code=0, display_size=None, internal_size=None, precision=10, scale=2, is_nullable=True),
        ResultMetadata(name="XVARCHAR", type_code=2, display_size=None, internal_size=16777216, precision=None, scale=None, is_nullable=True),
        # TODO: internal_size matches column size, ie: 20
        ResultMetadata(name='XVARCHAR20', type_code=2, display_size=None, internal_size=16777216, precision=None, scale=None, is_nullable=True),
        ResultMetadata(name='XDATE', type_code=3, display_size=None, internal_size=None, precision=None, scale=None, is_nullable=True),
        ResultMetadata(name='XTIME', type_code=12, display_size=None, internal_size=None, precision=0, scale=9, is_nullable=True),
        ResultMetadata(name='XTIMESTAMP', type_code=7, display_size=None, internal_size=None, precision=0, scale=9, is_nullable=True),
        ResultMetadata(name='XTIMESTAMP_NTZ', type_code=8, display_size=None, internal_size=None, precision=0, scale=9, is_nullable=True),
        ResultMetadata(name='XBINARY', type_code=11, display_size=None, internal_size=8388608, precision=None, scale=None, is_nullable=True),
        # TODO: handle ARRAY and OBJECT see https://github.com/tekumara/fakesnow/issues/26
        # ResultMetadata(name='XARRAY', type_code=10, display_size=None, internal_size=None, precision=None, scale=None, is_nullable=True),
        # ResultMetadata(name='XOBJECT', type_code=9, display_size=None, internal_size=None, precision=None, scale=None, is_nullable=True),
        ResultMetadata(name='XVARIANT', type_code=5, display_size=None, internal_size=None, precision=None, scale=None, is_nullable=True)

    ]
    # fmt: on


def test_server_types(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur = scur
    cur.execute(
        # TODO: match columns names without using AS
        """
        select
            true, 1::int, 2.0::float, to_decimal('12.3456', 10,2),
            'hello', 'hello'::varchar(20),
            to_date('2018-04-15'), to_time('04:15:29.123456'), to_timestamp_tz('2013-04-05 01:02:03.123456'), to_timestamp_ntz('2013-04-05 01:02:03.123456'),
            /* X'41424320E29D84', ARRAY_CONSTRUCT('foo'), */ OBJECT_CONSTRUCT('k','v1'), 1.23::VARIANT
            ,array_size(parse_json('["a","b"]')) /* duckdb uint64 */
        """
    )
    assert indent(cur.fetchall()) == [
        (
            True,
            1,
            2.0,
            Decimal("12.35"),
            "hello",
            "hello",
            datetime.date(2018, 4, 15),
            datetime.time(4, 15, 29, 123456),
            datetime.datetime(2013, 4, 5, 1, 2, 3, 123456, tzinfo=pytz.utc),
            datetime.datetime(2013, 4, 5, 1, 2, 3, 123456),
            # TODO
            # bytearray(b"ABC \xe2\x9d\x84"),
            # '[\n  "foo"\n]',
            '{\n  "k": "v1"\n}',
            "1.23",
            2,
        )
    ]


def test_server_async_query_with_retrieval(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    cur = scur
    # Execute an async query
    cur.execute_async("select 42 as answer, 'hello world' as message")
    async_sfqid = cur.sfqid
    assert async_sfqid

    # Execute a regular query
    cur.execute("select 'regular query' as type")
    regular_sfqid = cur.sfqid
    assert regular_sfqid

    # Test to retrieve results from the async query
    cur.get_results_from_sfqid(async_sfqid)
    async_results = cur.fetchall()
    assert async_results == [(42, "hello world")]

    # Test to retrieve results from the regular query
    cur.get_results_from_sfqid(regular_sfqid)
    regular_results = cur.fetchall()
    assert regular_results == [("regular query",)]


# avoid retries to shorten test time
@patch("snowflake.connector.cursor.ASYNC_RETRY_PATTERN", [0])
def test_server_monitoring_endpoint_error(scur: snowflake.connector.cursor.SnowflakeCursor):
    cur = scur
    cur.get_results_from_sfqid("00000000-0000-0000-0000-000000000000")
    with pytest.raises(
        snowflake.connector.errors.DatabaseError, match="Cannot retrieve data on the status of this query"
    ) as exc:
        cur.fetchall()
    assert exc.value.errno == -1


def test_server_result_scan(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """Test that result_scan() works end-to-end with cached query results."""
    cur = scur

    # Execute original query
    cur.execute("select 42 as answer, 'hello world' as message")
    original_sfqid = cur.sfqid
    original_result = cur.fetchall()

    # Use result_scan to retrieve cached results
    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{original_sfqid}'))")
    cached_result = cur.fetchall()

    # Verify results match
    assert cached_result == original_result
    assert cached_result == [(42, "hello world")]


def test_server_get_cached_query_result(server: dict) -> None:
    """Test the GET /queries/{query_id}/result endpoint directly."""
    # Connect and execute a query
    with snowflake.connector.connect(**server, database="db1", schema="schema1") as conn, conn.cursor() as cur:
        cur.execute("select 123 as number, 'test data' as text")
        query_id = cur.sfqid
        _ = cur.fetchall()  # Consume results to ensure they're cached

        # Get the token for authentication
        assert conn.rest and conn.rest.token

        # Make direct HTTP GET request to the endpoint
        response = requests.get(
            f"http://{server['host']}:{server['port']}/queries/{query_id}/result",
            headers={"Authorization": f'Snowflake Token="{conn.rest.token}"'},
            timeout=5,
        )

        assert response.status_code == 200
        json_response = response.json()

        # Verify response structure
        assert json_response["success"] is True
        assert json_response["code"] == "0"  # Success code
        assert "data" in json_response

        # Verify data fields are present
        data = json_response["data"]
        assert "rowtype" in data
        assert "rowset" in data
        assert "rowsetBase64" in data
        assert "total" in data
        assert "returned" in data

        # Verify data matches original query results.
        # Scale-0 fixed numerics are emitted as JSON strings in `rowset`
        # so the Node SDK can parse them via bigInt() without precision loss.
        assert data["total"] == 1
        assert data["returned"] == 1
        assert len(data["rowset"]) == 1
        assert data["rowset"][0] == ["123", "test data"]


def test_server_cache_eviction(sconn: snowflake.connector.SnowflakeConnection) -> None:
    """Test LRU cache behavior when >50 entries."""
    cur = sconn.cursor()

    # Execute 52 distinct queries to trigger eviction
    query_ids = []
    for i in range(52):
        cur.execute(f"select {i} as value")
        query_ids.append(cur.sfqid)

    # Verify first query is evicted from cache
    with pytest.raises(
        snowflake.connector.errors.ProgrammingError,
        match="Statement.*not found|Cannot retrieve data on the status of this query",
    ):
        cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{query_ids[0]}'))")

    # Verify 51st and 52nd queries are still in cache
    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{query_ids[50]}'))")
    assert cur.fetchall() == [(50,)]

    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{query_ids[51]}'))")
    assert cur.fetchall() == [(51,)]


def test_server_cache_empty_results(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """Test caching of queries with no results."""
    cur = scur

    # Execute query returning 0 rows
    cur.execute("SELECT * FROM (SELECT 1 AS id) WHERE 1=0")
    empty_sfqid = cur.sfqid
    assert cur.fetchall() == []
    assert cur.rowcount == 0

    # Use result_scan to retrieve cached empty results
    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{empty_sfqid}'))")
    cached_result = cur.fetchall()

    # Verify empty result set is handled correctly
    assert cached_result == []
    assert cur.rowcount == 0


def test_server_multiple_cached_queries(scur: snowflake.connector.cursor.SnowflakeCursor) -> None:
    """Test multiple queries cached independently without cross-contamination."""
    cur = scur

    # Execute 3 different queries with distinct results
    cur.execute("select 1 as first_query")
    query_id_1 = cur.sfqid
    result_1 = cur.fetchall()

    cur.execute("select 'second' as second_query, 999 as number")
    query_id_2 = cur.sfqid
    result_2 = cur.fetchall()

    cur.execute("select true as bool_col, 3.14 as pi")
    query_id_3 = cur.sfqid
    result_3 = cur.fetchall()

    # Retrieve results in random order using result_scan
    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{query_id_2}'))")
    assert cur.fetchall() == result_2

    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{query_id_1}'))")
    assert cur.fetchall() == result_1

    cur.execute(f"SELECT * FROM TABLE(RESULT_SCAN('{query_id_3}'))")
    assert cur.fetchall() == result_3


def test_server_get_cached_query_result_errors(server: dict) -> None:
    """Test error handling in GET endpoint."""
    # Connect to get a valid token
    with snowflake.connector.connect(**server, database="db1", schema="schema1") as conn:
        assert conn.rest and conn.rest.token
        token = conn.rest.token

        # Test 1: Non-existent query ID - should return 404 or error
        non_existent_uuid = "00000000-0000-0000-0000-000000000000"
        response = requests.get(
            f"http://{server['host']}:{server['port']}/queries/{non_existent_uuid}/result",
            headers={"Authorization": f'Snowflake Token="{token}"'},
            timeout=5,
        )

        # The response should indicate the query was not found
        # Snowflake returns success=false with an error code
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            json_response = response.json()
            assert json_response["success"] is False

        # Test 2: Invalid query ID format (not a UUID)
        response = requests.get(
            f"http://{server['host']}:{server['port']}/queries/not-a-uuid/result",
            headers={"Authorization": f'Snowflake Token="{token}"'},
            timeout=5,
        )

        # Should handle invalid UUID gracefully
        assert response.status_code in [200, 400, 404]
        if response.status_code == 200:
            json_response = response.json()
            assert json_response["success"] is False


# --- Node SDK BigInt parity (JS_TREAT_INTEGER_AS_BIGINT) ---------------------
#
# Real Snowflake's wire contract for the JSON `data.rowset` payload encodes
# every fixed-point numeric value as a JSON STRING (not a bare JSON number),
# so that the Node.js SDK can call `bigInt(rawColumnValue)` on a precision-
# safe string representation. The Node SDK only takes the BigInt path when
# (1) `JS_TREAT_INTEGER_AS_BIGINT` is echoed back in `data.parameters` and
# (2) the column's `rowtype.scale` is 0. fakesnow used to violate both halves
# of that contract -- these tests pin the fix.


def _login_and_get_token(server: dict, session_parameters: dict | None = None) -> str:
    """Issue a raw login request so we can inspect the per-query JSON payload."""
    payload = {
        "data": {
            "ACCOUNT_NAME": "fakesnow",
            "LOGIN_NAME": "fake",
            "PASSWORD": "snow",
            "SESSION_PARAMETERS": session_parameters or {},
            "CLIENT_APP_ID": "JavaScript",
            "CLIENT_APP_VERSION": "1.0.0",
            "CLIENT_ENVIRONMENT": {"APPLICATION": "test"},
        }
    }
    resp = requests.post(
        f"http://{server['host']}:{server['port']}/session/v1/login-request",
        json=payload,
        timeout=5,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"], body
    return body["data"]["token"]


def _exec_query(
    server: dict, token: str, sql: str, bindings: dict | None = None
) -> dict:
    payload: dict = {"sqlText": sql}
    if bindings is not None:
        payload["bindings"] = bindings
    resp = requests.post(
        f"http://{server['host']}:{server['port']}/queries/v1/query-request?requestId=node-sdk-test",
        headers={"Authorization": f'Snowflake Token="{token}"'},
        json=payload,
        timeout=5,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"], body
    return body["data"]


def test_server_node_sdk_rowset_integer_emitted_as_string(server: dict) -> None:
    """Snowflake wire contract: scale-0 numerics are JSON strings in `rowset`."""
    token = _login_and_get_token(server)
    data = _exec_query(server, token, "SELECT 42 AS answer")

    rowtype = data["rowtype"]
    assert rowtype[0]["type"] == "fixed"
    assert rowtype[0]["scale"] == 0
    # The column value MUST be a string, not a bare JSON number, so that
    # node-side `bigInt(rawColumnValue)` receives a precision-safe input.
    assert data["rowset"][0][0] == "42"


def test_server_node_sdk_rowset_large_integer_preserves_precision(server: dict) -> None:
    """Integers > 2^53 must round-trip without precision loss.

    If `rowset` ever encodes a Python int as a bare JSON number, the value is
    silently lost during JSON.parse on the Node side (max safe int is 2^53-1).
    Stringification is the only correct fix; we assert it directly here.
    """
    big = 2**62  # 4_611_686_018_427_387_904 — well past JS Number's safe range
    token = _login_and_get_token(server)
    data = _exec_query(server, token, f"SELECT {big} AS x")

    assert data["rowtype"][0]["type"] == "fixed"
    assert data["rowtype"][0]["scale"] == 0
    assert data["rowset"][0][0] == str(big)
    # Defensive: ensure we are NOT getting a Python-decoded int that JSON
    # would have rounded. The string equality above already proves this for
    # `requests.json()`, but a bare-number encoding would have produced a
    # Python int equal to `big` — distinguishing requires the type check.
    assert isinstance(data["rowset"][0][0], str)


def test_server_node_sdk_session_param_js_treat_integer_as_bigint_echoed(server: dict) -> None:
    """`data.parameters` must echo `JS_TREAT_INTEGER_AS_BIGINT` when the
    client supplied it at login. The Node SDK reads this from the per-query
    `parameters` array (snowflake-sdk result.js:65-70) to decide between
    `convertRawNumber` and `convertRawBigInt`."""
    token = _login_and_get_token(
        server,
        session_parameters={"JS_TREAT_INTEGER_AS_BIGINT": True},
    )
    data = _exec_query(server, token, "SELECT 1")

    params = {p["name"]: p["value"] for p in data["parameters"]}
    assert params.get("JS_TREAT_INTEGER_AS_BIGINT") is True


def test_server_node_sdk_session_param_js_treat_integer_as_bigint_omitted_when_not_set(
    server: dict,
) -> None:
    """If the client did not opt into JS_TREAT_INTEGER_AS_BIGINT at login, the
    parameter must NOT appear in `data.parameters`. Before BigInt
    normalization was added fakesnow only echoed TIMEZONE; this preserves
    backwards-compat for clients that never sent the flag (see
    `_build_query_parameters`)."""
    token = _login_and_get_token(server)  # no session_parameters
    data = _exec_query(server, token, "SELECT 1")

    names = {p["name"] for p in data["parameters"]}
    assert "JS_TREAT_INTEGER_AS_BIGINT" not in names


def test_server_node_sdk_rowset_negative_integer_emitted_as_string(server: dict) -> None:
    """Negative scale-0 fixed numerics must round-trip as the canonical
    signed string form."""
    token = _login_and_get_token(server)
    data = _exec_query(server, token, "SELECT -42 AS n")
    assert data["rowtype"][0]["type"] == "fixed"
    assert data["rowtype"][0]["scale"] == 0
    assert data["rowset"][0][0] == "-42"


def test_server_node_sdk_rowset_null_integer_remains_null(server: dict) -> None:
    """NULL must NOT be stringified to 'None' -- the helper guards on
    `row[i] is not None`. JSON null must round-trip as Python None."""
    token = _login_and_get_token(server)
    data = _exec_query(server, token, "SELECT CAST(NULL AS INTEGER) AS n")
    assert data["rowtype"][0]["type"] == "fixed"
    assert data["rowtype"][0]["scale"] == 0
    assert data["rowset"][0][0] is None


def test_server_node_sdk_rowset_multiple_rows_all_stringified(server: dict) -> None:
    """The inner row loop must visit every row, not just row 0."""
    token = _login_and_get_token(server)
    data = _exec_query(
        server, token, "SELECT * FROM (VALUES (1), (2), (3)) AS t(n) ORDER BY n"
    )
    assert data["rowtype"][0]["type"] == "fixed"
    assert data["rowtype"][0]["scale"] == 0
    assert [row[0] for row in data["rowset"]] == ["1", "2", "3"]


def test_server_node_sdk_cached_result_echoes_bigint_param_and_stringifies(
    server: dict,
) -> None:
    """The Node SDK BigInt wiring is symmetric across `query_request` AND
    `get_cached_query_result`. This test executes a query (which caches the
    result), then re-fetches via the cached-result endpoint and asserts BOTH
    contracts: rowset stringification AND `JS_TREAT_INTEGER_AS_BIGINT` echo.
    """
    token = _login_and_get_token(
        server, session_parameters={"JS_TREAT_INTEGER_AS_BIGINT": True}
    )
    # First query populates the cache.
    first = _exec_query(server, token, "SELECT 12345 AS n")
    query_id = first["queryId"]

    # Second fetch hits the cached-result endpoint.
    resp = requests.get(
        f"http://{server['host']}:{server['port']}/queries/{query_id}/result",
        headers={"Authorization": f'Snowflake Token="{token}"'},
        timeout=5,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"], body
    cached = body["data"]

    # Stringification through the cached path.
    assert cached["rowtype"][0]["type"] == "fixed"
    assert cached["rowtype"][0]["scale"] == 0
    assert cached["rowset"][0][0] == "12345"

    # BigInt-flag echo through the cached path.
    params = {p["name"]: p["value"] for p in cached["parameters"]}
    assert params.get("JS_TREAT_INTEGER_AS_BIGINT") is True


# --- Node SDK temporal-rowset parity (DATE / TIMESTAMP_NTZ) ------------------
#
# Real Snowflake's wire contract for the JSON `data.rowset` payload encodes
# DATE columns as days-since-epoch numeric strings (e.g. "20089" for
# 2025-01-01) and TIMESTAMP_NTZ columns as fractional-epoch-seconds strings
# (e.g. "1735693261.000000000"). The Node SDK's `convertRawDate` /
# `convertRawTimestampNtz` parse these via `Number(...)` / `BigNumber(...)`.
# fakesnow used to emit ISO strings ("2025-01-01") which silently coerced to
# NaN -> epoch zero -> 1970-01-01. The Python SDK is unaffected (it consumes
# `rowsetBase64`, not `rowset`).


def test_server_node_sdk_rowset_bound_date_emitted_as_days_since_epoch(
    server: dict,
) -> None:
    """A DATE bind round-trips to the Snowflake JSON wire format
    (days-since-epoch as a numeric string), not an ISO date string."""
    token = _login_and_get_token(server)
    # Mimic the Node SDK's wire format for a TEXT-typed binding -- the Node
    # SDK categorizes any string bind as TEXT (see snowflake-sdk
    # statement.js:buildBindsMap), and slonik's `sql.date(...)` fragment emits
    # the date in ISO-yyyy-mm-dd form. The downstream SQL casts via `:N::date`.
    data = _exec_query(
        server,
        token,
        "SELECT :1::date AS d",
        bindings={"1": {"type": "TEXT", "value": "2025-01-01"}},
    )
    assert data["rowtype"][0]["type"] == "date"
    # 2025-01-01 == day 20089 of the Unix epoch.
    assert data["rowset"][0][0] == "20089"


def test_server_node_sdk_rowset_bound_timestamp_emitted_as_fractional_epoch_seconds(
    server: dict,
) -> None:
    """A TIMESTAMP bind round-trips to the Snowflake JSON wire format
    (fractional epoch seconds as a numeric string at the column's scale),
    not an ISO datetime string."""
    token = _login_and_get_token(server)
    # Mimic slonik's `sql.timestamp(date)` fragment -- it emits
    # `to_timestamp(:N)` with the bind value being epoch-seconds-as-string.
    data = _exec_query(
        server,
        token,
        "SELECT to_timestamp(:1) AS t",
        bindings={"1": {"type": "TEXT", "value": "1735693261"}},
    )
    assert data["rowtype"][0]["type"] == "timestamp_ntz"
    scale = data["rowtype"][0]["scale"]
    # fakesnow's default TIMESTAMP_NTZ scale is 9 (nanosecond precision).
    # Sanity: if scale ever drifts, this test still validates the format.
    assert scale == 9
    # 1735693261 == 2025-01-01 01:01:01 UTC.
    assert data["rowset"][0][0] == "1735693261.000000000"


def test_server_node_sdk_rowset_null_date_remains_null(server: dict) -> None:
    """NULL DATE columns must round-trip as JSON null, not as the string
    "0" or any other coerced form."""
    token = _login_and_get_token(server)
    data = _exec_query(server, token, "SELECT CAST(NULL AS DATE) AS d")
    assert data["rowtype"][0]["type"] == "date"
    assert data["rowset"][0][0] is None


def test_server_node_sdk_rowset_null_timestamp_remains_null(server: dict) -> None:
    """NULL TIMESTAMP_NTZ columns must round-trip as JSON null."""
    token = _login_and_get_token(server)
    data = _exec_query(server, token, "SELECT CAST(NULL AS TIMESTAMP_NTZ) AS t")
    assert data["rowtype"][0]["type"] == "timestamp_ntz"
    assert data["rowset"][0][0] is None


def test_server_node_sdk_cached_result_normalizes_temporal_rowset(
    server: dict,
) -> None:
    """The temporal-rowset normalization must be symmetric across
    `query_request` AND `get_cached_query_result`, mirroring the BigInt
    parity coverage above."""
    token = _login_and_get_token(server)
    first = _exec_query(
        server,
        token,
        "SELECT :1::date AS d, to_timestamp(:2) AS t",
        bindings={
            "1": {"type": "TEXT", "value": "2025-01-01"},
            "2": {"type": "TEXT", "value": "1735693261"},
        },
    )
    query_id = first["queryId"]

    resp = requests.get(
        f"http://{server['host']}:{server['port']}/queries/{query_id}/result",
        headers={"Authorization": f'Snowflake Token="{token}"'},
        timeout=5,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"], body
    cached = body["data"]

    assert cached["rowset"][0][0] == "20089"
    assert cached["rowset"][0][1] == "1735693261.000000000"

