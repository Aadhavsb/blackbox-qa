import pytest

from blackbox_qa import tools
from blackbox_qa.tools import ToolError, validate_select


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM reports",
        "select count(*) from reports where ev_year = 2008",
        "SELECT ev_id FROM reports WHERE acft_make = 'Cessna';",  # trailing ; stripped
        "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
    ],
)
def test_validate_select_accepts_read_only(sql):
    cleaned = validate_select(sql)
    assert cleaned.lower().startswith(("select", "with"))
    assert ";" not in cleaned


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE reports",
        "UPDATE reports SET ev_year = 1999",
        "DELETE FROM reports",
        "INSERT INTO reports VALUES (1)",
        "SELECT 1; DROP TABLE reports",  # multi-statement
        "TRUNCATE reports",
        "",
        "   ",
        "explain analyze select 1",
    ],
)
def test_validate_select_rejects_writes_and_junk(sql):
    with pytest.raises(ToolError):
        validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT * FROM dblink('host=x', 'select 1') AS t(a int)",
        "SELECT set_config('x', 'y', false)",
        "SELECT current_setting('data_directory')",
        "select count(*) from reports where ev_id in (select pg_sleep(5))",
    ],
)
def test_validate_select_rejects_side_effect_functions(sql):
    with pytest.raises(ToolError):
        validate_select(sql)


def test_validate_select_allows_ordinary_functions():
    # Aggregates / harmless functions must still pass.
    assert validate_select("SELECT count(*), max(ev_year) FROM reports")
    assert validate_select("SELECT lower(acft_make) FROM reports")


def test_validate_select_rejects_non_string():
    with pytest.raises(ToolError):
        validate_select(123)  # type: ignore[arg-type]


def test_dispatch_unknown_tool():
    with pytest.raises(ToolError):
        tools.dispatch("nope", {})


def test_dispatch_bad_arguments_type():
    with pytest.raises(ToolError):
        tools.dispatch("hybrid_search", ["not", "a", "dict"])  # type: ignore[arg-type]


def test_dispatch_missing_required_arg_raises_toolerror():
    # fetch_full_report requires ev_id; omitting it should surface as ToolError,
    # not a raw TypeError, so the agent can feed it back for self-correction.
    with pytest.raises(ToolError):
        tools.dispatch("fetch_full_report", {})


def test_schemas_match_registry():
    schema_names = {s["function"]["name"] for s in tools.SCHEMAS}
    assert schema_names == set(tools.REGISTRY)
