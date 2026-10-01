from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typer.testing import CliRunner

from personamcp.analysis import analyze
from personamcp.cli import app
from personamcp.config import Config
from personamcp.storage import Store

runner = CliRunner()


def test_version_without_subcommand() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_json_pipeline_unicode_under_legacy_encoding(store: Store) -> None:
    with store.db:
        store.db.execute("UPDATE messages SET text='bună șț 😄' WHERE is_user=1")
    result = subprocess.run(
        [sys.executable, "-m", "personamcp", "--home", str(store.path.parent), "search", "bună"],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["results"][0]["historical_quote"]["user_message"] == "bună șț 😄"


def run(home: Path, *args: str) -> Any:
    return runner.invoke(app, ["--home", str(home), *args])


def test_cli_full_synthetic_workflow(tmp_path: Path) -> None:
    home = tmp_path / "private"
    assert run(home, "init").exit_code == 0
    assert run(home, "config", "set-name", "Owner").exit_code == 0
    source = tmp_path / "messages.json"
    source.write_text(
        json.dumps(
            {
                "id": "chat",
                "participants": ["Owner", "Alex"],
                "messages": [
                    {"sender": "Alex", "text": "ce faci?", "timestamp": "2024-01-01T10:00:00Z"},
                    {
                        "sender": "Owner",
                        "text": "acu gen vin =))",
                        "timestamp": "2024-01-01T10:00:10Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    original = source.read_bytes()
    assert run(home, "import", "json", str(source)).exit_code == 0
    assert json.loads(run(home, "import", "json", str(source)).stdout)["added"] == 0
    assert source.read_bytes() == original
    assert json.loads(run(home, "stats", "--json").stdout)["messages"] == 2
    assert run(home, "analyze").exit_code == 0
    assert json.loads(run(home, "search", "gen").stdout)["results"]
    assert json.loads(run(home, "similar", "ce faci", "--person", "Alex").stdout)["results"]
    assert json.loads(run(home, "person", "Alex").stdout)["profile"]["message_count"] == 1
    destination = tmp_path / "export.md"
    assert run(home, "export-profile", str(destination)).exit_code == 0
    assert destination.exists()
    assert run(home, "export-profile", str(destination)).exit_code == 1
    assert run(home, "delete-person", "Alex", "--yes").exit_code == 0
    assert json.loads(run(home, "stats", "--json").stdout)["messages"] == 0
    assert run(home, "reset", "--yes").exit_code == 0
    assert Config.load(home).name == ""


def test_import_failure_is_atomic(tmp_path: Path) -> None:
    home = tmp_path / "private"
    run(home, "init")
    run(home, "config", "set-name", "Owner")
    source = tmp_path / "input"
    source.mkdir()
    (source / "a.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "sender": "Owner",
                        "text": "must not enter DB",
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                ]
            }
        )
    )
    (source / "b.json").write_text('{"sensitive BROKEN')
    result = run(home, "import", "json", str(source))
    assert result.exit_code == 1
    assert "sensitive" not in result.output
    assert json.loads(run(home, "stats", "--json").stdout)["messages"] == 0


def test_owner_must_match_before_import(tmp_path: Path) -> None:
    run(tmp_path, "init")
    run(tmp_path, "config", "set-name", "Wrong")
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps([{"sender": "Owner", "text": "hello", "timestamp": "2024-01-01T00:00:00Z"}])
    )
    result = run(tmp_path, "import", "json", str(source))
    assert result.exit_code == 1
    assert "match no exported messages" in result.output


def test_serve_initializes_an_empty_home_and_lists_tools(tmp_path: Path) -> None:
    home = tmp_path / "first-launch"

    async def scenario() -> None:
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "personamcp", "--home", str(home), "serve"]
        )
        with anyio.fail_after(30):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    listed = await client.list_tools()
                    assert len(listed.tools) == 6
                    summary = await client.call_tool("get_persona_summary", {})
                    assert not summary.isError, summary

    anyio.run(scenario)
    assert (home / "config.json").exists()


def test_official_mcp_stdio_client_all_tools(store: Store, config: Config) -> None:
    analyze(store, config)

    async def scenario() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "personamcp", "--home", str(store.path.parent), "serve"],
        )
        with anyio.fail_after(30):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as client:
                    initialized = await client.initialize()
                    assert initialized.serverInfo.name == "PersonaMCP"
                    listed = await client.list_tools()
                    assert len(listed.tools) == 6
                    assert all(t.annotations and t.annotations.readOnlyHint for t in listed.tools)
                    calls = {
                        "get_style_profile": {},
                        "get_person_style": {"person": "Alex"},
                        "search_messages": {"query": "gen", "person": "Alex"},
                        "find_similar_interactions": {"message": "cafea", "person": "Alex"},
                        "get_writing_context": {"message": "cafea", "person": "Alex"},
                        "get_persona_summary": {},
                    }
                    for name, args in calls.items():
                        result = await client.call_tool(name, args)
                        assert not result.isError, result
                        data = result.structuredContent
                        assert data and "untrusted" in data["trust_notice"]
                        if args.get("person"):
                            assert "blair" not in json.dumps(data).casefold()
                    rejected = await client.call_tool(
                        "search_messages", {"query": "gen", "limit": 999}
                    )
                    assert rejected.isError

    anyio.run(scenario)
