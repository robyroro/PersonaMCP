from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated
from zipfile import BadZipFile

import typer

from personamcp import __version__
from personamcp.analysis import analyze
from personamcp.benchmark import benchmark as run_benchmark
from personamcp.config import Config, data_dir
from personamcp.embeddings import LocalEmbeddingProvider, prepare_model
from personamcp.importers import importer_for, read_sources
from personamcp.models import Conversation, stable_id
from personamcp.retrieval import index_interactions
from personamcp.service import PersonaService

app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help="Local-first communication memory for AI agents.",
)
config_app = typer.Typer(no_args_is_help=True, help="Configure explicit owner names and aliases.")
model_app = typer.Typer(no_args_is_help=True, help="Prepare local model assets explicitly.")
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")
state: dict[str, Path] = {}


@app.callback()
def main(
    home: Annotated[Path | None, typer.Option(help="Private data directory.")] = None,
    version: Annotated[bool, typer.Option("--version", is_eager=True)] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    state["home"] = home or data_dir()


@contextmanager
def session() -> Iterator[PersonaService]:
    service: PersonaService | None = None
    try:
        if not (state["home"] / "config.json").exists():
            raise ValueError("Run persona init first")
        service = PersonaService(state["home"])
        yield service
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError, BadZipFile) as exc:
        # Avoid dumping malformed imported text, private messages, or tracebacks.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        typer.echo("Error: " + message, err=True)
        raise typer.Exit(1) from None
    finally:
        if service:
            service.close()


def output(data: object) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


@app.command()
def init() -> None:
    """Initialize a private local database and configuration without overwriting existing data."""
    home = state["home"]
    with_init = Config.load(home)
    with_init.save(home)
    from personamcp.storage import Store

    Store(home).close()
    typer.echo(f"PersonaMCP initialized: {home}\nSet your owner name with persona config set-name.")


@config_app.command("set-name")
def set_name(name: str, platform: Annotated[str | None, typer.Option()] = None) -> None:
    """Set the exact owner display name, optionally scoped to a platform."""
    with session() as service:
        if not name.strip():
            raise ValueError("Owner name cannot be empty")
        if platform:
            service.config.platform_aliases[platform] = [name]
        else:
            service.config.name = name
        service.config.save(service.home)
        service.store.reidentify(service.config)
        typer.echo("Owner identity updated; derived profiles invalidated.")


@config_app.command("add-alias")
def add_alias(alias: str, platform: Annotated[str | None, typer.Option()] = None) -> None:
    """Add an owner name or account ID; platform aliases override global identity."""
    with session() as service:
        if not alias.strip():
            raise ValueError("Alias cannot be empty")
        if platform:
            aliases = service.config.platform_aliases.setdefault(
                platform, [service.config.name, *service.config.aliases]
            )
        else:
            aliases = service.config.aliases
        if alias not in aliases:
            aliases.append(alias)
        service.config.save(service.home)
        service.store.reidentify(service.config)
        typer.echo("Alias saved; identity flags and interactions recomputed.")


@config_app.command("show")
def show_config() -> None:
    """Show local owner configuration (contains private identifiers)."""
    with session() as service:
        output(json.loads((service.home / "config.json").read_text(encoding="utf-8")))


@app.command("import")
def import_data(
    platform: str,
    path: Path,
    month_first: Annotated[bool, typer.Option(help="WhatsApp US month/day dates.")] = False,
    conversation_id: Annotated[
        str | None,
        typer.Option(help="Stable WhatsApp conversation ID for renamed or refreshed files."),
    ] = None,
) -> None:
    """Import supported export files atomically. Raw files are never changed."""
    with session() as service:
        adapter = importer_for(platform, day_first=not month_first, conversation_id=conversation_id)
        conversations: list[Conversation] = []
        digest = hashlib.sha256()
        files = 0
        for source, text in read_sources(path, platform):
            digest.update(stable_id(source, text).encode())
            conversations.extend(adapter.parse(text, source))
            files += 1
        if not conversations:
            raise ValueError("No supported chat files found in the selected export")
        matches = sum(
            m.sender.strip().casefold() in service.config.identities(c.platform)
            or bool(m.sender_id and m.sender_id.casefold() in service.config.identities(c.platform))
            for c in conversations
            for m in c.messages
        )
        if not matches:
            raise ValueError(
                "Owner name/aliases match no exported messages. Configure the exact "
                "platform display name or account ID before importing"
            )
        digest.update(
            stable_id(
                platform,
                service.config.platform_aliases,
                service.config.name,
                service.config.aliases,
            ).encode()
        )
        result = service.store.import_conversations(
            conversations, digest.hexdigest(), platform, service.config
        )
        output({"files_read": files, "owner_messages_matched": matches, **result})


@app.command()
def stats(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show database counts and actual profile/vector coverage."""
    with session() as service:
        data = service.store.stats()
        indexed = service.store.db.execute(
            "SELECT count(*) FROM embeddings WHERE provider=?",
            (service.provider.identity if service.provider else "",),
        ).fetchone()[0]
        data["semantic_index"] = {
            "indexed": indexed,
            "total": data["interactions"],
            "ready": bool(data["interactions"] and indexed == data["interactions"]),
        }
        if as_json:
            output(data)
        else:
            typer.echo("PersonaMCP\n")
            for label, key in (
                ("Messages indexed", "messages"),
                ("User messages", "user_messages"),
                ("Conversations", "conversations"),
                ("People", "people"),
            ):
                typer.echo(f"{label + ':':24}{data[key]:>12,}")
            for name, count in data["platforms"].items():
                typer.echo(f"{name:24}{count:>12,}")
            typer.echo("\nStyle profile: " + ("ready" if data["style_profile"] else "not analyzed"))
            typer.echo(f"Semantic index: {indexed:,}/{data['interactions']:,} interactions")


@app.command("analyze")
def analyze_data() -> None:
    """Compute deterministic outgoing-message profiles and write local persona.md."""
    with session() as service:
        result = analyze(service.store, service.config)
        typer.echo(f"Generated {len(result['profiles'])} profiles: {service.home / 'persona.md'}")


@app.command()
def search(
    query: str,
    person: Annotated[str | None, typer.Option()] = None,
    platform: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option()] = 5,
) -> None:
    """Search only the owner's text with optional strict recipient scope."""
    with session() as service:
        output(service.search_messages(query, person, platform, limit))


@app.command("person")
def person_profile(name: str) -> None:
    """Show communication-style evidence for one exact recipient."""
    with session() as service:
        output(service.get_person_style(name))


@app.command()
def similar(
    message: str,
    person: Annotated[str | None, typer.Option()] = None,
    context: Annotated[str | None, typer.Option()] = None,
    platform: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option()] = 5,
) -> None:
    """Retrieve similar incoming/reply examples; clearly report lexical or semantic mode."""
    with session() as service:
        output(service.find_similar_interactions(message, person, context, platform, limit))


@app.command("writing-context")
def writing_context(
    message: str,
    person: Annotated[str | None, typer.Option()] = None,
    context: Annotated[str | None, typer.Option()] = None,
    platform: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Preview exactly the evidence an MCP writing agent receives."""
    with session() as service:
        output(service.get_writing_context(message, person, context, platform))


@app.command()
def conversations() -> None:
    """List local conversation IDs, titles, platforms and assigned contexts."""
    with session() as service:
        output(
            service.store.rows("SELECT id,title,platform,context FROM conversations ORDER BY title")
        )


@app.command("set-context")
def set_context(conversation_id: str, context: str) -> None:
    """Assign an explicit context (business, casual, family, or your own label)."""
    with session() as service:
        if len(context) > 80 or not context.strip():
            raise ValueError("Context must contain 1–80 characters")
        if not service.store.rows("SELECT id FROM conversations WHERE id=?", (conversation_id,)):
            raise ValueError("Conversation not found")
        with service.store.db:
            service.store.db.execute(
                "UPDATE conversations SET context=? WHERE id=?", (context, conversation_id)
            )
            service.store.db.execute(
                "UPDATE interactions SET context=? WHERE conversation_id=?",
                (context, conversation_id),
            )
            service.store.invalidate()
        typer.echo("Context assigned. Run persona analyze to refresh profiles.")


@app.command("export-profile")
def export_profile(
    destination: Annotated[Path, typer.Argument()] = Path("persona.md"),
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Explicitly copy the profile; no raw conversations are exported."""
    with session() as service:
        source = service.home / ("persona-profile.json" if as_json else "persona.md")
        if not source.exists():
            raise ValueError("Run persona analyze before exporting a profile")
        if destination.resolve() == source.resolve():
            raise ValueError("Choose a destination outside the generated profile file")
        if destination.exists():
            raise ValueError("Destination exists; choose another filename")
        shutil.copyfile(source, destination)
        typer.echo(f"Profile exported: {destination}")


@model_app.command("prepare")
def model_prepare() -> None:
    """Explicitly download model weights from Hugging Face; sends no conversation data."""
    with session() as service:
        typer.echo("Downloading local embedding model assets. Conversation data is not sent.")
        revision = prepare_model(service.home, service.config)
        typer.echo(f"Local model prepared at pinned revision {revision}.")


@app.command()
def index() -> None:
    """Build/resume the local semantic index using offline inference."""
    with session() as service:
        provider = LocalEmbeddingProvider(service.home, service.config)
        count = index_interactions(
            service.store, provider, lambda n: typer.echo(f"Indexed {n:,} interactions", err=True)
        )
        typer.echo(f"Added {count:,} local vectors.")


@app.command()
def benchmark(limit: Annotated[int, typer.Option()] = 30) -> None:
    """Run temporal holdout retrieval/style similarity; never call an external LLM."""
    with session() as service:
        output(run_benchmark(service.store, service.provider, limit))


@app.command("delete-conversation")
def delete_conversation(
    conversation_id: str, yes: Annotated[bool, typer.Option("--yes")] = False
) -> None:
    """Erase a conversation, derived profiles and vectors; original export files remain."""
    with session() as service:
        if not service.store.rows("SELECT id FROM conversations WHERE id=?", (conversation_id,)):
            raise ValueError("Conversation not found")
        if not yes:
            typer.confirm(
                "Delete this local conversation and invalidate derived profiles?", abort=True
            )
        service.store.delete_conversations([conversation_id])
        typer.echo("Conversation deleted; derived profiles invalidated and database vacuumed.")


@app.command("delete-person")
def delete_person(name: str, yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    """Erase all conversations containing this person, including entire group conversations."""
    with session() as service:
        ids = [
            r["conversation_id"]
            for r in service.store.rows(
                "SELECT DISTINCT conversation_id FROM participants WHERE name_key=?",
                (name.strip().casefold(),),
            )
        ]
        if not ids:
            raise ValueError("Person not found")
        if not yes:
            typer.confirm(
                f"Delete {len(ids)} conversations, including groups containing this person?",
                abort=True,
            )
        service.store.delete_conversations(ids)
        typer.echo(f"Deleted {len(ids)} conversations and derived evidence.")


@app.command()
def reset(yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    """Erase communication data and owner identities. Keep downloaded model assets."""
    with session() as service:
        if not yes:
            typer.confirm("Erase all local communication data and owner identities?", abort=True)
        service.store.delete_conversations(
            [r["id"] for r in service.store.rows("SELECT id FROM conversations")]
        )
        with service.store.db:
            service.store.db.execute("DELETE FROM profiles")
            service.store.db.execute("DELETE FROM imports")
            service.store.invalidate()
        service.config.name, service.config.aliases, service.config.platform_aliases = "", [], {}
        service.config.save(service.home)
        typer.echo("Personal data reset. Original exports and downloaded model assets remain.")


@app.command()
def serve() -> None:
    """Run official MCP SDK stdio transport. Stdout contains only MCP protocol messages."""
    if not (state["home"] / "config.json").exists():
        typer.echo("Run persona init first", err=True)
        raise typer.Exit(1)
    from personamcp.mcp_server import create_server

    create_server(state["home"]).run(transport="stdio")


@app.command("skill-path")
def skill_path() -> None:
    """Print the bundled writing skill path for copying into your agent's skill directory."""
    packaged = Path(__file__).parent / "skills" / "write-like-me" / "SKILL.md"
    source = Path(__file__).parents[2] / "skills" / "write-like-me" / "SKILL.md"
    selected = packaged if packaged.is_file() else source
    if not selected.is_file():
        typer.echo("Bundled skill is unavailable in this installation", err=True)
        raise typer.Exit(1)
    typer.echo(str(selected.resolve()))


if __name__ == "__main__":
    app()
