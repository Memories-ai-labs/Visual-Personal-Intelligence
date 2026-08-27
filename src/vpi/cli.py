"""`vpi` — the command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vpi import ingest as ingest_mod
from vpi.chat import run_chat
from vpi.config import get_settings
from vpi.session import MissingCollection, build_client, build_session, resolve_collection

app = typer.Typer(
    add_completion=False,
    help="Chat with your own video memory, grounded in the Memories.ai video datalake.",
)
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


@app.command()
def collections() -> None:
    """List your collections and their video counts."""
    settings = get_settings()
    with build_client(settings) as client:
        rows = client.list_collections()
        table = Table("id", "name", "videos", "faces", box=None)
        for collection in rows:
            table.add_row(
                collection.id,
                collection.name,
                str(collection.video_count),
                "on" if collection.face_recognition_enabled else "off",
            )
        console.print(table if rows else "No collections yet.")
        if not settings.demo:
            console.print(f"\nbalance ${client.balance_usd():.2f}")


@app.command()
def ingest(
    sources: list[str] = typer.Argument(..., help="Files, directories, or public direct URLs."),
    collection: str = typer.Option("", "--collection", "-c", help="Existing collection id."),
    name: str = typer.Option("", "--name", help="Create a new collection with this name."),
    watch: bool = typer.Option(True, help="Wait for indexing to finish."),
) -> None:
    """Index videos into a collection so you can chat with them."""
    settings = get_settings()
    if settings.demo:
        _fail("Demo mode is read-only. Unset VPI_DEMO to ingest real videos.")

    try:
        paths = ingest_mod.expand_sources(sources)
    except FileNotFoundError as exc:
        _fail(str(exc))
        return
    if not paths:
        _fail("Nothing to upload — no video files found in that path.")

    with build_client(settings) as client:
        if name:
            target = client.create_collection(name).id
            console.print(f"created collection [bold]{target}[/bold] ({name})")
        else:
            try:
                target = collection or resolve_collection(client, settings)
            except MissingCollection as exc:
                _fail(f"{exc}\n\nOr pass --name to create one: vpi ingest --name my-memory <path>")
                return

        console.print(f"uploading {len(paths)} item(s) to {target}")
        console.print(
            "[grey62]indexing is billed per minute of video "
            f"(${ingest_mod.estimate_cost_usd(1):.2f}/min at default fps)[/grey62]"
        )
        items = []
        for source in paths:
            item = ingest_mod.submit(client, target, source)
            label = Path(source).name if not source.startswith("http") else source
            if item.error:
                console.print(f"  [red]✗ {label}: {item.error}[/red]")
            else:
                console.print(f"  ↑ {label} → {item.video_id}")
            items.append(item)

        if not watch:
            console.print("\nnot waiting. `vpi collections` will show them when ready.")
            return

        console.print("\nwaiting for indexing (preprocess → index → derive)…")
        ok = 0
        for item in ingest_mod.wait(client, items):
            if item.ok:
                ok += 1
                console.print(f"  [green]✓ {item.video_id} ready[/green]")
            else:
                console.print(f"  [red]✗ {item.source}: {item.error}[/red]")
        console.print(f"\n{ok}/{len(items)} ready. Now run: vpi chat")
        if ok:
            console.print(f"[grey62]VPI_COLLECTION_ID={target}[/grey62]")


@app.command()
def chat(
    collection: str = typer.Option("", "--collection", "-c"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip the relevance pass."),
) -> None:
    """Open a terminal chat with your video memory."""
    try:
        session = build_session(collection_id=collection or None, verify_relevance=not no_verify)
    except MissingCollection as exc:
        _fail(str(exc))
        return
    try:
        run_chat(session, console=console)
    finally:
        session.close()


@app.command()
def ask(
    question: str = typer.Argument(...),
    collection: str = typer.Option("", "--collection", "-c"),
) -> None:
    """Ask one question and exit."""
    from vpi.chat.cli import render_events

    try:
        session = build_session(collection_id=collection or None)
    except MissingCollection as exc:
        _fail(str(exc))
        return
    try:
        render_events(session.agent, question, console)
    finally:
        session.close()


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        help="Bind address. Localhost by default — this server has no auth and reads your memory.",
    ),
    port: int = typer.Option(8033),
) -> None:
    """Serve the web chat UI and its streaming API."""
    import uvicorn

    console.print(f"vpi on http://{host}:{port}")
    uvicorn.run("vpi.server.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
