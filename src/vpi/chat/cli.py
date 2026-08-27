"""Terminal chat."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from vpi.agent.loop import Agent
from vpi.session import Session

DIM = "grey62"


def render_events(agent: Agent, question: str, console: Console) -> None:
    for event in agent.ask(question):
        if event.kind == "tool_call":
            console.print(f"  [{DIM}]→ {event.text}[/{DIM}]")
        elif event.kind == "tool_result":
            first = event.text.splitlines()[0] if event.text else ""
            evidence = event.data.get("evidence") or []
            tail = f" (+{len(evidence)} evidence)" if evidence else ""
            style = "red" if event.data.get("is_error") else DIM
            console.print(f"  [{style}]← {first[:110]}{tail}[/{style}]")
        elif event.kind in ("note", "warning"):
            colour = "yellow" if event.kind == "warning" else DIM
            console.print(f"  [{colour}]· {event.text}[/{colour}]")
        elif event.kind == "answer":
            console.print()
            console.print(Markdown(event.text))
        elif event.kind == "citations":
            console.print()
            for citation in event.data.get("citations", []):
                span = f"{citation['start']:.1f}-{citation['end']:.1f}s"
                console.print(
                    f"  [{DIM}][{citation['eid']}] {citation['video_id']} {span}"
                    f" — {citation['text'][:90]}[/{DIM}]"
                )
        elif event.kind == "cost":
            console.print(f"\n  [{DIM}]{event.text}[/{DIM}]\n")


def run_chat(session: Session, *, console: Console | None = None) -> None:
    console = console or Console()
    where = "demo fixtures" if session.settings.demo else session.collection_id
    console.print(
        Panel(
            f"Chatting with [bold]{where}[/bold]\n"
            f"model {session.settings.llm_model} · timezone {session.settings.timezone}\n"
            f"[{DIM}]Every claim is cited to a moment in your videos. "
            f"Ctrl-C or /exit to leave.[/{DIM}]",
            title="vpi",
            border_style=DIM,
        )
    )

    while True:
        try:
            question = console.input("\n[bold]you[/bold] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        if not question:
            continue
        if question in ("/exit", "/quit", "exit", "quit"):
            console.print("bye")
            return
        if question == "/evidence":
            console.print(session.ledger.render() or "(empty)")
            continue
        if question == "/cost":
            console.print(session.cost.summary())
            continue

        try:
            render_events(session.agent, question, console)
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]interrupted[/{DIM}]")
        except Exception as exc:  # noqa: BLE001 - a chat loop must survive one bad turn
            console.print(f"  [red]{type(exc).__name__}: {exc}[/red]")
