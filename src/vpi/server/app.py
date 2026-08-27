"""FastAPI app: a streaming chat endpoint and a signed-URL refresher.

Deliberately single-user and localhost-first. There is no auth layer because
there is no multi-user model to protect — binding this to a public interface
would expose your video memory to anyone who can reach the port, and the CLI
says so.

Signed URLs from the datalake expire (15 min for search thumbnails, 5 h for
clips), so the browser never gets a stored link: it asks `/api/media` for a
fresh one when it needs to play something.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vpi.config import get_settings
from vpi.datalake.errors import DataLakeError
from vpi.session import MissingCollection, Session, build_session

MAX_SESSIONS = 32
UI_DIST = Path(__file__).resolve().parents[3] / "ui" / "dist"

app = FastAPI(title="vpi", version="0.1.0", docs_url="/api/docs")


@dataclass
class _Store:
    """A tiny LRU of live conversations. Restarting the server forgets them."""

    sessions: OrderedDict[str, Session]

    def get(self, session_id: str) -> Session:
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)
            return self.sessions[session_id]
        session = build_session()
        self.sessions[session_id] = session
        while len(self.sessions) > MAX_SESSIONS:
            _, evicted = self.sessions.popitem(last=False)
            evicted.close()
        return session

    def drop(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True


store = _Store(sessions=OrderedDict())


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "ok": True,
        "demo": settings.demo,
        "model": settings.llm_model,
        "timezone": settings.timezone,
        "ui_built": UI_DIST.exists(),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """Stream one question's run as server-sent events."""
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "question is empty")
    try:
        session = store.get(request.session_id)
    except MissingCollection as exc:
        raise HTTPException(409, str(exc)) from exc

    def events():
        try:
            for event in session.agent.ask(question):
                yield _sse({"kind": event.kind, "text": event.text, **event.data})
        except DataLakeError as exc:
            yield _sse({"kind": "error", "text": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the stream reports, it does not crash
            yield _sse({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/session/{session_id}")
def reset(session_id: str) -> dict[str, bool]:
    return {"dropped": store.drop(session_id)}


@app.get("/api/media")
def media(ref: str, kind: str = "clip") -> dict[str, str]:
    """Mint a fresh signed URL for a moment. Links expire; we never cache them."""
    session = store.get("default")
    video_id, _, span = ref.partition("@")
    try:
        if kind == "clip" and span and "-" in span:
            start, end = (float(x) for x in span.split("-", 1))
            return {"url": session.client.get_clip_url(video_id, start, end)}
        if kind == "frame":
            t = float(span.split("-", 1)[0]) if span else 0.0
            return {"url": session.client.get_frame_url(video_id, t)}
    except DataLakeError as exc:
        raise HTTPException(502, str(exc)) from exc
    raise HTTPException(400, f"cannot serve kind={kind!r} for ref={ref!r}")


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(UI_DIST / "index.html")

else:

    @app.get("/")
    def index_missing() -> dict[str, str]:
        return {
            "message": "UI is not built. Run `npm install && npm run build` in ui/, "
            "or use `vpi chat` in the terminal.",
        }
