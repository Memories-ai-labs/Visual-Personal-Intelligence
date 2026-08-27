import { useEffect, useRef, useState } from "react";

type Citation = {
  eid: string;
  ref: string;
  video_id: string;
  start: number;
  end: number;
  text: string;
  thumbnail_url?: string;
  clip_url?: string;
  tags?: string[];
};

type StreamEvent = {
  kind: string;
  text?: string;
  citations?: Citation[];
  is_error?: boolean;
  name?: string;
  arguments?: Record<string, unknown>;
};

type Turn = {
  question: string;
  trace: { text: string; level: "info" | "warn" | "err" }[];
  answer: string;
  citations: Citation[];
  cost: string;
  running: boolean;
};

const SESSION_ID = `web-${Math.random().toString(36).slice(2, 10)}`;

/** Turn "…mixed sanitiser [E1]." into text plus clickable citation chips. */
function renderAnswer(answer: string, onJump: (eid: string) => void) {
  const parts = answer.split(/(\[E\d+\])/g);
  return parts.map((part, index) => {
    const match = /^\[(E\d+)\]$/.exec(part);
    if (!match) return <span key={index}>{part}</span>;
    const eid = match[1];
    return (
      <button key={index} className="cite" onClick={() => onJump(eid)} title={`jump to ${eid}`}>
        {eid}
      </button>
    );
  });
}

function EvidenceCard({ citation, flash }: { citation: Citation; flash: boolean }) {
  const [clip, setClip] = useState(citation.clip_url || "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function play() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `/api/media?ref=${encodeURIComponent(citation.ref)}&kind=clip`,
      );
      if (!response.ok) throw new Error(await response.text());
      const body = await response.json();
      if (!body.url) throw new Error("no clip available for this moment");
      setClip(body.url);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }

  const span = `${citation.start.toFixed(1)}–${citation.end.toFixed(1)}s`;
  const playable = citation.end > citation.start;

  return (
    <div className={flash ? "card flash" : "card"} id={`ev-${citation.eid}`}>
      <div>
        <span className="eid">{citation.eid}</span>{" "}
        <span className="ref">
          {citation.video_id} · {span}
          {citation.tags?.length ? ` · ${citation.tags.join(" / ")}` : ""}
        </span>
        <div className="text">{citation.text}</div>
        {error ? <div className="ref" style={{ color: "var(--err)" }}>{error}</div> : null}
      </div>
      {playable && !clip ? (
        <button onClick={play} disabled={loading}>
          {loading ? "…" : "play clip"}
        </button>
      ) : (
        <span />
      )}
      {clip ? <video src={clip} controls preload="metadata" /> : null}
    </div>
  );
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<{ demo?: boolean; model?: string; timezone?: string }>({});
  const [flashed, setFlashed] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({}));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  function jump(eid: string) {
    document.getElementById(`ev-${eid}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlashed(eid);
    setTimeout(() => setFlashed(""), 1200);
  }

  function patchLast(update: (turn: Turn) => Turn) {
    setTurns((current) =>
      current.map((turn, index) => (index === current.length - 1 ? update(turn) : turn)),
    );
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const asked = question.trim();
    if (!asked || busy) return;
    setQuestion("");
    setBusy(true);
    setTurns((current) => [
      ...current,
      { question: asked, trace: [], answer: "", citations: [], cost: "", running: true },
    ]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: asked, session_id: SESSION_ID }),
      });
      if (!response.ok || !response.body) {
        throw new Error(`server said ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const line = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6)) as StreamEvent;
          patchLast((turn) => applyEvent(turn, event));
        }
      }
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      patchLast((turn) => ({
        ...turn,
        trace: [...turn.trace, { text: message, level: "err" }],
      }));
    } finally {
      patchLast((turn) => ({ ...turn, running: false }));
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>your video memory</h1>
        <p>
          {health.demo ? "demo fixtures" : "your collection"} · {health.model ?? "…"} ·{" "}
          {health.timezone ?? "UTC"} · every claim is cited to a moment you can play
        </p>
      </header>

      {turns.length === 0 ? (
        <div className="empty">
          <p>Ask about something you recorded. For example:</p>
          <ul>
            <li>what was I saying about the sanitizer mix?</li>
            <li>what did we decide about pricing, and who owns the follow-up?</li>
            <li>what was on screen during Monday's stand-up?</li>
          </ul>
          <p>
            No videos yet? Index some with <code>vpi ingest ~/Videos --name my-memory</code>, or
            run with <code>VPI_DEMO=1</code> to try the fixtures.
          </p>
        </div>
      ) : null}

      {turns.map((turn, index) => (
        <div className="turn" key={index}>
          <div className="question">{turn.question}</div>
          {turn.trace.length ? (
            <details className="trace" open={turn.running}>
              <summary>{turn.running ? "working…" : `${turn.trace.length} steps`}</summary>
              {turn.trace.map((line, i) => (
                <div key={i} className={line.level === "info" ? "" : line.level}>
                  {line.text}
                </div>
              ))}
            </details>
          ) : null}
          {turn.answer ? (
            <div className="answer">{renderAnswer(turn.answer, jump)}</div>
          ) : null}
          {turn.citations.length ? (
            <div className="evidence">
              {turn.citations.map((citation) => (
                <EvidenceCard
                  key={citation.eid}
                  citation={citation}
                  flash={flashed === citation.eid}
                />
              ))}
            </div>
          ) : null}
          {turn.cost ? <div className="cost">{turn.cost}</div> : null}
        </div>
      ))}

      <div ref={bottom} />

      <form onSubmit={ask}>
        <div className="inner">
          <input
            type="text"
            value={question}
            placeholder="what do you want to remember?"
            onChange={(e) => setQuestion(e.target.value)}
            autoFocus
          />
          <button type="submit" disabled={busy || !question.trim()}>
            {busy ? "…" : "ask"}
          </button>
        </div>
      </form>
    </div>
  );
}

function applyEvent(turn: Turn, event: StreamEvent): Turn {
  switch (event.kind) {
    case "tool_call":
      return { ...turn, trace: [...turn.trace, { text: `→ ${event.text}`, level: "info" }] };
    case "tool_result": {
      const first = (event.text ?? "").split("\n")[0].slice(0, 140);
      return {
        ...turn,
        trace: [...turn.trace, { text: `← ${first}`, level: event.is_error ? "err" : "info" }],
      };
    }
    case "note":
      return { ...turn, trace: [...turn.trace, { text: `· ${event.text}`, level: "info" }] };
    case "warning":
      return { ...turn, trace: [...turn.trace, { text: `· ${event.text}`, level: "warn" }] };
    case "error":
      return { ...turn, trace: [...turn.trace, { text: event.text ?? "error", level: "err" }] };
    case "answer":
      return { ...turn, answer: event.text ?? "" };
    case "citations":
      return { ...turn, citations: event.citations ?? [] };
    case "cost":
      return { ...turn, cost: event.text ?? "" };
    default:
      return turn;
  }
}
