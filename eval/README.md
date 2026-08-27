# Eval

Grounding, not vibes. Run:

```bash
VPI_DEMO=1 ANTHROPIC_API_KEY=sk-ant-... python eval/run_eval.py
```

Half the query set is questions the corpus **cannot** answer. Those rows pass
only if the agent refuses. A run that answers them all with confident prose
scores worse than one that says "I couldn't find it" — which is the point.

A row fails outright if the answer cites an evidence id that does not exist,
regardless of how right the prose looks.

Add your own set with `--queries eval/mine.json` and point it at real videos
with `--collection col_xxx`. Results land in `eval/results/last.md`.
