"""The grounding gate.

The agent may only assert what the ledger supports. This module does not trust
the model to have followed that instruction — it checks.

A sentence passes if it cites at least one evidence id that actually exists. Two
kinds of sentence are exempt: statements that no evidence was found, and
questions back to the user. Everything else that carries no valid citation is
either repaired (one retry, asking for citations) or cut.

A citation of an id that is not in the ledger is worse than no citation — it is
a fabricated source — so it is stripped and the sentence treated as uncited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[(E\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")

# Sentences that make no claim about the videos: an admission of absence, or a
# question. Kept deliberately short — when in doubt, a sentence needs a citation.
_NO_EVIDENCE_MARKERS = (
    "couldn't find",
    "could not find",
    "no evidence",
    "nothing in",
    "did not find",
    "didn't find",
    "no matching",
    "no results",
    "not in your",
    "没有找到",
    "未找到",
    "没有证据",
)

NO_EVIDENCE_ANSWER = (
    "I couldn't find anything in your indexed videos that supports an answer to that. "
    "Try a narrower time range, a different phrasing, or check that the relevant "
    "video is in this collection and finished indexing."
)


@dataclass
class GateResult:
    text: str
    cited_ids: set[str] = field(default_factory=set)
    dropped: list[str] = field(default_factory=list)
    invalid_ids: set[str] = field(default_factory=set)
    grounded: bool = True

    @property
    def needs_repair(self) -> bool:
        return bool(self.dropped or self.invalid_ids)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def citations_in(text: str) -> set[str]:
    return set(CITATION_RE.findall(text))


def is_exempt(sentence: str) -> bool:
    lowered = sentence.lower()
    if sentence.rstrip().endswith(("?", "？")):
        return True
    return any(marker in lowered for marker in _NO_EVIDENCE_MARKERS)


def check(text: str, valid_ids: set[str]) -> GateResult:
    """Report which sentences are grounded, without changing the text."""
    invalid: set[str] = set()
    cited: set[str] = set()
    dropped: list[str] = []

    for sentence in split_sentences(text):
        ids = citations_in(sentence)
        bad = ids - valid_ids
        good = ids & valid_ids
        invalid |= bad
        cited |= good
        if good or is_exempt(sentence):
            continue
        dropped.append(sentence)

    return GateResult(
        text=text,
        cited_ids=cited,
        dropped=dropped,
        invalid_ids=invalid,
        grounded=not dropped and not invalid,
    )


def enforce(text: str, valid_ids: set[str]) -> GateResult:
    """Cut what cannot be supported. Returns the answer the user should see."""
    result = check(text, valid_ids)
    if result.grounded:
        return result

    kept: list[str] = []
    for sentence in split_sentences(text):
        cleaned = _strip_invalid(sentence, valid_ids)
        ids = citations_in(cleaned) & valid_ids
        if ids or is_exempt(cleaned):
            kept.append(cleaned)

    surviving = " ".join(kept).strip()
    if not surviving:
        surviving = NO_EVIDENCE_ANSWER

    return GateResult(
        text=surviving,
        cited_ids=citations_in(surviving) & valid_ids,
        dropped=result.dropped,
        invalid_ids=result.invalid_ids,
        grounded=False,
    )


def _strip_invalid(sentence: str, valid_ids: set[str]) -> str:
    """Remove citations of ids that do not exist, keep the real ones."""

    def replace(match: re.Match[str]) -> str:
        eid = match.group(1)
        return f" [{eid}]" if eid in valid_ids else ""

    return re.sub(r"\s*\[(E\d+)\]", replace, sentence).strip()


REPAIR_INSTRUCTION = (
    "Your previous answer contained statements with no supporting evidence id.\n"
    "Unsupported sentences:\n{unsupported}\n"
    "{invalid_note}"
    "Rewrite the answer. Every sentence that states something about the user's "
    "videos must end with at least one citation like [E3], and every id must be one "
    "that appears in the evidence list above. Delete any claim you cannot cite — do "
    "not soften it, do not guess. If nothing in the evidence answers the question, "
    "say plainly that you could not find it."
)


def repair_prompt(result: GateResult) -> str:
    unsupported = "\n".join(f"- {s}" for s in result.dropped) or "- (none)"
    invalid_note = ""
    if result.invalid_ids:
        invalid_note = (
            "You also cited ids that do not exist: "
            + ", ".join(sorted(result.invalid_ids))
            + ". Never invent an evidence id.\n"
        )
    return REPAIR_INSTRUCTION.format(unsupported=unsupported, invalid_note=invalid_note)
