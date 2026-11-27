"""Translate spoken lines so each one fits the time it has on screen.

The model writes; Echo measures. Every line gets a syllable budget from its
duration and the speaking rate of the dub voice. Lines that would run long or
leave the mouth moving with nothing to say go back to the model with their
counts, and the best version of each line from any round is kept.
"""

from __future__ import annotations

import json
from typing import Callable

from . import syllables
from .llm import LLM

EMOTIONS = ["neutral", "calm", "happy", "excited", "tender", "sad", "angry", "fearful", "surprised"]

SYSTEM = """You are a dubbing writer. You translate film and video dialogue so a voice \
actor can say it in the same time as the original line. Natural spoken {target} comes \
first: contractions, everyday words, the way people really talk. Keep the meaning, \
the names and the speaker's attitude; do not add information or explain.

Every line has a syllable budget. Aim for it; going over is worse than going under, \
because a long line has to be sped up. Translate each line on its own (its words must \
belong to its own time slot), but read the lines around it so pronouns and tone stay \
consistent. Telugu and other verb-final languages often hold the point of a sentence \
until the end: when a sentence runs across lines, keep each line's words in that line \
and make each line read naturally.

For each line also give how the original is performed: one emotion from the list, \
and an intensity from 0 (flat) to 1 (as strong as it gets). Use the words, the \
context and the delivery hints (loud, soft, fast) to judge it."""

LINE = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "text": {"type": "string"},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "intensity": {"type": "number"},
    },
    "required": ["id", "text", "emotion", "intensity"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": LINE}},
    "required": ["lines"],
    "additionalProperties": False,
}

LOW, HIGH = 0.75, 1.12  # a line between these fractions of its budget fits


def budget(seconds: float, rate: float) -> int:
    return max(1, round(seconds * rate))


def fit(text: str, target: int) -> float:
    return syllables.count(text) / max(1, target)


def _score(text: str, target: int) -> float:
    r = fit(text, target)
    # Long lines cost twice as much as short ones: they force a speed-up.
    return 2 * (r - 1) if r > 1 else (1 - r)


def _brief(segs: list[dict]) -> str:
    rows = []
    for s in segs:
        rows.append({
            "id": s["id"],
            "speaker": s.get("speaker", "S1"),
            "seconds": round(s["end"] - s["start"], 2),
            "syllable_budget": s["budget"],
            "delivery": s.get("delivery", ""),
            "original": s["source"],
        })
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


def draft_prompt(segs: list[dict], source: str, target: str) -> str:
    return (
        f"Dialogue in {source}, one JSON object per line, in order. Translate every "
        f"line into {target}.\n\n{_brief(segs)}\n\n"
        "Return every line with id, text, emotion and intensity."
    )


def revise_prompt(segs: list[dict], flagged: list[int], target: str) -> str:
    by_id = {s["id"]: s for s in segs}
    out = ["These lines do not fit their time. Rewrite only them, keeping the meaning.", ""]
    for i in flagged:
        s = by_id[i]
        n = syllables.count(s["text"])
        way = "too long: cut words" if n > s["budget"] else "too short: say a little more, naturally"
        out.append(f"Line {i} ({s['end'] - s['start']:.1f} s, budget {s['budget']} syllables)")
        out.append(f"  original: {s['source']}")
        out.append(f"  current:  {s['text']}  ({n} syllables, {way})")
        out.append("")
    out.append(f"Return only these lines, in {target}, with id, text, emotion and intensity.")
    return "\n".join(out)


def shorten_prompt(items: list[tuple[dict, int]], target: str) -> str:
    out = ["The voice took too long to say these lines. Say each one in at most the given "
           "number of syllables. Keep the core meaning; drop what the picture already shows.", ""]
    for s, limit in items:
        out.append(f"Line {s['id']}: {s['text']}  (now {syllables.count(s['text'])}, at most {limit})")
        out.append(f"  original: {s['source']}")
    out.append("")
    out.append(f"Return only these lines, in {target}, with id, text, emotion and intensity.")
    return "\n".join(out)


def _apply(segs: list[dict], reply: dict, only: set[int] | None = None) -> dict[int, dict]:
    by_id = {s["id"]: s for s in segs}
    got = {}
    for e in reply.get("lines", []):
        try:
            i = int(e["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if i not in by_id or (only is not None and i not in only):
            continue
        text = str(e.get("text", "")).strip().strip('"')
        if not text:
            continue
        emo = e.get("emotion") if e.get("emotion") in EMOTIONS else by_id[i].get("emotion", "neutral")
        try:
            inten = min(1.0, max(0.0, float(e.get("intensity", 0.5))))
        except (TypeError, ValueError):
            inten = 0.5
        got[i] = {"text": text, "emotion": emo, "intensity": inten}
    return got


def translate(
    segs: list[dict],
    llm: LLM,
    source: str = "Telugu",
    target: str = "English",
    rounds: int = 2,
    log: Callable[[str], None] = lambda s: None,
) -> list[dict]:
    """Fill ``text``, ``emotion`` and ``intensity`` on every segment (needs ``budget``)."""
    system = SYSTEM.format(target=target)
    log(f"translating {len(segs)} lines with {llm.describe()}")
    for i in range(0, len(segs), 40):  # long videos go in chunks the model can hold
        chunk = segs[i:i + 40]
        for k, v in _apply(chunk, llm.json(system, draft_prompt(chunk, source, target), SCHEMA)).items():
            next(s for s in chunk if s["id"] == k).update(v)
    for s in segs:
        s.setdefault("text", "")
        s.setdefault("emotion", "neutral")
        s.setdefault("intensity", 0.5)
    best = {s["id"]: dict(s) for s in segs}

    for r in range(1, rounds + 1):
        flagged = [s["id"] for s in segs
                   if not s["text"] or not LOW <= fit(s["text"], s["budget"]) <= HIGH]
        if not flagged:
            break
        log(f"round {r}: lines {', '.join(map(str, flagged))} do not fit, revising")
        reply = llm.json(system, revise_prompt(segs, flagged, target), SCHEMA)
        for k, v in _apply(segs, reply, set(flagged)).items():
            s = next(s for s in segs if s["id"] == k)
            if _score(v["text"], s["budget"]) < _score(best[k]["text"] or "", s["budget"]) or not best[k]["text"]:
                s.update(v)
                best[k] = dict(s)
    missing = [s["id"] for s in segs if not s["text"]]
    if missing:
        log(f"no translation for lines {missing}; they will stay silent")
    return segs


def shorten(segs: list[dict], limits: dict[int, int], llm: LLM, target: str = "English") -> dict[int, dict]:
    """Ask for shorter versions of lines whose audio ran long. Returns id -> new fields."""
    items = [(s, limits[s["id"]]) for s in segs if s["id"] in limits]
    if not items:
        return {}
    reply = llm.json(SYSTEM.format(target=target), shorten_prompt(items, target), SCHEMA)
    got = _apply(segs, reply, set(limits))
    return {k: v for k, v in got.items()
            if syllables.count(v["text"]) < syllables.count(next(s for s in segs if s["id"] == k)["text"])}
