"""Find the spoken lines in a vocal track, read and write subtitle files.

Lines are found by energy on the separated vocal stem, which is clean enough
that a level threshold works. Short pauses inside a sentence are bridged, and
very long stretches are cut at their quietest point, so each line is one
breath group an actor would deliver in one go.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge(regions: list[Region], min_gap: float) -> list[Region]:
    """Join regions separated by less than ``min_gap`` seconds."""
    out: list[Region] = []
    for r in sorted(regions, key=lambda r: r.start):
        if out and r.start - out[-1].end < min_gap:
            out[-1].end = max(out[-1].end, r.end)
        else:
            out.append(Region(r.start, r.end))
    return out


def split_long(regions: list[Region], energy: np.ndarray, hop: float, max_len: float) -> list[Region]:
    """Cut regions longer than ``max_len`` at the quietest frame in their middle half."""
    out: list[Region] = []
    stack = list(regions)
    while stack:
        r = stack.pop(0)
        if r.duration <= max_len:
            out.append(r)
            continue
        a = int((r.start + r.duration * 0.25) / hop)
        b = int((r.start + r.duration * 0.75) / hop)
        b = max(a + 1, min(b, len(energy)))
        cut = (a + int(np.argmin(energy[a:b]))) * hop if a < len(energy) else r.start + r.duration / 2
        stack[:0] = [Region(r.start, cut), Region(cut, r.end)]
    return out


def find_lines(
    y: np.ndarray,
    sr: int,
    top_db: float = 35.0,
    min_gap: float = 0.45,
    min_len: float = 0.5,
    max_len: float = 12.0,
    pad: float = 0.08,
) -> list[Region]:
    """Spoken lines in a mono signal, as (start, end) seconds."""
    hop_len = 512
    frame = 2048
    if y.size == 0:
        return []
    cs = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    centers = np.arange(0, len(y), hop_len)
    lo = np.clip(centers - frame // 2, 0, len(y))
    hi = np.clip(centers + frame // 2, 0, len(y))
    rms = np.sqrt(np.maximum(cs[hi] - cs[lo], 0) / frame)
    if rms.max() <= 0:
        return []
    ref = float(np.percentile(rms[rms > 0], 95))  # one shout should not hide quiet lines
    loud = 20 * np.log10(np.maximum(rms, 1e-10) / ref) > -top_db
    hop = hop_len / sr
    regions, start = [], None
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append(Region(start * hop, i * hop))
            start = None
    if start is not None:
        regions.append(Region(start * hop, len(loud) * hop))
    regions = merge(regions, min_gap)
    regions = split_long(regions, rms, hop, max_len)
    total = len(y) / sr
    return [
        Region(max(0.0, r.start - pad), min(total, r.end + pad))
        for r in regions if r.duration >= min_len
    ]


# --- subtitles -------------------------------------------------------------

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _seconds(stamp: str) -> float:
    m = _TIME.search(stamp)
    if not m:
        raise ValueError(f"bad SRT time {stamp!r}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def stamp(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def read_srt(text: str) -> list[dict]:
    """Subtitle cues as dicts with start, end and text."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        arrow = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if arrow is None:
            continue
        a, b = lines[arrow].split("-->")
        body = " ".join(ln.strip() for ln in lines[arrow + 1:])
        body = re.sub(r"<[^>]+>", "", body).strip()
        if body:
            cues.append({"start": _seconds(a), "end": _seconds(b), "text": body})
    return cues


def write_srt(cues: list[dict], key: str = "text") -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{c.get(key, '')}\n")
    return "\n".join(out)


# --- speakers --------------------------------------------------------------

def cluster(embeddings: np.ndarray, threshold: float = 0.72, n: int | None = None) -> list[int]:
    """Group voice embeddings into speakers by average-linkage cosine similarity.

    With ``n`` set, merge until exactly n groups remain; otherwise merge while
    the two closest groups are more similar than ``threshold``. Labels are
    numbered by first appearance.
    """
    k = len(embeddings)
    if k == 0:
        return []
    e = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    sim = e @ e.T
    groups = [[i] for i in range(k)]
    target = max(1, n) if n else 1
    while len(groups) > target:
        best, pair = -2.0, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                s = float(sim[np.ix_(groups[i], groups[j])].mean())
                if s > best:
                    best, pair = s, (i, j)
        if n is None and best < threshold:
            break
        i, j = pair
        groups[i] += groups.pop(j)
    label = {}
    for gi, g in enumerate(groups):
        for m in g:
            label[m] = gi
    order, out = {}, []
    for m in range(k):
        out.append(order.setdefault(label[m], len(order)))
    return out
