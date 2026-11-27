"""Timing and mixing: fit each dubbed line into its slot and lay it over the background."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_SPEEDUP = 1.3   # beyond this, sped-up speech starts to sound rushed
ASK_SHORTER = 1.25  # a line this much longer than its slot goes back to the writer
OVERHANG = 0.4      # a line may run this far past its end into silence
GAP = 0.05          # never closer than this to the next line


@dataclass
class Fit:
    slot: float     # seconds available
    natural: float  # seconds the voice took
    tempo: float    # speed-up applied (1.0 = none)
    final: float    # seconds after the speed-up
    cut: bool       # still too long: trimmed with a fade

    @property
    def ratio(self) -> float:
        return self.natural / self.slot if self.slot else 0.0


def slots(segs: list[dict], total: float) -> list[float]:
    """Seconds each line may use: its own time plus a little silence after it."""
    out = []
    for i, s in enumerate(segs):
        nxt = segs[i + 1]["start"] - GAP if i + 1 < len(segs) else total
        out.append(max(0.1, min(s["end"] + OVERHANG, nxt) - s["start"]))
    return out


def plan_fit(natural: float, slot: float) -> Fit:
    if natural <= slot:
        return Fit(slot, natural, 1.0, natural, False)
    tempo = min(natural / slot, MAX_SPEEDUP)
    final = natural / tempo
    return Fit(slot, natural, round(tempo, 4), final, final > slot + 1e-6)


def trim_silence(y: np.ndarray, sr: int, top_db: float = 40.0) -> np.ndarray:
    """Drop leading and trailing silence the voice model leaves around a line."""
    if y.size == 0:
        return y
    frame = max(1, int(0.02 * sr))
    n = len(y) // frame
    if n == 0:
        return y
    rms = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    on = np.nonzero(20 * np.log10(np.maximum(rms, 1e-10) / max(rms.max(), 1e-10)) > -top_db)[0]
    if on.size == 0:
        return y[:0]
    a = max(0, on[0] * frame - frame)
    b = min(len(y), (on[-1] + 2) * frame)
    return y[a:b]


def db(y: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(np.square(y, dtype=np.float64))), 1e-10))) if y.size else -200.0


def fade(y: np.ndarray, sr: int, seconds: float = 0.03) -> np.ndarray:
    n = min(len(y) // 2, int(seconds * sr))
    if n > 0:
        y = y.copy()
        ramp = np.linspace(0.0, 1.0, n, dtype=y.dtype)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def place(track: np.ndarray, line: np.ndarray, start: float, sr: int, limit: float | None = None) -> None:
    """Add ``line`` into mono ``track`` at ``start`` seconds, cut (with a fade) at ``limit`` seconds."""
    a = int(round(start * sr))
    if limit is not None:
        line = line[: max(0, int(limit * sr))]
    line = fade(line, sr)
    b = min(len(track), a + len(line))
    if b > a:
        track[a:b] += line[: b - a]


def match_level(line: np.ndarray, reference: np.ndarray, max_gain_db: float = 12.0) -> np.ndarray:
    """Scale a dubbed line to the loudness of the original line, so shouts stay loud and whispers soft."""
    if line.size == 0 or reference.size == 0:
        return line
    gain = np.clip(db(reference) - db(line), -max_gain_db, max_gain_db)
    return line * (10 ** (gain / 20))


def combine(background: np.ndarray, dub: np.ndarray) -> np.ndarray:
    """Background (samples, channels) plus the mono dub, kept clear of clipping."""
    out = background.astype(np.float32).copy()
    n = min(len(out), len(dub))
    out[:n] += dub[:n, None]
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
    return out
