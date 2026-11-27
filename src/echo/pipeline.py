"""The dub, stage by stage. Every stage writes its result into the output folder,
so a run can be resumed and a person can correct any line before re-voicing it.

    source.wav ─ separate ─► vocals.wav + background.wav
    vocals ─ find lines ─ transcribe ─ group by voice ─► segments.json
    segments ─ translate to fit the time ─► segments.json (text, emotion)
    segments ─ voice each line in its own speaker's voice ─ fit the slot ─► lines/
    lines + background ─ level-match and mix ─► dub.wav, dub.mp4, dub.<lang>.srt, report.html
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import EchoError, media, mix, report, segment, syllables
from .llm import LLM, LLMError
from .translate import budget, shorten, translate

SR = 44100
LANGS = {"te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam",
         "en": "English", "bn": "Bengali", "mr": "Marathi"}


@dataclass
class Options:
    source: str = "te"
    target: str = "en"
    transcript: Path | None = None
    separate: bool = True
    speakers: int | None = None      # None: find them
    rate: float = 5.0                # Chatterbox speaks about 4.7 to 5.8 syllables a second once trimmed; re-measured per speaker after the first pass
    rounds: int = 2
    resume: bool = False
    device: str = "auto"
    asr_model: str | None = None
    log: Callable[[str], None] = print
    timings: dict = field(default_factory=dict)


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y, sr


def _write(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="FLOAT")


def delivery_hints(segs: list[dict], vocals_mono: np.ndarray, sr: int) -> None:
    """Mark lines that are clearly louder, softer, faster or slower than this speaker's usual."""
    levels = [mix.db(vocals_mono[int(s["start"] * sr):int(s["end"] * sr)]) for s in segs]
    paces = [len(s["source"].replace(" ", "")) / max(0.3, s["end"] - s["start"]) for s in segs]
    lv, pc = float(np.median(levels)), float(np.median(paces))
    for s, l, p in zip(segs, levels, paces):
        hints = []
        if l > lv + 5:
            hints.append("loud")
        elif l < lv - 7:
            hints.append("soft")
        if p > pc * 1.35:
            hints.append("fast")
        elif p < pc * 0.65:
            hints.append("slow")
        s["delivery"] = ", ".join(hints)


def reference_clip(segs: list[dict], vocals_mono: np.ndarray, sr: int, speaker: str, seconds: float = 10.0) -> np.ndarray:
    """A speaker's longest clean lines, joined, as the voice to clone."""
    mine = sorted((s for s in segs if s["speaker"] == speaker), key=lambda s: s["end"] - s["start"], reverse=True)
    parts, total = [], 0.0
    gap = np.zeros(int(0.15 * sr), dtype=np.float32)
    for s in mine:
        clip = vocals_mono[int(s["start"] * sr):int(s["end"] * sr)]
        parts += [clip, gap]
        total += len(clip) / sr
        if total >= seconds:
            break
    return np.concatenate(parts) if parts else np.zeros(sr, dtype=np.float32)


class Dub:
    def __init__(self, src: Path, out: Path, llm: LLM | None, opt: Options):
        self.src, self.out, self.llm, self.opt = Path(src), Path(out), llm, opt
        self.work = self.out / "work"
        self.seg_path = self.out / "segments.json"
        self.log = opt.log

    def _stage(self, name: str):
        t = time.time()
        self.log(f"· {name}")
        return lambda: self.opt.timings.__setitem__(name, round(time.time() - t, 1))

    # 1. audio in, voices apart -----------------------------------------
    def prepare(self) -> dict:
        info = media.probe(self.src)
        self.info = info
        src_wav = self.work / "source.wav"
        if not (self.opt.resume and src_wav.exists()):
            self.work.mkdir(parents=True, exist_ok=True)
            media.extract_audio(self.src, src_wav, SR, 2)
        vox, bg = self.work / "vocals.wav", self.work / "background.wav"
        if not (self.opt.resume and vox.exists() and bg.exists()):
            done = self._stage("separating voice from music and effects")
            y, sr = _read(src_wav)
            if self.opt.separate:
                from . import models
                v, b = models.separate(y, sr, models.device(self.opt.device))
            else:
                v, b = y, np.zeros_like(y)
            _write(vox, v, sr)
            _write(bg, b, sr)
            done()
        return info

    # 2. what is said, by whom ------------------------------------------
    def lines(self) -> list[dict]:
        if self.opt.resume and self.seg_path.exists():
            data = json.loads(self.seg_path.read_text(encoding="utf-8"))
            self.log(f"· reusing {len(data['segments'])} lines from {self.seg_path.name}")
            return data["segments"]
        from . import models
        dev = models.device(self.opt.device)
        v, sr = _read(self.work / "vocals.wav")
        mono = v.mean(axis=1)
        mono16 = models.resample(mono, sr, 16000)

        if self.opt.transcript:
            cues = segment.read_srt(Path(self.opt.transcript).read_text(encoding="utf-8"))
            segs = [{"id": i + 1, "start": c["start"], "end": c["end"], "source": c["text"]} for i, c in enumerate(cues)]
            self.log(f"· {len(segs)} lines from {Path(self.opt.transcript).name}")
        else:
            done = self._stage("finding and transcribing lines")
            regions = segment.find_lines(mono16, 16000)
            asr = models.Transcriber(self.opt.source, dev, self.opt.asr_model)
            segs = []
            for r in regions:
                text = asr(mono16[int(r.start * 16000):int(r.end * 16000)])
                if text:
                    segs.append({"id": len(segs) + 1, "start": round(r.start, 3), "end": round(r.end, 3), "source": text})
            asr.close()
            done()
        if not segs:
            raise EchoError("no speech found")

        done = self._stage("telling voices apart")
        if self.opt.speakers == 1 or len(segs) == 1:
            labels = [0] * len(segs)
        else:
            enc = models.SpeakerEncoder(dev)
            emb = enc([mono16[int(s["start"] * 16000):int(s["end"] * 16000)] for s in segs])
            enc.close()
            labels = segment.cluster(emb, n=self.opt.speakers)
        for s, l in zip(segs, labels):
            s["speaker"] = f"S{l + 1}"
        delivery_hints(segs, mono, sr)
        done()
        return segs

    def references(self, segs: list[dict]) -> dict[str, Path]:
        v, sr = _read(self.work / "vocals.wav")
        mono = v.mean(axis=1)
        refs = {}
        for spk in sorted({s["speaker"] for s in segs}):
            p = self.work / "refs" / f"{spk}.wav"
            if not (self.opt.resume and p.exists()):
                _write(p, reference_clip(segs, mono, sr, spk)[:, None], sr)
            refs[spk] = p
        return refs

    # 3. words that fit the time ----------------------------------------
    def translate(self, segs: list[dict]) -> None:
        for s in segs:
            s["budget"] = budget(s["end"] - s["start"], self.opt.rate)
        if all(s.get("text") for s in segs):
            return
        if self.llm is None:
            raise EchoError("no language model to translate with; pass --llm or edit segments.json")
        done = self._stage("translating to fit the time")
        translate(segs, self.llm, LANGS.get(self.opt.source, self.opt.source),
                  LANGS.get(self.opt.target, self.opt.target), self.opt.rounds, lambda m: self.log("  " + m))
        done()

    # 4. the voice --------------------------------------------------------
    def voice(self, segs: list[dict], refs: dict[str, Path], total: float) -> dict[int, np.ndarray]:
        from . import models
        dev = models.device(self.opt.device)
        slot = dict(zip((s["id"] for s in segs), mix.slots(segs, total)))
        done = self._stage("voicing each line in its speaker's voice")
        voice = models.Voice(dev)
        for spk, p in refs.items():
            voice.add_speaker(spk, str(p))
        audio: dict[int, np.ndarray] = {}

        def run(ids):
            for s in segs:
                if s["id"] in ids and s.get("text"):
                    st = models.style(s.get("emotion", "neutral"), float(s.get("intensity", 0.5)), s.get("delivery", ""))
                    s["style"] = st.__dict__
                    y = mix.trim_silence(voice.say(s["text"], s["speaker"], st, seed=s["id"]), voice.sr)
                    audio[s["id"]] = y
                    self.log(f"  line {s['id']}: {len(y) / voice.sr:.2f}s for a {slot[s['id']]:.2f}s slot")

        run({s["id"] for s in segs})
        vsr = voice.sr

        # Measure how fast this voice really speaks, then send lines that ran long back to the writer.
        rates = {}
        for spk in refs:
            n = sum(syllables.count(s["text"]) for s in segs if s["speaker"] == spk and s["id"] in audio)
            t = sum(len(audio[s["id"]]) / vsr for s in segs if s["speaker"] == spk and s["id"] in audio)
            rates[spk] = n / t if t else self.opt.rate
        self.rates = rates
        long = {s["id"]: max(1, math.floor(slot[s["id"]] * rates[s["speaker"]] * 1.05))
                for s in segs if s["id"] in audio and len(audio[s["id"]]) / vsr > slot[s["id"]] * mix.ASK_SHORTER}
        if long and self.llm is not None:
            self.log(f"  lines {', '.join(map(str, long))} ran long; asking for shorter wording")
            voice.close()  # free the GPU for a local model
            try:
                new = shorten(segs, long, self.llm, LANGS.get(self.opt.target, self.opt.target))
            except LLMError as e:
                self.log(f"  could not shorten: {e}")
                new = {}
            voice = models.Voice(dev)
            for spk, p in refs.items():
                voice.add_speaker(spk, str(p))
            if new:
                for s in segs:
                    if s["id"] in new:
                        s["first_draft"] = s["text"]
                        s.update(new[s["id"]])
                run(set(new))
        voice.close()
        self.vsr = vsr
        done()
        return audio

    # 5. fit and mix ------------------------------------------------------
    def mixdown(self, segs: list[dict], audio: dict[int, np.ndarray], total: float) -> Path:
        from . import models
        done = self._stage("fitting and mixing")
        v, sr = _read(self.work / "vocals.wav")
        vocals = v.mean(axis=1)
        bg, _ = _read(self.work / "background.wav")
        track = np.zeros(len(bg), dtype=np.float32)
        lines_dir = self.work / "lines"
        for s, sl in zip(segs, mix.slots(segs, total)):
            y = audio.get(s["id"])
            s["syllables"] = syllables.count(s.get("text", ""))
            if y is None or y.size == 0:
                continue
            f = mix.plan_fit(len(y) / self.vsr, sl)
            raw = lines_dir / f"{s['id']:04d}.wav"
            _write(raw, y[:, None], self.vsr)
            if f.tempo > 1.0:
                fast = lines_dir / f"{s['id']:04d}.fit.wav"
                media.change_tempo(raw, fast, f.tempo)
                y, _ = _read(fast)
                y = y[:, 0]
            y = models.resample(y, self.vsr, sr)
            y = mix.match_level(y, vocals[int(s["start"] * sr):int(s["end"] * sr)])
            mix.place(track, y, s["start"], sr, limit=sl)
            s["fit"] = {"slot": round(f.slot, 3), "natural": round(f.natural, 3), "tempo": f.tempo, "cut": f.cut}
        out = mix.combine(bg, track)
        dub = self.out / "dub.wav"
        _write(dub, out, sr)
        _write(self.work / "dub_voice.wav", track[:, None], sr)
        done()
        return dub

    def save(self, segs: list[dict], extra: dict | None = None) -> None:
        data = {"input": str(self.src.name), "source_language": self.opt.source,
                "target_language": self.opt.target, **(extra or {}), "segments": segs}
        self.seg_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def run(self) -> dict:
        info = self.prepare()
        segs = self.lines()
        self.save(segs)
        refs = self.references(segs)
        self.translate(segs)
        self.save(segs)
        audio = self.voice(segs, refs, info["duration"])
        dub = self.mixdown(segs, audio, info["duration"])
        final = dub
        if info["video"]:
            final = media.mux(self.src, dub, self.out / "dub.mp4")
        lang = self.opt.target
        (self.out / f"dub.{lang}.srt").write_text(segment.write_srt(segs, "text"), encoding="utf-8")
        (self.out / f"original.{self.opt.source}.srt").write_text(segment.write_srt(segs, "source"), encoding="utf-8")
        meta = {
            "input": self.src.name,
            "source_language": LANGS.get(self.opt.source, self.opt.source),
            "target_language": LANGS.get(lang, lang),
            "speakers": f"{len(refs)} voice{'s' if len(refs) != 1 else ''}",
            "llm": self.llm.describe() if self.llm else "translations from segments.json",
            "speaking_rate": {k: round(v, 2) for k, v in getattr(self, "rates", {}).items()},
            "timings_s": self.opt.timings,
        }
        self.save(segs, {k: meta[k] for k in ("llm", "speaking_rate", "timings_s")})
        (self.out / "report.html").write_text(report.html(meta, segs, final.name), encoding="utf-8")
        return {"output": final, "segments": segs, "meta": meta}
