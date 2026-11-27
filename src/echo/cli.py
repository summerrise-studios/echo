"""echo-dub: dub a video or audio file into another language in the speakers' own voices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import EchoError, __version__
from .llm import LLMError, pick

CONSENT = """Echo clones the voices in your file. Only dub voices you have the right to use:
your own, or people who agreed to have their voice cloned for this. Run again with
--consent to confirm that you do."""


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="echo-dub", description=__doc__)
    p.add_argument("input", type=Path, help="video or audio file")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: <input>.dub)")
    p.add_argument("--from", dest="source", default="te", help="spoken language, e.g. te, hi, ta (default te)")
    p.add_argument("--to", dest="target", default="en", help="dub language (default en; English is the supported voice today)")
    p.add_argument("--transcript", type=Path, help="an .srt of the original lines, to skip speech recognition")
    p.add_argument("--speakers", type=int, help="number of speakers, if you know it")
    p.add_argument("--no-separate", action="store_true", help="the file is clean speech: skip music separation")
    p.add_argument("--llm", default="auto", help="claude, ollama or auto (default)")
    p.add_argument("--model", help="language model name for the chosen backend")
    p.add_argument("--rounds", type=int, default=2, help="translation revise rounds (default 2)")
    p.add_argument("--rate", type=float, default=5.0, help="dub voice syllables per second, for the first draft (default 5.0)")
    p.add_argument("--resume", action="store_true", help="reuse finished stages; re-voice edited segments.json")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument("--asr-model", help="Hugging Face speech recognition model to use instead of the default")
    p.add_argument("--consent", action="store_true", help="confirm you may clone every voice in the file")
    p.add_argument("--version", action="version", version=f"echo {__version__}")
    a = p.parse_args(argv)

    if not a.consent:
        print(CONSENT, file=sys.stderr)
        return 2
    if a.target != "en":
        print("echo-dub: only English dubs are supported in this version (--to en)", file=sys.stderr)
        return 2
    if not a.input.exists():
        print(f"echo-dub: {a.input} not found", file=sys.stderr)
        return 2

    from .pipeline import Dub, Options

    out = a.out or a.input.with_name(a.input.stem + ".dub")
    llm = None
    try:
        # keep_alive=0 unloads a local model after each call, so the GPU is free for the voice.
        llm = pick(a.llm, a.model, keep_alive=0, think=False)
    except LLMError as e:
        if not (a.resume and (out / "segments.json").exists()):
            print(f"echo-dub: {e}", file=sys.stderr)
            return 2
    opt = Options(source=a.source, target=a.target, transcript=a.transcript, separate=not a.no_separate,
                  speakers=a.speakers, rate=a.rate, rounds=a.rounds, resume=a.resume, device=a.device,
                  asr_model=a.asr_model)
    try:
        result = Dub(a.input, out, llm, opt).run()
    except (EchoError, LLMError) as e:
        print(f"echo-dub: {e}", file=sys.stderr)
        return 1
    print(f"\ndub:    {result['output']}")
    print(f"review: {out / 'report.html'}")
    print(f"edit:   {out / 'segments.json'}  (then run again with --resume)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
