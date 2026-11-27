"""A review page for the dub: every line with its timing, translation, emotion and fit."""

from __future__ import annotations

from html import escape

CSS = """
:root{--bg:#fbfaf7;--fg:#11100f;--mute:#5e5a53;--line:#e6e2da;--accent:#ff7445;--ok:#2f8f4e;--warn:#b7791f;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#11100f;--fg:#f3f0ea;--mute:#a39e95;--line:#2a2825;--ok:#5cc47f;--warn:#e0a84a;--bad:#ef6b5b}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h1 b{color:var(--accent)}
.meta{color:var(--mute);margin-bottom:16px}
audio,video{width:100%;max-width:720px;margin:8px 0 16px}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:720px}
th,td{border-bottom:1px solid var(--line);padding:8px 6px;text-align:left;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.src{color:var(--mute)}.num{white-space:nowrap;font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.note{color:var(--mute);font-size:13px;margin-top:16px}
"""


def _status(s: dict) -> tuple[str, str]:
    f = s.get("fit") or {}
    if not s.get("text"):
        return "bad", "no line"
    if f.get("cut"):
        return "bad", "trimmed"
    if f.get("tempo", 1.0) > 1.15:
        return "warn", f"sped up ×{f['tempo']:.2f}"
    if f.get("tempo", 1.0) > 1.0:
        return "ok", f"×{f['tempo']:.2f}"
    return "ok", "fits"


def html(meta: dict, segs: list[dict], media: str | None) -> str:
    rows = []
    for s in segs:
        cls, label = _status(s)
        f = s.get("fit") or {}
        rows.append(
            "<tr>"
            f"<td class=num>{s['id']}</td>"
            f"<td class=num>{s['start']:.2f} to {s['end']:.2f}</td>"
            f"<td>{escape(s.get('speaker', ''))}</td>"
            f"<td><div>{escape(s.get('text', ''))}</div><div class=src>{escape(s.get('source', ''))}</div></td>"
            f"<td>{escape(s.get('emotion', ''))} {float(s.get('intensity', 0)):.1f}"
            f"<div class=src>{escape(s.get('delivery', ''))}</div></td>"
            f"<td class=num>{s.get('syllables', 0)} / {s.get('budget', 0)}</td>"
            f"<td class=num>{f.get('natural', 0):.2f}s / {f.get('slot', 0):.2f}s</td>"
            f"<td class='num {cls}'>{escape(label)}</td>"
            "</tr>"
        )
    player = ""
    if media:
        tag = "video" if media.endswith((".mp4", ".mkv", ".mov", ".webm")) else "audio"
        player = f'<{tag} controls src="{escape(media)}"></{tag}>'
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Echo dub report</title><style>{CSS}</style></head><body><main>
<h1><b>Echo</b> dub report</h1>
<div class=meta>{escape(meta.get('input', ''))} · {escape(meta.get('source_language', ''))} → {escape(meta.get('target_language', ''))}
 · {len(segs)} lines · {escape(meta.get('speakers', ''))} · {escape(meta.get('llm', ''))}</div>
{player}
<div class=wrap><table><tr><th>#</th><th>Time</th><th>Voice</th><th>Line (original below)</th><th>Emotion</th>
<th>Syllables</th><th>Voice / slot</th><th>Fit</th></tr>
{''.join(rows)}</table></div>
<p class=note>Machine draft for review. Edit <code>segments.json</code> and run again with <code>--resume</code> to
re-voice your changes. The voice clone was made with the speaker's consent, as confirmed when the dub was run;
the audio carries Resemble AI's Perth watermark marking it as synthetic.</p>
</main></body></html>"""
