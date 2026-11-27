import numpy as np
import pytest

from echo import cli, llm, mix, report, segment, syllables
from echo.models import style
from echo.pipeline import delivery_hints, reference_clip
from echo.translate import budget, fit, shorten, translate
from echo.media import atempo_chain

SR = 16000


def tone(seconds, amp=0.5, f=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- finding lines -----------------------------------------------------------

def test_lines_are_found_and_short_pauses_bridged():
    y = np.concatenate([silence(0.5), tone(1.0), silence(0.2), tone(1.0), silence(1.5), tone(0.8), silence(0.5)])
    lines = segment.find_lines(y, SR, pad=0.0)
    assert len(lines) == 2  # the 0.2 s pause is inside a line, the 1.5 s one is between lines
    assert lines[0].start == pytest.approx(0.5, abs=0.1)
    assert lines[0].end == pytest.approx(2.7, abs=0.1)
    assert lines[1].start == pytest.approx(4.2, abs=0.1)


def test_long_lines_are_cut_at_the_quietest_point():
    y = np.concatenate([tone(8), tone(0.3, amp=0.02), tone(8)])
    lines = segment.find_lines(y, SR, max_len=10.0, top_db=60, pad=0.0)
    assert len(lines) == 2
    assert lines[0].end == pytest.approx(8.15, abs=0.3)


def test_silence_has_no_lines():
    assert segment.find_lines(silence(2), SR) == []
    assert segment.find_lines(np.zeros(0, dtype=np.float32), SR) == []


# --- subtitles ---------------------------------------------------------------

def test_srt_round_trip():
    text = "1\n00:00:01,500 --> 00:00:03,250\n<i>Hello</i>\nthere\n\n2\n00:01:00,000 --> 00:01:02,000\nనమస్తే\n"
    cues = segment.read_srt(text)
    assert cues[0] == {"start": 1.5, "end": 3.25, "text": "Hello there"}
    assert cues[1]["text"] == "నమస్తే"
    again = segment.read_srt(segment.write_srt(cues))
    assert again == cues


def test_stamp_formats_hours():
    assert segment.stamp(3725.042) == "01:02:05,042"


# --- speakers ----------------------------------------------------------------

def test_cluster_groups_similar_voices():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=256), rng.normal(size=256)
    emb = np.stack([a + rng.normal(scale=0.2, size=256) for _ in range(3)]
                   + [b + rng.normal(scale=0.2, size=256) for _ in range(2)])
    emb = emb[[0, 3, 1, 4, 2]]  # interleave the two voices
    assert segment.cluster(emb) == [0, 1, 0, 1, 0]
    assert segment.cluster(emb, n=1) == [0] * 5


def test_reference_clip_takes_a_speakers_longest_lines():
    vox = np.concatenate([tone(2, f=200), tone(5, f=300), tone(1, f=200)])
    segs = [{"speaker": "S1", "start": 0, "end": 2}, {"speaker": "S2", "start": 2, "end": 7},
            {"speaker": "S1", "start": 7, "end": 8}]
    ref = reference_clip(segs, vox, SR, "S1", seconds=10)
    assert len(ref) == pytest.approx(3.3 * SR, rel=0.01)  # 2 s + 1 s + two 0.15 s gaps


def test_delivery_hints_mark_loud_and_soft_lines():
    vox = np.concatenate([tone(1, 0.1), tone(1, 0.1), tone(1, 0.6), tone(1, 0.02)])
    segs = [{"start": i, "end": i + 1, "source": "abcdefgh"} for i in range(4)]
    delivery_hints(segs, vox, SR)
    assert [s["delivery"] for s in segs] == ["", "", "loud", "soft"]


# --- translation -------------------------------------------------------------

class FakeLLM(llm.LLM):
    def __init__(self, replies):
        super().__init__(name="fake", model="scripted")
        self.replies, self.prompts = list(replies), []

    def json(self, system, prompt, schema):
        self.calls += 1
        self.prompts.append(prompt)
        return self.replies.pop(0)


def segs3():
    return [
        {"id": 1, "start": 0.0, "end": 2.0, "source": "a", "speaker": "S1"},
        {"id": 2, "start": 2.5, "end": 4.0, "source": "b", "speaker": "S2"},
        {"id": 3, "start": 4.5, "end": 5.5, "source": "c", "speaker": "S1"},
    ]


def test_budget_and_fit():
    assert budget(2.0, 3.6) == 7
    assert syllables.count("I can't believe you came back") == 7
    assert fit("I can't believe you came back", 7) == 1.0
    assert syllables.count("It cost 20 dollars") == 6


def test_translate_revises_only_lines_that_do_not_fit():
    segs = segs3()
    for s in segs:
        s["budget"] = budget(s["end"] - s["start"], 3.6)  # 7, 5, 4
    fake = FakeLLM([
        {"lines": [
            {"id": 1, "text": "I can't believe you came back", "emotion": "surprised", "intensity": 0.8},
            {"id": 2, "text": "I told you a thousand times already, didn't I", "emotion": "angry", "intensity": 1.4},
            {"id": 3, "text": "Go home now", "emotion": "bogus", "intensity": 0.2},
        ]},
        {"lines": [{"id": 2, "text": "I told you so", "emotion": "angry", "intensity": 0.9},
                   {"id": 1, "text": "should be ignored", "emotion": "calm", "intensity": 0}]},
    ])
    translate(segs, fake, rounds=2)
    assert fake.calls == 2
    assert "Line 2" in fake.prompts[1] and "Line 1 " not in fake.prompts[1]
    assert segs[0]["text"] == "I can't believe you came back"
    assert segs[1]["text"] == "I told you so"
    assert segs[2]["emotion"] == "neutral"  # unknown labels fall back
    assert 0 <= segs[1]["intensity"] <= 1


def test_a_worse_revision_is_not_kept():
    segs = segs3()[:1]
    segs[0]["budget"] = 7
    fake = FakeLLM([
        {"lines": [{"id": 1, "text": "You came", "emotion": "calm", "intensity": 0.3}]},
        {"lines": [{"id": 1, "text": "You came back here after everything that happened between us", "emotion": "calm", "intensity": 0.3}]},
        {"lines": [{"id": 1, "text": "Hi", "emotion": "calm", "intensity": 0.3}]},
    ])
    translate(segs, fake, rounds=2)
    assert segs[0]["text"] == "You came"


def test_shorten_keeps_only_shorter_lines():
    segs = segs3()
    segs[0]["text"] = "I really cannot believe you came back"
    segs[1]["text"] = "No"
    fake = FakeLLM([{"lines": [{"id": 1, "text": "You came back", "emotion": "surprised", "intensity": 0.7},
                               {"id": 2, "text": "No no no no no", "emotion": "angry", "intensity": 1}]}])
    got = shorten(segs, {1: 4, 2: 1}, fake)
    assert got == {1: {"text": "You came back", "emotion": "surprised", "intensity": 0.7}}


# --- timing and mixing ---------------------------------------------------------

def test_slots_allow_a_little_overhang_but_not_into_the_next_line():
    s = mix.slots(segs3(), total=6.0)
    assert s[0] == pytest.approx(2.4)          # 0.4 s of the 0.5 s gap
    assert s[1] == pytest.approx(1.9)          # 0.4 s overhang, still 50 ms clear of line 3
    assert s[2] == pytest.approx(1.4)


def test_plan_fit():
    assert mix.plan_fit(1.0, 2.0).tempo == 1.0
    f = mix.plan_fit(2.4, 2.0)
    assert f.tempo == pytest.approx(1.2) and not f.cut
    f = mix.plan_fit(4.0, 2.0)
    assert f.tempo == mix.MAX_SPEEDUP and f.cut


def test_atempo_chain_stays_in_range():
    assert atempo_chain(1.2) == "atempo=1.2000"
    assert atempo_chain(3.0) == "atempo=2.0000,atempo=1.5000"
    assert atempo_chain(0.4) == "atempo=0.5000,atempo=0.8000"


def test_trim_silence():
    y = np.concatenate([silence(0.5), tone(1.0), silence(0.7)])
    assert len(mix.trim_silence(y, SR)) / SR == pytest.approx(1.04, abs=0.05)


def test_match_level_and_place():
    quiet, loud = tone(1, 0.2), tone(1, 0.5)
    assert mix.db(mix.match_level(quiet, loud)) == pytest.approx(mix.db(loud), abs=0.1)
    assert mix.db(mix.match_level(tone(1, 0.001), loud)) == pytest.approx(mix.db(tone(1, 0.001)) + 12, abs=0.1)
    track = np.zeros(SR * 3, dtype=np.float32)
    mix.place(track, tone(2, 0.5), start=1.0, sr=SR, limit=1.5)
    assert np.abs(track[: SR - 1]).max() == 0
    assert np.abs(track[int(2.6 * SR):]).max() == 0
    assert np.abs(track[int(1.5 * SR):int(2.0 * SR)]).max() > 0.4


def test_combine_never_clips():
    bg = np.full((SR, 2), 0.8, dtype=np.float32)
    out = mix.combine(bg, np.full(SR, 0.8, dtype=np.float32))
    assert np.abs(out).max() <= 0.98 + 1e-6


# --- voice style, report, cli -------------------------------------------------

def test_style_is_calmer_for_calm_lines():
    calm, angry = style("calm", 0.5), style("angry", 1.0, "loud")
    assert calm.exaggeration < angry.exaggeration
    assert angry.cfg_weight < calm.cfg_weight
    assert 0.25 <= style("happy", -3).exaggeration <= 1.0


def test_report_escapes_text():
    segs = [{"id": 1, "start": 0, "end": 1, "speaker": "S1", "text": "<b>hi</b>", "source": "x",
             "emotion": "happy", "intensity": 0.5, "budget": 3, "syllables": 1,
             "fit": {"slot": 1.0, "natural": 0.8, "tempo": 1.0, "cut": False}}]
    page = report.html({"input": "a.mp4"}, segs, "dub.mp4")
    assert "&lt;b&gt;hi&lt;/b&gt;" in page and "<video" in page


def test_cli_requires_consent(tmp_path, capsys):
    f = tmp_path / "a.wav"
    f.write_bytes(b"")
    assert cli.main([str(f)]) == 2
    assert "consent" in capsys.readouterr().err
