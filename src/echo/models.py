"""The speech models: source separation, recognition, speaker embeddings and the voice.

Each loads on first use and is freed after its stage, so the whole pipeline
fits a 6 GB laptop GPU: only one model holds the GPU at a time.

- Separation: Demucs htdemucs (MIT), vocals against everything else.
- Recognition: BuzzASR/telugu for Telugu (Whisper large-v3 fine-tune, MIT),
  openai/whisper-large-v3-turbo (MIT) for other languages.
- Voice: Chatterbox (Resemble AI, MIT), zero-shot voice cloning from a few
  seconds of reference with an emotion-exaggeration control. Its output
  carries Resemble's imperceptible Perth watermark, which we keep.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import EchoError

ASR_MODELS = {"te": "BuzzASR/telugu"}
ASR_DEFAULT = "openai/whisper-large-v3-turbo"
WHISPER_LANG = {"te": "telugu", "hi": "hindi", "ta": "tamil", "kn": "kannada", "ml": "malayalam",
                "en": "english", "bn": "bengali", "mr": "marathi"}


def _torch():
    try:
        import torch
    except ImportError as e:
        raise EchoError('the speech models need PyTorch: pip install -e ".[speech]"') from e
    return torch


def device(name: str = "auto") -> str:
    if name != "auto":
        return name
    return "cuda" if _torch().cuda.is_available() else "cpu"


def free() -> None:
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resample(y: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return y
    import librosa
    return librosa.resample(np.ascontiguousarray(y, dtype=np.float32), orig_sr=sr, target_sr=to)


def separate(mix: np.ndarray, sr: int, dev: str) -> tuple[np.ndarray, np.ndarray]:
    """(vocals, background) for a (samples, channels) float array at ``sr``."""
    torch = _torch()
    from demucs.api import Separator
    sep = Separator(model="htdemucs", device=dev, progress=False)
    ssr = sep.samplerate
    wav = torch.from_numpy(resample(mix.T, sr, ssr)).float()
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    _, stems = sep.separate_tensor(wav, ssr)
    vocals = stems["vocals"].cpu().numpy()
    background = sum(v for k, v in stems.items() if k != "vocals").cpu().numpy()
    del sep, stems, wav
    free()
    return resample(vocals, ssr, sr).T, resample(background, ssr, sr).T


def _generation_file(model_id: str) -> dict:
    import json
    try:
        path = Path(model_id) / "generation_config.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(model_id, "generation_config.json"))
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class Transcriber:
    """Whisper-family speech recognition, called the way the model cards show.

    Fine-tunes such as BuzzASR bake their language into the generation config
    (and replace the tokenizer), so they get no language argument; base Whisper
    models are told the language."""

    def __init__(self, language: str, dev: str, model: str | None = None):
        torch = _torch()
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.model_id = model or ASR_MODELS.get(language, ASR_DEFAULT)
        self.dev = dev
        self.dtype = torch.float16 if dev == "cuda" else torch.float32
        self.proc = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id, dtype=self.dtype).to(dev).eval()
        # Greedy with light repetition control, as BuzzASR recommends; it also
        # stops base Whisper looping on music or breaths.
        self.kwargs: dict = {"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}
        raw = _generation_file(self.model_id)
        if not (raw.get("language") or raw.get("forced_decoder_ids")):
            self.kwargs.update(language=WHISPER_LANG.get(language, language), task="transcribe")

    def __call__(self, y16k: np.ndarray) -> str:
        torch = _torch()
        feats = self.proc(np.ascontiguousarray(y16k, dtype=np.float32), sampling_rate=16000,
                          return_tensors="pt").input_features.to(self.dev, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(feats, **self.kwargs)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def close(self) -> None:
        del self.model
        free()


class SpeakerEncoder:
    """Chatterbox's voice encoder: one 256-d embedding per stretch of speech."""

    def __init__(self, dev: str):
        from chatterbox.models.voice_encoder import VoiceEncoder
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.ve = VoiceEncoder()
        self.ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")))
        self.ve.to(dev).eval()

    def __call__(self, clips16k: list[np.ndarray]) -> np.ndarray:
        # The encoder wants at least ~1.6 s; loop short clips rather than drop them.
        need = int(1.6 * 16000)
        padded = [np.tile(c, int(np.ceil(need / max(1, len(c)))))[:max(need, len(c))] for c in clips16k]
        return self.ve.embeds_from_wavs(padded, sample_rate=16000, trim_top_db=None)

    def close(self) -> None:
        del self.ve
        free()


@dataclass
class Style:
    exaggeration: float
    cfg_weight: float
    temperature: float


CALM = {"neutral", "calm", "tender", "sad"}


def style(emotion: str, intensity: float, delivery: str = "") -> Style:
    """Map an emotion label to Chatterbox controls.

    Chatterbox's guidance: exaggeration 0.5 and cfg 0.5 suit most speech; more
    exaggeration with less cfg gives more dramatic delivery and slower,
    more deliberate pacing is offset by lowering cfg.
    """
    intensity = min(1.0, max(0.0, intensity))
    if "loud" in delivery:
        intensity = min(1.0, intensity + 0.15)
    if emotion in CALM:
        ex = 0.35 + 0.25 * intensity
        cfg = 0.5
    else:
        ex = 0.5 + 0.45 * intensity
        cfg = 0.5 - 0.2 * intensity
    return Style(round(min(1.0, max(0.25, ex)), 3), round(cfg, 3), 0.8)


class Voice:
    """Chatterbox, conditioned once per speaker on that speaker's own reference audio."""

    def __init__(self, dev: str):
        import torch
        from chatterbox.tts import ChatterboxTTS
        self.torch = torch
        self.tts = ChatterboxTTS.from_pretrained(device=dev)
        self.sr = self.tts.sr
        self._conds: dict[str, object] = {}

    def add_speaker(self, name: str, reference_wav: str) -> None:
        self.tts.prepare_conditionals(reference_wav, exaggeration=0.5)
        self._conds[name] = self.tts.conds

    def say(self, text: str, speaker: str, st: Style, seed: int = 0) -> np.ndarray:
        self.tts.conds = self._conds[speaker]
        self.torch.manual_seed(seed)
        wav = self.tts.generate(text, exaggeration=st.exaggeration, cfg_weight=st.cfg_weight,
                                temperature=st.temperature)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)

    def close(self) -> None:
        del self.tts, self._conds
        free()
