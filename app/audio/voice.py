"""ERA AI — voice / recording (spec Section 9).

Modular audio: a Recorder captures mic audio to a WAV file, and a
pluggable STT engine transcribes it. Backends are optional imports so
the core app runs without a microphone or speech model installed.

Nothing is transmitted anywhere — transcription is fully local.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


# ---------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------
class Recorder:
    """Records mic audio to a local WAV file using sounddevice."""

    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._frames = []
        self._stream = None

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except Exception:
            return False

    def start(self):
        import sounddevice as sd  # optional
        self._frames = []

        def cb(indata, frames, time_info, status):  # noqa: ANN001
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=self.channels, callback=cb
        )
        self._stream.start()

    def stop(self, out_path: Path) -> Path:
        import numpy as np
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.concatenate(self._frames) if self._frames else np.zeros((1, self.channels))
        pcm = (data * 32767).astype("<i2")
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(pcm.tobytes())
        return out_path


# ---------------------------------------------------------------------
# Speech-to-text (pluggable)
# ---------------------------------------------------------------------
class STTEngine(Protocol):
    def transcribe(self, wav_path: Path) -> str: ...


@dataclass
class WhisperSTT:
    """faster-whisper backend (local). Model downloaded once, then offline."""
    model_size: str = "base"
    _model: Optional[object] = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="auto",
                                       compute_type="int8")

    def transcribe(self, wav_path: Path) -> str:
        self._ensure()
        segments, _ = self._model.transcribe(str(wav_path))
        return " ".join(seg.text.strip() for seg in segments).strip()


def default_stt() -> Optional[STTEngine]:
    if WhisperSTT.available():
        return WhisperSTT()
    return None
