"""
whisper.py
Offline speech-to-text transcription using OpenAI Whisper (local model).
Audio files are never sent to any external service.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mkv"
}


class WhisperTranscriber:
    """
    Wraps the openai-whisper library for fully local transcription.

    Parameters
    ----------
    model_name:
        Whisper model size: "tiny", "base", "small", "medium", "large".
        Larger models are more accurate but slower.  Default: "small".
    language:
        ISO-639-1 code (e.g. "en", "de", "pt").  None = auto-detect.
    device:
        "cpu" or "cuda".  None = auto-select based on availability.
    """

    def __init__(
        self,
        model_name: str = "small",
        language: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self._model = None  # loaded lazily on first use

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: str | Path) -> str:
        """
        Transcribe an audio file and return the plain-text transcript.

        Parameters
        ----------
        audio_path:
            Path to an audio file (mp3, wav, m4a, flac, …).

        Returns
        -------
        str
            Transcribed text.

        Raises
        ------
        FileNotFoundError
            If the audio file does not exist.
        ValueError
            If the file extension is not supported.
        """
        audio_path = Path(audio_path).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ext = audio_path.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )

        self._load_model()
        logger.info("Transcribing: %s (model=%s)", audio_path.name, self.model_name)

        options: dict = {"fp16": False}
        if self.language:
            options["language"] = self.language

        result = self._model.transcribe(str(audio_path), **options)
        transcript: str = result.get("text", "").strip()

        logger.info(
            "Transcription complete: %d chars, detected_lang=%s",
            len(transcript),
            result.get("language", "unknown"),
        )
        return transcript

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """
        Transcribe raw audio bytes by writing them to a temp file first.

        Parameters
        ----------
        audio_bytes:
            Raw audio data.
        suffix:
            File extension hint (e.g. ".mp3").

        Returns
        -------
        str
            Transcribed text.
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore  # openai-whisper package
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. "
                "Install it with: pip install openai-whisper"
            ) from exc

        import torch  # type: ignore

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading Whisper model '%s' on %s…", self.model_name, device)
        self._model = whisper.load_model(self.model_name, device=device)
        logger.info("Whisper model ready.")
