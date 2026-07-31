"""
test_speech.py
Unit tests for the speech/whisper module.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speech.whisper import WhisperTranscriber, SUPPORTED_AUDIO_EXTENSIONS


class TestWhisperTranscriber:
    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "audio.odt"
        p.write_bytes(b"fake")
        t = WhisperTranscriber()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            t.transcribe_file(p)

    def test_file_not_found_raises(self, tmp_path):
        t = WhisperTranscriber()
        with pytest.raises(FileNotFoundError):
            t.transcribe_file(tmp_path / "missing.mp3")

    def test_transcribe_file_calls_whisper(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake mp3 data")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "  Patient has cough for 3 days.  ",
            "language": "en",
        }

        t = WhisperTranscriber(model_name="small")
        t._model = mock_model
        result = t.transcribe_file(audio_file)

        assert result == "Patient has cough for 3 days."
        mock_model.transcribe.assert_called_once()

    def test_transcribe_with_language_option(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Paciente tem tosse.", "language": "pt"}

        t = WhisperTranscriber(model_name="small", language="pt")
        t._model = mock_model
        result = t.transcribe_file(audio_file)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "pt"
        assert result == "Paciente tem tosse."

    def test_transcribe_bytes_creates_temp_file(self, tmp_path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello.", "language": "en"}

        t = WhisperTranscriber()
        t._model = mock_model
        result = t.transcribe_bytes(b"fake audio bytes", suffix=".wav")
        assert result == "Hello."

    def test_supported_extensions_set(self):
        assert ".mp3" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".m4a" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".flac" in SUPPORTED_AUDIO_EXTENSIONS

    def test_load_model_raises_on_missing_whisper(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                raise ImportError("No module named 'whisper'")
            return real_import(name, *args, **kwargs)

        t = WhisperTranscriber()
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="openai-whisper"):
                t._load_model()
