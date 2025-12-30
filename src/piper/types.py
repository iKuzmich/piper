from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_MAX_WAV_VALUE = 32767.0


@dataclass
class PhonemeAlignment:
    phoneme: str
    phoneme_ids: Sequence[int]
    num_samples: int


@dataclass
class SkidbladnirWordAlignment:
    word: str
    word_index: int
    phonemes: Sequence[str]
    phoneme_ids: Sequence[int]
    duration: float
    timeline_offset: float


@dataclass
class SkidbladnirSentenceAlignment:
    words: Sequence[SkidbladnirWordAlignment]
    sentence_index: int


@dataclass
class SkidbladnirTextAlignment:
    sentences: Sequence[SkidbladnirSentenceAlignment]
    original_text: str

@dataclass
class AudioChunk:
    """Chunk of raw audio."""

    sample_rate: int
    """Rate of chunk samples in Hertz."""

    sample_width: int
    """Width of chunk samples in bytes."""

    sample_channels: int
    """Number of channels in chunk samples."""

    audio_float_array: np.ndarray
    """Audio data as float numpy array in [-1, 1]."""

    phonemes: list[str]
    """Phonemes that produced this audio chunk."""

    phoneme_ids: list[int]
    """Phoneme ids that produced this audio chunk."""

    phoneme_id_samples: Optional[np.ndarray] = None
    """Number of audio samples for each phoneme id (alignments).

    Only available for supported voice models.
    """

    phoneme_alignments: Optional[list[PhonemeAlignment]] = None
    """Alignments between phonemes and audio samples."""

    skidbladnir_alignments: Optional[SkidbladnirTextAlignment] = None
    """Alignments for Skidbladnir."""

    # ---

    _audio_int16_array: Optional[np.ndarray] = None
    _audio_int16_bytes: Optional[bytes] = None
    _phoneme_alignments: Optional[list[PhonemeAlignment]] = None

    @property
    def audio_int16_array(self) -> np.ndarray:
        """
        Get audio as an int16 numpy array.

        :return: Audio data as int16 numpy array.
        """
        if self._audio_int16_array is None:
            self._audio_int16_array = np.clip(
                self.audio_float_array * _MAX_WAV_VALUE, -_MAX_WAV_VALUE, _MAX_WAV_VALUE
            ).astype(np.int16)

        return self._audio_int16_array

    @property
    def audio_int16_bytes(self) -> bytes:
        """
        Get audio as 16-bit PCM bytes.

        :return: Audio data as signed 16-bit sample bytes.
        """
        return self.audio_int16_array.tobytes()
