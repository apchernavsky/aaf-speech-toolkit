"""Общие преобразования PCM для детекторов речи (без зависимости от конкретной модели)."""

from __future__ import annotations

import audioop

import numpy as np


def _pcm16_mono_to_float32_16k(pcm: bytes, sr: int) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    if sr != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, sr, 16000, None)
    return np.frombuffer(pcm, dtype=np.int16).copy().astype(np.float32) / 32768.0
