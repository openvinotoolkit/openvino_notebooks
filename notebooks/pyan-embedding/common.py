# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""Shared helpers for the pyannote/embedding PyTorch and OpenVINO scripts.

The pyannote/embedding model (XVectorSincNet) maps a single-channel 16 kHz
waveform of shape ``(batch, 1, samples)`` to a fixed ``(batch, 512)`` speaker
embedding. Two embeddings of the same speaker have a small cosine distance.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000  # model is trained on 16 kHz audio
EMBEDDING_DIM = 512  # output dimensionality
MODEL_ID = "pyannote/embedding"


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float32 waveform resampled to ``target_sr``.

    Returns a 1-D ``np.ndarray`` of shape ``(samples,)`` in the range [-1, 1].
    """
    import soundfile as sf

    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns (frames, channels); average channels down to mono.
    waveform = waveform.mean(axis=1)

    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), int(target_sr))
        waveform = resample_poly(waveform, target_sr // g, sr // g).astype("float32")

    return np.ascontiguousarray(waveform, dtype="float32")


def to_model_input(waveform: np.ndarray) -> np.ndarray:
    """Reshape a 1-D waveform to the model input shape ``(1, 1, samples)``."""
    waveform = np.asarray(waveform, dtype="float32").reshape(-1)
    return waveform[None, None, :]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1-D embedding vectors (0 = identical)."""
    a = np.asarray(a, dtype="float64").reshape(-1)
    b = np.asarray(b, dtype="float64").reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)
