"""Deterministic providers for tests — no ML model is ever loaded or downloaded."""

import hashlib
from typing import List, Optional

from .base import EmbeddingProvider, Extracted, MatcherProvider, _cosine_distance


def _hash_to_vector(data: bytes, dim: int) -> List[float]:
    """Same input bytes always produce the same vector — no randomness."""
    digest = hashlib.sha256(data).digest()
    while len(digest) < dim:
        digest += hashlib.sha256(digest).digest()
    return [b / 255.0 for b in digest[:dim]]


class FakeEmbeddingProvider(EmbeddingProvider):
    """extract() hashes the sample to a fixed-length vector; distance is cosine."""

    kind = "embedding"

    def __init__(
        self,
        modality: str = "face",
        provider_name: str = "fake_embedding",
        default_threshold: float = 0.68,
        dim: int = 8,
        **kwargs,
    ):
        self.modality = modality
        self.provider_name = provider_name
        self.default_threshold = default_threshold
        self.dim = dim

    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        return Extracted(vector=_hash_to_vector(sample, self.dim))

    def distance(self, a: List[float], b: List[float]) -> float:
        return _cosine_distance(a, b)


class FakeMatcherProvider(MatcherProvider):
    """match() is exact-equality: 100 when probe == reference, else 0."""

    kind = "template"

    def __init__(
        self,
        modality: str = "fingerprint",
        provider_name: str = "fake_matcher",
        default_threshold: float = 50.0,
        **kwargs,
    ):
        self.modality = modality
        self.provider_name = provider_name
        self.default_threshold = default_threshold

    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        return Extracted(template=sample)

    def match(self, probe: bytes, reference: bytes) -> float:
        return 100.0 if probe == reference else 0.0
