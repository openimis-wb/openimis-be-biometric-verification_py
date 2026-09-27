"""Deterministic providers for tests — no ML model is ever loaded or downloaded."""

import hashlib
from typing import List, Optional

from .base import EmbeddingProvider, Extracted, FaceGeometry, MatcherProvider, _cosine_distance


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
        face_geometry: Optional[FaceGeometry] = None,
        reported_quality: Optional[float] = None,
        **kwargs,
    ):
        self.modality = modality
        self.provider_name = provider_name
        self.default_threshold = default_threshold
        self.dim = dim
        self.face_geometry = face_geometry
        self.reported_quality = reported_quality

    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        return Extracted(
            vector=_hash_to_vector(sample, self.dim),
            quality=self.reported_quality,
            face=self.face_geometry,
        )

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
        reported_quality: Optional[float] = None,
        **kwargs,
    ):
        self.modality = modality
        self.provider_name = provider_name
        self.default_threshold = default_threshold
        self.reported_quality = reported_quality

    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        return Extracted(template=sample, quality=self.reported_quality)

    def match(self, probe: bytes, reference: bytes) -> float:
        return 100.0 if probe == reference else 0.0
