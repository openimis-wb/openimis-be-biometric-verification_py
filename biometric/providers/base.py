import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Result of services.verify() — populated for both the server and device paths."""
    verified: bool
    confidence: Optional[float] = None
    provider: Optional[str] = None
    modality: Optional[str] = None
    origin: Optional[str] = None
    threshold: Optional[float] = None
    error: Optional[str] = None
    risk_profile: str = ""       # the named profile the threshold was resolved under, "" for the base
    impersonation: Optional[Any] = None  # impersonation.ImpersonationProbe, None when the probe did not run
    impersonation_skip_reason: str = ""  # impersonation.SKIP_REASONS entry when an enabled probe did not run
    template_skip_reason: str = ""  # services.PREPROCESSING_MISMATCH when a template was left out, "" otherwise


@dataclass
class FaceGeometry:
    """
    Where a provider found the face in the sample, read by the quality gate.

    Coordinates are pixels of the sample as Pillow decodes it without EXIF
    transpose: origin top-left, x to the right, y downwards.
    - box: (x, y, width, height) of the detected face.
    - landmarks: (x, y) per name; 'left_eye', 'right_eye', 'nose',
      'mouth_left' and 'mouth_right' are read, other keys are ignored.
    - pose: degrees, any subset of 'yaw', 'pitch', 'roll'; positive roll
      turns the head clockwise in the image.
    """
    box: Optional[Tuple[float, float, float, float]] = None
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pose: Optional[Dict[str, float]] = None


@dataclass
class Extracted:
    """What a provider produces from one raw sample."""
    vector: Optional[List[float]] = None       # EmbeddingProvider
    template: Optional[bytes] = None           # MatcherProvider, vendor format
    template_iso: Optional[bytes] = None       # ISO/IEC 19794 where the SDK gives it
    quality: Optional[float] = None            # 0-100, NFIQ-like where applicable
    metadata: dict = field(default_factory=dict)
    face: Optional[FaceGeometry] = None        # read by the quality gate, never stored


class ModalityProvider(ABC):
    """Common base for embedding and template-matching providers."""

    modality: str            # "face" | "fingerprint" | "voice" | "iris" | "palmvein"
    provider_name: str
    kind: str                # "embedding" | "template"
    default_threshold: float
    # Names how extract() turns a sample into the model's input. Templates
    # recorded under another value are never compared with this provider's
    # output (services.comparable_preprocessing). "" declares none.
    preprocessing: str = ""

    @abstractmethod
    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        """Extract a vector or template from one raw sample."""

    def health_check(self) -> bool:
        return True


class EmbeddingProvider(ModalityProvider):
    """Produces a comparable float vector; distance() drives matching."""

    kind = "embedding"

    @abstractmethod
    def distance(self, a: List[float], b: List[float]) -> float:
        """Lower = closer."""

    def similarity(self, a: List[float], b: List[float]) -> float:
        return 1.0 - self.distance(a, b)


class MatcherProvider(ModalityProvider):
    """Produces an opaque template; match() compares two templates directly."""

    kind = "template"

    @abstractmethod
    def match(self, probe: bytes, reference: bytes) -> float:
        """Higher = more similar."""


def _cosine_distance(a: list, b: list) -> float:
    """
    Cosine distance between two vectors.
    0.0 = identical direction, 1.0 = orthogonal, 2.0 = opposite.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}. "
            "The stored embedding was computed with a different model."
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))
