import io
import logging
from typing import Optional

from .base import EmbeddingProvider, Extracted, _cosine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded import — deepface is an optional dependency.
# The module loads cleanly without it; errors surface only at call time.
# ---------------------------------------------------------------------------
try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:
    _DeepFace = None
    _DEEPFACE_AVAILABLE = False


_SUPPORTED_MODELS = {
    "ArcFace": 512,
    "Facenet512": 512,
    "Buffalo_L": 512,
    "GhostFaceNet": 512,
    "SFace": 128,
    "Facenet": 128,
    "VGG-Face": 2622,
    "OpenFace": 128,
    "DeepFace": 4096,
    "DeepID": 160,
    "Dlib": 128,
}

_DEFAULT_MODEL = "ArcFace"
_DEFAULT_DETECTOR = "retinaface"

# Similarity-scale operating point (1.0 - 0.68, the legacy insuree flow's
# distance cutoff). Configurable per deployment via MODALITIES["face"]["threshold"].
_DEFAULT_THRESHOLD = 0.32


class DeepFaceProvider(EmbeddingProvider):
    """
    Face EmbeddingProvider backed by the DeepFace library, on the similarity
    scale (higher = more similar). Modality identity and default threshold
    are fixed here — the caller configures a different threshold via
    BIOMETRIC["MODALITIES"]["face"]["threshold"] rather than any legacy config.
    """

    provider_name = "deepface"
    modality = "face"
    default_threshold = _DEFAULT_THRESHOLD

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        detector_backend: str = _DEFAULT_DETECTOR,
        enforce_detection: bool = True,
        default_threshold: Optional[float] = None,
        **kwargs,
    ):
        if model_name not in _SUPPORTED_MODELS:
            logger.warning(
                "DeepFaceProvider: unknown model '%s'. Supported: %s.",
                model_name, list(_SUPPORTED_MODELS.keys()),
            )
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        if default_threshold is not None:
            self.default_threshold = default_threshold

        if kwargs:
            logger.debug("DeepFaceProvider: unused config keys: %s", list(kwargs.keys()))

    # ------------------------------------------------------------------
    # EmbeddingProvider implementation
    # ------------------------------------------------------------------

    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        """Face has no position — wraps _get_embedding()."""
        return Extracted(vector=self._get_embedding(sample))

    def distance(self, a: list, b: list) -> float:
        return _cosine_distance(a, b)

    def health_check(self) -> bool:
        """Return True if deepface is importable."""
        return _DEEPFACE_AVAILABLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, image: bytes) -> list:
        """Compute the face embedding vector for one image."""
        self._assert_available()
        try:
            img_array = _to_numpy(image)
            result = _DeepFace.represent(
                img_path=img_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
            )
            if not result:
                raise ValueError("DeepFace.represent() returned no faces.")
            return result[0]["embedding"]
        except Exception as exc:
            logger.exception("DeepFaceProvider._get_embedding() failed")
            error_msg = str(exc).lower()
            if "face" in error_msg and "detect" in error_msg:
                raise ValueError(
                    "No face detected in image. Please ensure face is clearly visible and centered."
                )
            raise

    def _assert_available(self) -> None:
        if not _DEEPFACE_AVAILABLE:
            raise RuntimeError(
                "deepface is not installed. "
                "Run: pip install 'openimis-be-biometric_verification[deepface]'"
            )


def _to_numpy(image_bytes: bytes):
    """Convert raw image bytes to a numpy array suitable for DeepFace."""
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(img)
    except ImportError:
        return image_bytes
