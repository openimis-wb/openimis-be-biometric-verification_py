import io
import logging
import math
from typing import Optional

from .base import EmbeddingProvider, Extracted, FaceGeometry, _cosine_distance

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

# Tag of _to_numpy(): Pillow decode to RGB, channels reversed to BGR.
_PREPROCESSING = "pillow_bgr"

# facial_area keys holding (x, y) points. The eye points are present on every
# supported DeepFace version (None when not found); nose and mouth corners only
# on versions that pass them through, for detectors that report them.
_LANDMARK_KEYS = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


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
    preprocessing = _PREPROCESSING

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
        """
        Face has no position. The vector and the face geometry come from the
        same DeepFace result, so the quality gate judges the face that was embedded.
        """
        result = self._represent(sample)
        return Extracted(
            vector=result["embedding"],
            face=face_geometry_from_deepface(result, detector_backend=self.detector_backend),
        )

    def distance(self, a: list, b: list) -> float:
        return _cosine_distance(a, b)

    def health_check(self) -> bool:
        """Return True if deepface is importable."""
        return _DEEPFACE_AVAILABLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _represent(self, image: bytes) -> dict:
        """DeepFace.represent()'s result for the first face found in one image."""
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
            return result[0]
        except Exception as exc:
            logger.exception("DeepFaceProvider._represent() failed")
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


def _finite_number(value) -> Optional[float]:
    """value as a float when it is a finite real number (NumPy scalars included, bool excluded)."""
    if isinstance(value, (bool, str, bytes)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _point(value):
    """An (x, y) pair of finite numbers, else None."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    x, y = _finite_number(value[0]), _finite_number(value[1])
    if x is None or y is None:
        return None
    return x, y


def face_geometry_from_deepface(result, *, detector_backend) -> Optional[FaceGeometry]:
    """
    FaceGeometry from one DeepFace.represent() result dict, in the pixel grid
    of the image passed to DeepFace. DeepFace reports no head pose, so pose is
    None and the quality gate derives roll and yaw_ratio from the landmarks.

    None when there is no face to describe: detector_backend "skip" (the region
    is the whole frame) or the enforce_detection=False fallback (confidence 0
    and no eye found). A box that is not four finite numbers with a positive
    width and height is dropped; a malformed or missing landmark is skipped.
    """
    if detector_backend == "skip" or not isinstance(result, dict):
        return None
    area = result.get("facial_area")
    if not isinstance(area, dict):
        return None

    landmarks = {}
    for key in _LANDMARK_KEYS:
        point = _point(area.get(key))
        if point is not None:
            landmarks[key] = point

    confidence = _finite_number(result.get("face_confidence"))
    if not confidence and "left_eye" not in landmarks and "right_eye" not in landmarks:
        return None

    box = tuple(_finite_number(area.get(key)) for key in ("x", "y", "w", "h"))
    if any(value is None for value in box) or box[2] <= 0 or box[3] <= 0:
        box = None
    return FaceGeometry(box=box, landmarks=landmarks, pose=None)


def _to_numpy(image_bytes: bytes):
    """
    The sample as the height x width x 3 uint8 array DeepFace.represent()
    takes: BGR channel order, the OpenCV convention DeepFace documents for
    numpy input. Pillow decodes without EXIF transpose, as the quality gate
    does; this is the preprocessing tagged _PREPROCESSING.
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.ascontiguousarray(np.array(img)[:, :, ::-1])
    except ImportError:
        return image_bytes
