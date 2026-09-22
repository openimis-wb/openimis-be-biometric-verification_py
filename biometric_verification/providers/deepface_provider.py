import io
import logging
from typing import Optional

from .base import BaseBiometricProvider, VerificationResult

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


# Models supported by DeepFace and their embedding dimensions.
# Used for documentation and sanity-checking at instantiation time.
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
_DEFAULT_DETECTOR = "retinaface"  # Changed from opencv - more accurate face detection


class DeepFaceProvider(BaseBiometricProvider):
    """
    Local face verification using the DeepFace library.

    DeepFace wraps several state-of-the-art face recognition models
    (ArcFace, Facenet512, GhostFaceNet, Buffalo_L, SFace) and handles
    face detection, alignment, and embedding in a single call.

    Configuration (via BIOMETRIC_VERIFICATION["PROVIDER_CONFIG"]):

        model_name        (str)   Model to use. Default: "ArcFace".
                                  See _SUPPORTED_MODELS for valid values.
        detector_backend  (str)   Face detector. Default: "retinaface".
                                  Options: opencv | retinaface | mtcnn |
                                           ssd | dlib | mediapipe | yolov8
                                  Recommended: retinaface (most accurate)
        enforce_detection (bool)  Raise if no face detected. Default: True.
                                  CRITICAL: Must be True for accurate results.
                                  False allows processing without face detection,
                                  which produces unreliable similarity scores.

    Performance notes:
        - The model is loaded into memory on first call (~3–5 s).
          Subsequent calls reuse the in-memory model (<500 ms on CPU).
        - For CPU-constrained deployments prefer GhostFaceNet or SFace.
        - Tune SIMILARITY_THRESHOLD per deployment — lighting conditions
          and enrollment photo quality vary significantly between countries.
    """

    provider_name = "deepface"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        detector_backend: str = _DEFAULT_DETECTOR,
        enforce_detection: bool = True,  # Changed to True - CRITICAL for accurate results
        **kwargs,
    ):
        if model_name not in _SUPPORTED_MODELS:
            logger.warning(
                "DeepFaceProvider: unknown model '%s'. "
                "Supported: %s. Proceeding anyway.",
                model_name,
                list(_SUPPORTED_MODELS.keys()),
            )
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

        if kwargs:
            logger.debug(
                "DeepFaceProvider: unused config keys: %s", list(kwargs.keys())
            )

    # ------------------------------------------------------------------
    # BaseBiometricProvider implementation
    # ------------------------------------------------------------------

    def verify(
        self,
        probe_image: bytes,
        reference_image: bytes,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """
        Full 1:1 face comparison between two raw images.

        Both images go through face detection + embedding internally.
        Use verify_from_embedding() for faster point-of-service checks
        when a stored embedding is available for the reference side.
        """
        self._assert_available()
        try:
            logger.info(f"🔍 Starting face verification with {self.model_name} model...")
            logger.info(f"   Detector: {self.detector_backend}")
            logger.info(f"   Enforce detection: {self.enforce_detection}")

            # Convert images to numpy arrays
            probe_array = _to_numpy(probe_image)
            reference_array = _to_numpy(reference_image)

            logger.info(f"   Probe image shape: {probe_array.shape if hasattr(probe_array, 'shape') else 'N/A'}")
            logger.info(f"   Reference image shape: {reference_array.shape if hasattr(reference_array, 'shape') else 'N/A'}")

            result = _DeepFace.verify(
                img1_path=probe_array,
                img2_path=reference_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                # distance_metric must be cosine so our threshold is consistent
                # with verify_from_embedding() which uses cosine distance.
                distance_metric="cosine",
            )
            distance = float(result.get("distance", 1.0))
            verified = result.get("verified", False)
            deepface_threshold = result.get("threshold", 0.68)

            if threshold is not None:
                verified = distance <= threshold
                logger.info(f"📏 Custom threshold applied: {threshold}")
            else:
                logger.info(f"📏 Using DeepFace default threshold: {deepface_threshold}")

            confidence = round(max(0.0, min(100.0, (1.0 - distance) * 100)), 2)

            # Detailed logging with interpretation
            logger.info(f"")
            logger.info(f"{'='*70}")
            logger.info(f"🎯 VERIFICATION RESULT")
            logger.info(f"{'='*70}")
            logger.info(f"✓ Model:         {self.model_name}")
            logger.info(f"✓ Distance:      {distance:.6f} (cosine distance)")
            logger.info(f"✓ Threshold:     {threshold or deepface_threshold:.6f}")
            logger.info(f"✓ Confidence:    {confidence:.2f}%")
            logger.info(f"✓ Verified:      {verified}")
            logger.info(f"")

            if verified:
                logger.info(f"✅ MATCH! Distance {distance:.6f} ≤ threshold {threshold or deepface_threshold:.6f}")
            else:
                logger.info(f"❌ NO MATCH. Distance {distance:.6f} > threshold {threshold or deepface_threshold:.6f}")
                gap = distance - (threshold or deepface_threshold)
                logger.warning(f"   Distance exceeded threshold by {gap:.6f}")

                # Provide helpful interpretation
                if distance > 0.8:
                    logger.warning(f"   ⚠ Very high distance (>0.8) - faces likely completely different")
                    logger.warning(f"   💡 Check: Are you comparing the correct insuree?")
                elif distance > 0.6:
                    logger.warning(f"   ⚠ High distance (0.6-0.8) - faces appear different")
                    logger.warning(f"   💡 Check: Lighting, angle, or photo quality issues?")
                elif distance > threshold or deepface_threshold:
                    logger.warning(f"   ⚠ Marginal fail - close to threshold")
                    logger.warning(f"   💡 Try: Better lighting, center face, remove glasses/mask")

            logger.info(f"{'='*70}")
            logger.info(f"")

            return VerificationResult(
                verified=verified,
                confidence=confidence,
                distance=round(distance, 6),
                provider=self.provider_name,
                metadata={
                    "model": self.model_name,
                    "detector": self.detector_backend,
                    "deepface_threshold": result.get("threshold"),
                },
            )
        except Exception as exc:
            logger.exception("DeepFaceProvider.verify() failed")
            logger.error(f"❌ VERIFICATION ERROR: {str(exc)}")

            # Provide helpful error messages
            error_msg = str(exc).lower()
            if "face" in error_msg and "detect" in error_msg:
                # Determine which image has the problem
                if "img1_path" in error_msg:
                    logger.error(f"   💡 No face detected in PROBE image (webcam/live capture)")
                    logger.error(f"   ⚠ The photo being verified does not contain a detectable face")
                    user_error = "No face detected in the captured photo. Please ensure your face is clearly visible, well-lit, and centered in the camera."
                elif "img2_path" in error_msg:
                    logger.error(f"   💡 No face detected in REFERENCE image (stored photo)")
                    logger.error(f"   ⚠ The insuree's enrollment photo does not contain a detectable face")
                    user_error = "No face detected in the stored enrollment photo. The insuree's photo may need to be retaken with better quality."
                else:
                    logger.error(f"   💡 No face detected in image(s)")
                    user_error = "No face detected in image. Please ensure face is clearly visible and centered."

                logger.error(f"   ⚠ Possible causes:")
                logger.error(f"      - Image quality too poor (blurry, dark)")
                logger.error(f"      - Face not visible or too small")
                logger.error(f"      - Wrong image format or corrupted")
                logger.error(f"      - Face obscured (mask, sunglasses, hand)")
            else:
                user_error = str(exc)

            return VerificationResult(
                verified=False,
                provider=self.provider_name,
                error=user_error,
            )

    def get_embedding(self, image: bytes) -> list:
        """
        Compute and return the face embedding vector for one image.

        Returns a list of floats. Dimension depends on the model:
        512 for ArcFace / Facenet512 / Buffalo_L / GhostFaceNet,
        128 for SFace.
        """
        self._assert_available()
        try:
            logger.info(f"📐 Computing embedding with {self.model_name}")
            logger.info(f"   Detector: {self.detector_backend}")
            logger.info(f"   Enforce detection: {self.enforce_detection}")

            img_array = _to_numpy(image)
            logger.info(f"   Image shape: {img_array.shape if hasattr(img_array, 'shape') else 'N/A'}")

            result = _DeepFace.represent(
                img_path=img_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
            )
            # represent() returns a list of dicts (one per detected face).
            # We always use the first (most prominent) face.
            if not result:
                raise ValueError("DeepFace.represent() returned no faces.")

            embedding = result[0]["embedding"]
            logger.info(f"✓ Embedding computed: {len(embedding)} dimensions")
            return embedding
        except Exception as exc:
            logger.exception("DeepFaceProvider.get_embedding() failed")

            # Provide helpful error messages
            error_msg = str(exc).lower()
            if "face" in error_msg and "detect" in error_msg:
                logger.error(f"   💡 No face detected in image")
                logger.error(f"   ⚠ Check image quality and face visibility")
                raise ValueError("No face detected in image. Please ensure face is clearly visible and centered.")
            raise

    def health_check(self) -> bool:
        """Return True if deepface is importable."""
        return _DEEPFACE_AVAILABLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_available(self) -> None:
        if not _DEEPFACE_AVAILABLE:
            raise RuntimeError(
                "deepface is not installed. "
                "Run: pip install 'openimis-be-biometric_verification[deepface]'"
            )


# ---------------------------------------------------------------------------
# Image conversion helper
# ---------------------------------------------------------------------------

def _to_numpy(image_bytes: bytes):
    """
    Convert raw image bytes to a numpy array suitable for DeepFace.

    DeepFace accepts file paths, numpy arrays, or base64 strings.
    Passing a numpy array avoids writing a temp file to disk.
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(img)
    except ImportError:
        # numpy / Pillow not available — fall back to bytes directly.
        # DeepFace can also handle raw bytes in some versions.
        return image_bytes
