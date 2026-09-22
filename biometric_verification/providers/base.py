
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Canonical result returned by every provider for both verify() and
    verify_from_embedding(). Mapped to GraphQL VerificationResultType by
    the service layer.
    """
    verified: bool
    confidence: Optional[float] = None   # 0–100 percentage
    distance: Optional[float] = None     # raw model distance (lower = more similar)
    provider: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


class BaseBiometricProvider(ABC):
    """
    Abstract interface that every biometric provider must implement.

    Built-in providers (deepface_provider.py) and external licensed SDKs
    (providers/external/) all extend this class.  The registry and service
    layer only ever talk to this interface.
    """

    #: Unique lowercase identifier — must match the value used in
    #: BIOMETRIC_VERIFICATION["PROVIDER"] / BiometricVerificationConfig.provider.
    provider_name: str

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every provider
    # ------------------------------------------------------------------

    @abstractmethod
    def verify(
        self,
        probe_image: bytes,
        reference_image: bytes,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """
        1:1 face comparison between two raw image byte-strings.

        Used when no pre-computed embedding is available for the reference side.

        Args:
            probe_image:     Raw bytes of the webcam frame (JPEG/PNG).
            reference_image: Raw bytes of the insuree's reference photo.
            threshold:       Distance threshold override. When None the provider
                             uses its own default or the global
                             SIMILARITY_THRESHOLD config value.

        Returns:
            VerificationResult with verified, confidence, distance filled in.
        """

    @abstractmethod
    def get_embedding(self, image: bytes) -> list:
        """
        Compute and return the face embedding vector for one image.

        Args:
            image: Raw bytes of the image (JPEG/PNG).

        Returns:
            A list of floats (dimension depends on the model:
            512 for ArcFace/Facenet512, 128 for SFace, etc.).
        """

    # ------------------------------------------------------------------
    # Optional override — providers can supply a faster native path
    # ------------------------------------------------------------------

    def verify_from_embedding(
        self,
        probe_image: bytes,
        reference_embedding: list,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """
        Verify a probe image against a pre-computed reference embedding.

        Default implementation: compute the probe's embedding on the fly,
        then measure cosine distance against the stored reference vector.

        Override in a provider subclass if the provider exposes a native
        optimised comparison path (e.g. AWS Rekognition SearchFacesByImage).

        Args:
            probe_image:         Raw bytes of the webcam frame.
            reference_embedding: Previously stored float vector.
            threshold:           Distance threshold override.
        """
        try:
            logger.info("🚀 Fast path: verify_from_embedding (using pre-computed reference)")
            logger.info(f"   Reference embedding dimension: {len(reference_embedding)}")

            probe_embedding = self.get_embedding(probe_image)
            logger.info(f"   Probe embedding dimension: {len(probe_embedding)}")

            if threshold is None:
                from biometric_verification.apps import BiometricVerificationConfig
                # Use threshold directly - for cosine distance, lower values mean more similar
                # ArcFace default threshold is 0.68 (already a distance threshold, not similarity score)
                threshold = BiometricVerificationConfig.similarity_threshold or 0.68
                logger.info(f"📏 Using configured threshold: {threshold:.6f}")
            else:
                logger.info(f"📏 Using custom threshold: {threshold:.6f}")

            distance = _cosine_distance(probe_embedding, reference_embedding)
            verified = distance <= threshold
            confidence = round(max(0.0, min(100.0, (1.0 - distance) * 100)), 2)

            # Detailed logging with interpretation
            logger.info("")
            logger.info("="*70)
            logger.info("🎯 EMBEDDING VERIFICATION RESULT")
            logger.info("="*70)
            logger.info(f"✓ Provider:      {self.provider_name}")
            logger.info(f"✓ Distance:      {distance:.6f} (cosine distance)")
            logger.info(f"✓ Threshold:     {threshold:.6f}")
            logger.info(f"✓ Confidence:    {confidence:.2f}%")
            logger.info(f"✓ Verified:      {verified}")
            logger.info("")

            if verified:
                logger.info(f"✅ MATCH! Distance {distance:.6f} ≤ threshold {threshold:.6f}")
                if confidence > 90:
                    logger.info("   🌟 Excellent match (>90% confidence)")
                elif confidence > 75:
                    logger.info("   👍 Good match (75-90% confidence)")
                else:
                    logger.info("   ✓ Acceptable match")
            else:
                logger.info(
                    f"❌ NO MATCH. Distance {distance:.6f} > threshold {threshold:.6f}"
                )
                gap = distance - threshold
                logger.warning(f"   Distance exceeded threshold by {gap:.6f}")

                # Provide helpful interpretation
                if distance > 0.8:
                    logger.warning("   ⚠ Very high distance (>0.8) - likely different people")
                    logger.warning("   💡 Check: Correct insuree? Correct reference photo?")
                elif distance > 0.6:
                    logger.warning("   ⚠ High distance (0.6-0.8) - faces appear quite different")
                    logger.warning("   💡 Possible causes:")
                    logger.warning("      - Significant aging since enrollment")
                    logger.warning("      - Different lighting conditions")
                    logger.warning("      - Facial hair, glasses, or other changes")
                    logger.warning("      - Poor camera angle or quality")
                elif distance > threshold:
                    logger.warning("   ⚠ Marginal fail - close to passing threshold")
                    logger.warning("   💡 Suggestions to improve:")
                    logger.warning("      - Ensure good lighting (avoid shadows on face)")
                    logger.warning("      - Center face in camera")
                    logger.warning("      - Remove glasses/mask if possible")
                    logger.warning("      - Look directly at camera")
                    logger.warning(
                        f"      - Or adjust threshold to {distance + 0.01:.6f} if appropriate"
                    )

            logger.info("="*70)
            logger.info("")

            return VerificationResult(
                verified=verified,
                confidence=confidence,
                distance=round(distance, 6),
                provider=self.provider_name,
            )
        except Exception as exc:
            logger.exception("verify_from_embedding failed in %s", self.provider_name)
            logger.error(f"❌ EMBEDDING VERIFICATION ERROR: {str(exc)}")
            return VerificationResult(
                verified=False,
                provider=self.provider_name,
                error=str(exc),
            )

    def health_check(self) -> bool:
        """
        Return True if the provider is initialised and reachable.
        Override for providers that call an external service.
        """
        return True


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _cosine_distance(a: list, b: list) -> float:
    """
    Cosine distance between two vectors.
    0.0 = identical direction, 1.0 = orthogonal, 2.0 = opposite.
    Face recognition models typically produce distances in [0, 1].
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
