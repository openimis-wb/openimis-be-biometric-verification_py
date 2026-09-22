import base64
import logging
import os
import statistics

from .providers.base import VerificationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphQL adapter objects
# schema.py declares Output = VerificationResultType / EmbeddingResultType.
# Graphene resolves those types by reading attributes — plain objects with the
# right attribute names are enough; no need to import the graphene type here.
# ---------------------------------------------------------------------------

class _VerifyFaceResult:
    __slots__ = ("verified", "confidence", "distance", "provider", "error")

    def __init__(self, result: VerificationResult):
        self.verified = result.verified
        self.confidence = result.confidence
        self.distance = result.distance
        self.provider = result.provider
        self.error = result.error


class _EmbeddingResult:
    __slots__ = ("success", "model", "provider", "error")

    def __init__(self, success: bool, model=None, provider=None, error=None):
        self.success = success
        self.model = model
        self.provider = provider
        self.error = error


# ---------------------------------------------------------------------------
# BiometricService
# ---------------------------------------------------------------------------

class BiometricService:
    """
    Facade between the GraphQL mutations and the provider layer.

    All business logic lives here so that schema.py stays thin and
    providers stay stateless (pure image-in / result-out).
    """

    # ------------------------------------------------------------------
    # Public API — called by schema.py mutations
    # ------------------------------------------------------------------

    @staticmethod
    def verify_face(insuree_uuid: str, frame_b64: str, user) -> _VerifyFaceResult:
        """
        Verify an insuree's identity from a webcam frame.

        Fast path (STORE_EMBEDDINGS=True):
            Uses the stored embedding for the reference side.
            Only the probe (webcam frame) goes through model inference.

        Slow path (no stored embedding or STORE_EMBEDDINGS=False):
            Full image-vs-image comparison via provider.verify().
        """
        try:
            from .registry import ProviderRegistry
            from .apps import BiometricVerificationConfig
            from insuree.models import Insuree

            logger.info("━" * 70)
            logger.info("🔐 BiometricService.verify_face() called")
            logger.info(f"   Insuree UUID: {insuree_uuid}")
            logger.info(f"   Frame size: {len(frame_b64)} chars (base64)")
            logger.info("━" * 70)

            provider = ProviderRegistry.get_active_provider()
            logger.info(f"✓ Active provider: {provider.provider_name}")

            probe_bytes = BiometricService._decode_frame(frame_b64)
            logger.info(f"✓ Decoded frame: {len(probe_bytes)} bytes")

            insuree = Insuree.objects.get(uuid=insuree_uuid, validity_to__isnull=True)
            logger.info(f"✓ Found insuree: {insuree.chf_id} ({insuree.other_names} {insuree.last_name})")

            # Fast path — stored embedding present and active
            if BiometricVerificationConfig.store_embeddings:
                logger.info("🚀 Fast path enabled (STORE_EMBEDDINGS=True)")
                try:
                    stored = insuree.biometric_embedding  # OneToOne reverse accessor
                    if stored.is_active:
                        logger.info(f"✓ Found active embedding: {stored.model_name}, computed {stored.computed_at}")
                        logger.info(f"   Embedding dimension: {len(stored.embedding)}")

                        # Check if the embedding was computed with the current configuration
                        current_config = {
                            "model_name": getattr(provider, "model_name", "unknown"),
                            "provider": provider.provider_name,
                            **BiometricVerificationConfig.provider_config,
                        }

                        # Compare stored config with current config
                        config_changed = stored.metadata != current_config

                        if config_changed:
                            logger.warning("⚠ Embedding configuration has changed!")
                            logger.warning(f"   Stored config: {stored.metadata}")
                            logger.warning(f"   Current config: {current_config}")
                            logger.info("🔄 Recalculating embedding with new configuration...")

                            # Recalculate embedding with current config
                            embedding_result = BiometricService.compute_insuree_embedding(
                                insuree_uuid, user=None
                            )

                            if embedding_result.success:
                                logger.info("✓ Embedding recalculated successfully")
                                # Reload the fresh embedding
                                insuree.refresh_from_db()
                                stored = insuree.biometric_embedding
                            else:
                                logger.error(f"✗ Failed to recalculate embedding: {embedding_result.error}")
                                logger.info("→ Falling back to slow path")
                                raise Exception("Embedding recalculation failed")
                        else:
                            logger.info("✓ Embedding config matches current config")

                        result = provider.verify_from_embedding(
                            probe_image=probe_bytes,
                            reference_embedding=stored.embedding,
                        )
                        return _VerifyFaceResult(result)
                    else:
                        logger.warning("⚠ Stored embedding is inactive, falling back to slow path")
                except Exception as e:
                    # No stored embedding yet — fall through to slow path
                    logger.warning(
                        f"⚠ No active embedding for insuree {insuree_uuid}: {e}"
                    )
                    logger.info("→ Falling back to image-vs-image comparison")

            # Slow path — full image-vs-image comparison
            logger.info("🐢 Slow path: Full image-vs-image comparison")
            reference_bytes = BiometricService._fetch_insuree_photo(insuree)
            logger.info(f"✓ Loaded reference photo: {len(reference_bytes)} bytes")

            result = provider.verify(
                probe_image=probe_bytes,
                reference_image=reference_bytes,
            )

            # After successful slow path verification, compute and store embedding for future fast path use
            if result.verified and BiometricVerificationConfig.store_embeddings:
                logger.info("💾 Verification successful - computing embedding for future fast path...")
                try:
                    embedding_result = BiometricService.compute_insuree_embedding(
                        insuree_uuid, user=None
                    )
                    if embedding_result.success:
                        logger.info("✓ Embedding stored - next verification will use fast path")
                    else:
                        logger.warning(f"⚠ Failed to store embedding: {embedding_result.error}")
                except Exception as e:
                    logger.warning(f"⚠ Failed to compute embedding after verification: {e}")

            return _VerifyFaceResult(result)

        except Exception as exc:
            logger.exception("verify_face failed for insuree %s", insuree_uuid)
            logger.error(f"❌ BiometricService.verify_face FAILED: {str(exc)}")
            return _VerifyFaceResult(
                VerificationResult(verified=False, error=str(exc))
            )

    @staticmethod
    def compute_insuree_embedding(insuree_uuid: str, user) -> _EmbeddingResult:
        """
        Pre-compute and persist the face embedding for an insuree.

        Invalidates any existing active embedding before storing the new one
        (soft-delete: sets validity_to on the old row).
        """
        try:
            from .registry import ProviderRegistry
            from .models import BiometricEmbedding
            from insuree.models import Insuree
            from django.utils import timezone

            provider = ProviderRegistry.get_active_provider()
            insuree = Insuree.objects.get(uuid=insuree_uuid, validity_to__isnull=True)
            photo_bytes = BiometricService._fetch_insuree_photo(insuree)

            embedding_vector = provider.get_embedding(photo_bytes)
            model_name = getattr(provider, "model_name", "unknown")

            # Capture the complete provider configuration used to compute this embedding
            # This allows detecting when the config has changed (e.g. detector_backend changed)
            from .apps import BiometricVerificationConfig
            provider_config_snapshot = {
                "model_name": model_name,
                "provider": provider.provider_name,
                **BiometricVerificationConfig.provider_config,  # Includes detector_backend, enforce_detection, etc.
            }

            # Invalidate previous active embedding (soft-delete)
            BiometricEmbedding.objects.filter(
                insuree=insuree,
                validity_to__isnull=True,
            ).update(validity_to=timezone.now())

            # Persist new embedding with configuration snapshot
            BiometricEmbedding.objects.create(
                insuree=insuree,
                embedding=embedding_vector,
                model_name=model_name,
                provider=provider.provider_name,
                metadata=provider_config_snapshot,  # Store complete config
            )

            logger.info(
                "Computed and stored embedding for insuree %s (model=%s, provider=%s)",
                insuree_uuid,
                model_name,
                provider.provider_name,
            )
            logger.info(
                "Stored embedding with config snapshot: %s", provider_config_snapshot
            )
            return _EmbeddingResult(
                success=True,
                model=model_name,
                provider=provider.provider_name,
            )

        except Exception as exc:
            logger.exception(
                "compute_insuree_embedding failed for insuree %s", insuree_uuid
            )
            return _EmbeddingResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_frame(frame_b64: str) -> bytes:
        """
        Decode a base64 image string to raw bytes.
        Strips the optional data URI prefix (data:image/jpeg;base64,…).
        """
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        return base64.b64decode(frame_b64)

    @staticmethod
    def _fetch_insuree_photo(insuree) -> bytes:
        """
        Read the insuree's current reference photo from disk and return raw bytes.

        Uses the same path convention as the insuree module:
            InsureeConfig.insuree_photos_root_path / photo.folder / photo.filename
        """
        from insuree.apps import InsureeConfig

        photo = getattr(insuree, "photo", None)
        if photo is None:
            raise ValueError(
                f"Insuree {insuree.uuid} has no reference photo attached."
            )

        root = InsureeConfig.insuree_photos_root_path
        if not root:
            raise ValueError(
                "InsureeConfig.insuree_photos_root_path is not configured. "
                "Set insuree_photos_root_path in ModuleConfiguration or "
                "the PHOTO_ROOT_PATH environment variable."
            )

        photo_path = os.path.join(root, photo.folder or "", photo.filename or "")
        if not os.path.isfile(photo_path):
            raise FileNotFoundError(
                f"Reference photo not found on disk: {photo_path}"
            )

        with open(photo_path, "rb") as fh:
            return fh.read()

    # ------------------------------------------------------------------
    # Claim Fraud Risk Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_global_risk_score(claim_id):
        """
        Calculate the global fraud risk score for a claim based on all facial audits.

        Risk score ranges from 0.0 (safe) to 1.0 (fraud suspected).

        Business rules:
        - If ANY audit has is_verified=False → risk jumps to 0.8+
        - If variance in similarity scores is high → risk increases
        - If all audits pass with consistent scores → low risk

        Args:
            claim_id: UUID or ID of the Claim

        Returns:
            dict with:
                - risk_score (float): 0.0 to 1.0
                - audit_count (int): number of audits
                - failed_audits (int): number of failed verifications
                - score_variance (float): variance in similarity scores
                - risk_level (str): 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        """
        from .models import ClaimFacialAudit

        # Fetch all facial audits for this claim
        # Note: ClaimFacialAudit is an immutable audit trail - all records are active
        audits = ClaimFacialAudit.objects.filter(
            claim_id=claim_id,
        ).order_by("audit_date")

        audit_count = audits.count()

        if audit_count == 0:
            # No audits yet - no risk assessment possible
            return {
                "risk_score": 0.0,
                "audit_count": 0,
                "failed_audits": 0,
                "score_variance": 0.0,
                "risk_level": "UNKNOWN",
                "message": "No facial audits recorded for this claim",
            }

        # Extract data from audits
        failed_audits = audits.filter(is_verified=False).count()
        similarity_scores = list(audits.values_list("similarity_score", flat=True))

        # Calculate variance in similarity scores
        score_variance = 0.0
        if len(similarity_scores) > 1:
            try:
                score_variance = statistics.variance(similarity_scores)
            except statistics.StatisticsError:
                score_variance = 0.0

        # Initialize base risk score
        risk_score = 0.0

        # Rule 1: If ANY audit failed → HIGH RISK (0.8+)
        if failed_audits > 0:
            failed_ratio = failed_audits / audit_count
            risk_score = 0.8 + (0.2 * failed_ratio)  # 0.8 to 1.0
            risk_score = min(risk_score, 1.0)

        # Rule 2: High variance in similarity scores → MEDIUM to HIGH RISK
        elif score_variance > 0.1:  # Threshold: 0.1 for variance
            # Variance penalty: higher variance = more suspicious
            variance_penalty = min(score_variance * 5, 0.6)  # Cap at 0.6
            risk_score = 0.4 + variance_penalty  # 0.4 to 1.0

        # Rule 3: Low average similarity score → MEDIUM RISK
        elif similarity_scores:
            avg_score = statistics.mean(similarity_scores)
            if avg_score < 0.6:  # Low average confidence
                risk_score = 0.4 + (0.6 - avg_score) * 0.5  # 0.4 to 0.7

        # Rule 4: All audits pass with consistent scores → LOW RISK
        else:
            risk_score = 0.1  # Minimal baseline risk

        # Determine risk level category
        if risk_score >= 0.8:
            risk_level = "CRITICAL"
        elif risk_score >= 0.5:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        logger.info(
            f"Claim {claim_id} risk score: {risk_score:.2f} "
            f"({risk_level}) - {audit_count} audits, {failed_audits} failed"
        )

        return {
            "risk_score": round(risk_score, 3),
            "audit_count": audit_count,
            "failed_audits": failed_audits,
            "score_variance": round(score_variance, 3),
            "risk_level": risk_level,
            "avg_similarity": round(statistics.mean(similarity_scores), 3) if similarity_scores else 0.0,
        }
