import uuid
import logging

from django.db import models
from core.models import HistoryModel

logger = logging.getLogger(__name__)


class BiometricEmbedding(models.Model):
    """
    Stores a pre-computed face embedding for one insuree.

    One row per insuree (OneToOneField). When a new embedding is computed
    for an insuree who already has one, the existing row is replaced
    (upsert via services.py).

    The embedding vector is stored as a JSON array of floats. Its dimension
    depends on the model used (e.g. ArcFace → 512 floats, Facenet512 → 512,
    SFace → 128).

    validity_to is set when the embedding is superseded or invalidated
    (e.g. the insuree's reference photo was updated). NULL means currently
    active. This mirrors the soft-delete convention used across openIMIS.
    """

    id = models.AutoField(primary_key=True)
    uuid = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        db_index=True,
    )

    # Lazy FK — avoids a hard import-time dependency on the insuree module.
    insuree = models.OneToOneField(
        "insuree.Insuree",
        on_delete=models.CASCADE,
        related_name="biometric_embedding",
        db_index=True,
    )

    # The embedding vector produced by the model.
    embedding = models.JSONField(
        help_text="Float vector produced by the face recognition model.",
    )

    # Provenance — which model and provider produced this embedding.
    model_name = models.CharField(
        max_length=64,
        help_text="Model used to compute the embedding (e.g. ArcFace, Facenet512).",
    )
    provider = models.CharField(
        max_length=64,
        help_text="Provider that computed the embedding (e.g. deepface, aws_rekognition).",
    )

    # Configuration snapshot — stores the complete provider config used to compute this embedding.
    # This allows detecting when the config has changed (e.g. detector_backend changed from opencv to retinaface)
    # and automatically invalidating/recalculating outdated embeddings.
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Complete configuration used to compute this embedding (detector_backend, enforce_detection, etc.).",
    )

    # Timestamps
    computed_at = models.DateTimeField(
        auto_now=True,
        help_text="Last time this embedding was (re)computed.",
    )
    validity_from = models.DateTimeField(
        auto_now_add=True,
        help_text="When this embedding became active.",
    )
    validity_to = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="When this embedding was superseded or invalidated. NULL = active.",
    )

    class Meta:
        db_table = "biometric_embedding"
        verbose_name = "Biometric Embedding"
        verbose_name_plural = "Biometric Embeddings"

    def __str__(self):
        return (
            f"BiometricEmbedding(insuree={self.insuree_id}, "
            f"model={self.model_name}, provider={self.provider})"
        )

    @property
    def is_active(self):
        return self.validity_to is None


class ClaimFacialAudit(HistoryModel):
    """
    Records each biometric checkpoint during the patient's journey
    in the FOSA (Facility Of Services Administration) workflow.

    This model allows tracing facial identity verifications at each step
    of care (reception, consultation, pharmacy, etc.) to reduce fraud.

    Security: Does NOT store ANY images. Only scores and metadata.

    Note: HistoryModel provides 'id' and 'uuid' automatically via annotation.
    Do not define them explicitly.
    """

    # Foreign key to Claim (healthcare service claim)
    claim = models.ForeignKey(
        "claim.Claim",
        on_delete=models.CASCADE,
        related_name="facial_audits",
        db_index=True,
        help_text="The claim associated with this biometric checkpoint",
    )

    # Specific medical service (optional - identifies care step)
    service = models.ForeignKey(
        "medical.Service",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="facial_audits",
        help_text="The specific medical service related to this checkpoint",
    )

    # Facial verification data
    similarity_score = models.FloatField(
        help_text="Similarity score between 0.0 (different) and 1.0 (identical)",
    )

    threshold_used = models.FloatField(
        default=0.4,
        help_text="Confidence threshold used for this verification",
    )

    is_verified = models.BooleanField(
        help_text="True if verification succeeded (similarity_score >= threshold_used)",
    )

    # Context metadata
    step_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Name of the care journey step (e.g. 'reception', 'consultation', 'pharmacy')",
    )

    device_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Identifier of the device that performed the verification",
    )

    # Automatic timestamp
    audit_date = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Date and time of the biometric checkpoint",
    )

    # Additional metadata (JSON for flexibility)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional metadata (provider, model_name, etc.)",
    )

    class Meta:
        db_table = "claim_facial_audit"
        verbose_name = "Claim Facial Audit"
        verbose_name_plural = "Claim Facial Audits"
        ordering = ["-audit_date"]
        indexes = [
            models.Index(fields=["claim", "audit_date"]),
            models.Index(fields=["claim", "is_verified"]),
            models.Index(fields=["step_name", "audit_date"]),
        ]

    def __str__(self):
        verification_status = "✓" if self.is_verified else "✗"
        return (
            f"FacialAudit({verification_status} Claim={self.claim_id}, "
            f"Step={self.step_name}, Score={self.similarity_score:.2f})"
        )

