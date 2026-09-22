import uuid
import logging

from django.db import models
from core.models import HistoryModel

from .legacy import HAS_INSUREE, HAS_CLAIM

logger = logging.getLogger(__name__)


if HAS_INSUREE:
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


if HAS_CLAIM:
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


# ---------------------------------------------------------------------------
# Multimodal biometric identity + deduplication seam models (§3.3)
#
# subject_model / subject_id address any HistoryModel subject (e.g.
# "individual.Individual") by dotted label + string pk, with no FK and no
# contenttypes dependency — see docs/wb-biometric-dedup-seam.md §1.
# ---------------------------------------------------------------------------

class SubjectRef(models.Model):
    """Abstract (subject_model, subject_id) reference shared by every new table."""

    subject_model = models.CharField(max_length=64)
    subject_id = models.CharField(max_length=64, db_index=True)

    class Meta:
        abstract = True


class BiometricTemplate(SubjectRef):
    """
    One enrolled biometric sample for one subject/modality/position.

    Superseded rather than updated: enrolling again sets validity_to on the
    previous active row and inserts a new one. At most one active row per
    (subject, modality, position, provider, model_name).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    modality = models.CharField(max_length=16)
    position = models.CharField(max_length=16, blank=True, default="")
    kind = models.CharField(max_length=16, help_text='"embedding" or "template"')

    # Plaintext when TEMPLATE_KEY is unset, Fernet ciphertext otherwise.
    # Never read directly — use services.templates_of() to decrypt with audit.
    vector = models.JSONField(null=True, blank=True)
    template = models.BinaryField(null=True, blank=True)
    template_iso = models.BinaryField(null=True, blank=True)
    encrypted = models.BooleanField(default=False)

    quality = models.FloatField(null=True, blank=True)
    provider = models.CharField(max_length=64)
    model_name = models.CharField(max_length=64)
    metadata = models.JSONField(default=dict, blank=True)

    validity_from = models.DateTimeField(auto_now_add=True)
    validity_to = models.DateTimeField(null=True, blank=True, db_index=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "biometric_template"
        constraints = [
            models.UniqueConstraint(
                fields=["subject_model", "subject_id", "modality", "position", "provider", "model_name"],
                condition=models.Q(validity_to__isnull=True),
                name="biometric_template_unique_active",
            ),
        ]
        indexes = [
            models.Index(
                fields=["modality", "provider", "model_name", "validity_to"],
                name="biometric_template_gallery_idx",
            ),
        ]

    @property
    def is_active(self):
        return self.validity_to is None

    def __str__(self):
        return f"BiometricTemplate({self.subject_model}:{self.subject_id}, {self.modality})"


class BiometricVerification(SubjectRef):
    """
    Audit row for one verify() call. Never stores a sample — score/threshold/
    verdict only. The legacy insuree flow keeps using ClaimFacialAudit.
    """

    ORIGIN_SERVER = "server"
    ORIGIN_DEVICE = "device"
    ORIGIN_CHOICES = [(ORIGIN_SERVER, "server"), (ORIGIN_DEVICE, "device")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    modality = models.CharField(max_length=16)
    score = models.FloatField(null=True, blank=True)
    threshold = models.FloatField()
    verified = models.BooleanField()
    origin = models.CharField(max_length=8, choices=ORIGIN_CHOICES, default=ORIGIN_SERVER)
    fallback = models.BooleanField(default=False)
    context = models.JSONField(default=dict, blank=True)
    device_id = models.CharField(max_length=255, blank=True, default="")
    actor = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "biometric_verification"
        indexes = [
            models.Index(fields=["subject_model", "subject_id", "modality"]),
        ]

    def __str__(self):
        return f"BiometricVerification({self.subject_model}:{self.subject_id}, {self.modality}, verified={self.verified})"


class BiometricConsent(SubjectRef):
    """Consent record gating enrol() when REQUIRE_CONSENT is set."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    modality = models.CharField(max_length=16)
    granted = models.BooleanField()
    recorded_by = models.CharField(max_length=64)
    recorded_at = models.DateTimeField(auto_now_add=True)
    note = models.TextField(blank=True, default="")

    class Meta:
        db_table = "biometric_consent"
        indexes = [
            models.Index(fields=["subject_model", "subject_id", "modality"]),
        ]

    def __str__(self):
        return f"BiometricConsent({self.subject_model}:{self.subject_id}, {self.modality}, granted={self.granted})"


class BiometricRetentionPolicy(models.Model):
    """
    Retention policy — one meaningful row, read via services.purge().
    A purge only acts when both fields are set (purge_enabled requires a
    retention window); enforced here and re-checked in the service.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    template_retention_days = models.IntegerField(null=True, blank=True)
    purge_enabled = models.BooleanField(default=False)

    class Meta:
        db_table = "biometric_retention_policy"
        constraints = [
            models.CheckConstraint(
                check=~models.Q(purge_enabled=True) | models.Q(template_retention_days__isnull=False),
                name="biometric_retention_policy_requires_days",
            ),
        ]

    def __str__(self):
        return f"BiometricRetentionPolicy(enabled={self.purge_enabled}, days={self.template_retention_days})"


class BiometricErasure(models.Model):
    """Tombstone left behind when a subject's templates are purged/erased."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Bare strings, no FK — the subject may already be gone.
    subject_model = models.CharField(max_length=64)
    subject_id = models.CharField(max_length=64, db_index=True)

    modalities = models.JSONField(default=list)
    erased = models.JSONField(default=dict, help_text="Counts of rows erased per modality.")
    reason = models.CharField(max_length=32)
    erased_by = models.CharField(max_length=64)
    erased_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "biometric_erasure"

    def __str__(self):
        return f"BiometricErasure({self.subject_model}:{self.subject_id}, reason={self.reason})"


class BiometricAccessLog(SubjectRef):
    """Written every time services.templates_of() decrypts a plaintext template."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    actor = models.CharField(max_length=64)
    purpose = models.CharField(max_length=32)
    template_ids = models.JSONField(default=list)
    at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "biometric_access_log"

    def __str__(self):
        return f"BiometricAccessLog({self.subject_model}:{self.subject_id}, purpose={self.purpose})"
