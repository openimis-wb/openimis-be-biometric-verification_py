import uuid

from django.db import models

# ---------------------------------------------------------------------------
# Multimodal biometric identity + deduplication seam models
# (docs/wb-biometric-dedup-seam.md §3.3, §6.1, §6.3)
#
# subject_model / subject_id address any HistoryModel subject (e.g.
# "individual.Individual") by dotted label + string pk, with no FK and no
# contenttypes dependency — see docs/wb-biometric-dedup-seam.md §1.
# ---------------------------------------------------------------------------

class SubjectRef(models.Model):
    """Abstract (subject_model, subject_id) reference shared by every table."""

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
    """Audit row for one verify() call. Never stores a sample — score/threshold/verdict only."""

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

    A purge only acts on superseded templates when purge_enabled and
    template_retention_days are both set. It also acts on still-active
    templates when purge_active_enabled and active_template_retention_days
    are both set (§6.3) — a separate, off-by-default window, checked after
    the superseded pass.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    template_retention_days = models.IntegerField(null=True, blank=True)
    purge_enabled = models.BooleanField(default=False)
    active_template_retention_days = models.IntegerField(null=True, blank=True)
    purge_active_enabled = models.BooleanField(default=False)

    class Meta:
        db_table = "biometric_retention_policy"
        constraints = [
            models.CheckConstraint(
                check=~models.Q(purge_enabled=True) | models.Q(template_retention_days__isnull=False),
                name="biometric_retention_policy_requires_days",
            ),
            models.CheckConstraint(
                check=~models.Q(purge_active_enabled=True) | models.Q(active_template_retention_days__isnull=False),
                name="biometric_retention_policy_active_requires_days",
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
