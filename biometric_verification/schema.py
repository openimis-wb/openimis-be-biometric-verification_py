import base64

import graphene
from django.core.exceptions import PermissionDenied
from django.utils.translation import gettext as _

from .apps import BiometricVerificationConfig


def _decode_sample(sample):
    """Base64 sample, optionally with a data-URI prefix — see services._decode_frame."""
    if sample is None:
        return None
    if "," in sample:
        sample = sample.split(",", 1)[1]
    return base64.b64decode(sample)


def _require_perms(user, perms):
    if user.is_anonymous:
        raise PermissionDenied(_("unauthorized"))
    if perms and not user.has_perms(perms):
        raise PermissionDenied(_("unauthorized"))


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class VerificationResultType(graphene.ObjectType):
    """Returned by verifyFace — result of a 1:1 face comparison."""
    verified = graphene.Boolean(required=True)
    confidence = graphene.Float(
        description="Match confidence as a percentage (0–100)."
    )
    distance = graphene.Float(
        description="Raw embedding distance — lower means more similar."
    )
    provider = graphene.String(
        description="Name of the biometric provider that performed the check."
    )
    error = graphene.String(
        description="Error message if verification could not be completed."
    )


class EmbeddingResultType(graphene.ObjectType):
    """Returned by computeInsureeEmbedding — result of pre-computing a face embedding."""
    success = graphene.Boolean(required=True)
    model = graphene.String(
        description="Model name used to compute the embedding (e.g. ArcFace)."
    )
    provider = graphene.String(
        description="Name of the biometric provider that computed the embedding."
    )
    error = graphene.String(
        description="Error message if the embedding could not be computed."
    )


class ClaimRiskAssessmentType(graphene.ObjectType):
    """Fraud risk assessment for a claim based on facial audits."""
    risk_score = graphene.Float(
        required=True,
        description="Fraud risk score from 0.0 (safe) to 1.0 (suspected fraud)."
    )
    risk_level = graphene.String(
        required=True,
        description="Risk category: UNKNOWN, LOW, MEDIUM, HIGH, CRITICAL."
    )
    audit_count = graphene.Int(
        required=True,
        description="Total number of facial audits for this claim."
    )
    failed_audits = graphene.Int(
        required=True,
        description="Number of failed verifications."
    )
    score_variance = graphene.Float(
        required=True,
        description="Variance in similarity scores across audits."
    )
    avg_similarity = graphene.Float(
        required=True,
        description="Average similarity score across all audits."
    )
    message = graphene.String(
        description="Additional information message."
    )


class ClaimFacialAuditType(graphene.ObjectType):
    """Facial audit checkpoint record."""
    uuid = graphene.String(required=True)
    claim_id = graphene.String(required=True)
    service_id = graphene.String()
    similarity_score = graphene.Float(required=True)
    threshold_used = graphene.Float(required=True)
    is_verified = graphene.Boolean(required=True)
    step_name = graphene.String(required=True)
    device_id = graphene.String()
    audit_date = graphene.DateTime(required=True)
    metadata = graphene.JSONString()


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

class VerifyFaceMutation(graphene.Mutation):
    """
    Compare a live webcam frame (base64 JPEG) against the reference photo
    stored for the given insuree.

    When STORE_EMBEDDINGS is True the reference side uses the pre-computed
    embedding — only the probe (webcam frame) goes through model inference,
    making point-of-service checks significantly faster.

    Returns the verification result immediately (synchronous).
    """

    class Arguments:
        uuid = graphene.String(
            required=True,
            description="UUID of the insuree whose identity is being verified.",
        )
        frame = graphene.String(
            required=True,
            description="Base64-encoded JPEG frame captured from the webcam "
                        "(data:image/jpeg;base64,… prefix is accepted).",
        )

    # Output fields - directly return VerificationResultType fields
    verified = graphene.Boolean(required=True)
    confidence = graphene.Float(description="Match confidence as a percentage (0–100).")
    distance = graphene.Float(description="Raw embedding distance — lower means more similar.")
    provider = graphene.String(description="Name of the biometric provider that performed the check.")
    error = graphene.String(description="Error message if verification could not be completed.")

    @classmethod
    def mutate(cls, root, info, uuid, frame):
        user = info.context.user

        # verifyFace is whitelisted in JWT_ALLOW_ANY_CLASSES so anonymous
        # callers (public kiosk page) are allowed through.
        # Authenticated users are still subject to permission checks if
        # gql_mutation_verify_face_perms is configured.
        # See SECURITY.md for the risk assessment of this design choice.
        if not user.is_anonymous:
            if BiometricVerificationConfig.gql_mutation_verify_face_perms:
                if not user.has_perms(BiometricVerificationConfig.gql_mutation_verify_face_perms):
                    raise PermissionDenied(_("unauthorized"))

        from .services import BiometricService  # noqa: PLC0415

        result = BiometricService.verify_face(
            insuree_uuid=uuid,
            frame_b64=frame,
            user=user,
        )

        # Return result immediately
        return cls(
            verified=result.verified,
            confidence=result.confidence,
            distance=result.distance,
            provider=result.provider,
            error=result.error,
        )


class ComputeInsureeEmbeddingMutation(graphene.Mutation):
    """
    Pre-compute and store the face embedding for an insuree's reference photo.
    Call this once at enrollment so that point-of-service verifyFace calls
    only need to run inference on the probe frame.
    """

    class Arguments:
        insuree_uuid = graphene.String(
            required=True,
            description="UUID of the insuree whose embedding should be computed.",
        )

    Output = EmbeddingResultType

    @classmethod
    def mutate(cls, root, info, insuree_uuid):
        user = info.context.user
        if user.is_anonymous:
            raise PermissionDenied(_("unauthorized"))
        if BiometricVerificationConfig.gql_mutation_compute_embedding_perms:
            if not user.has_perms(
                BiometricVerificationConfig.gql_mutation_compute_embedding_perms
            ):
                raise PermissionDenied(_("unauthorized"))

        from .services import BiometricService  # noqa: PLC0415
        return BiometricService.compute_insuree_embedding(
            insuree_uuid=insuree_uuid,
            user=user,
        )


class CreateClaimFacialAuditMutation(graphene.Mutation):
    """
    Create a facial audit checkpoint for a claim.

    Called by mobile apps or FOSA kiosks after performing a facial verification
    at any step of the care journey (reception, consultation, pharmacy, etc.).

    The signal will automatically recalculate the claim's fraud risk score.
    """

    class Arguments:
        claim_uuid = graphene.String(
            required=True,
            description="UUID of the claim being audited.",
        )
        service_uuid = graphene.String(
            required=False,
            description="UUID of the medical service (optional).",
        )
        similarity_score = graphene.Float(
            required=True,
            description="Similarity score from 0.0 to 1.0.",
        )
        threshold_used = graphene.Float(
            required=True,
            description="Confidence threshold used (default 0.4).",
        )
        is_verified = graphene.Boolean(
            required=True,
            description="Whether the verification succeeded.",
        )
        step_name = graphene.String(
            required=True,
            description="Care journey step (e.g. 'reception', 'consultation').",
        )
        device_id = graphene.String(
            required=False,
            description="Device identifier.",
        )
        metadata = graphene.JSONString(
            required=False,
            description="Additional metadata (provider, model, etc.).",
        )

    Output = ClaimFacialAuditType

    @classmethod
    def mutate(cls, root, info, claim_uuid, similarity_score, threshold_used,
               is_verified, step_name, service_uuid=None, device_id=None, metadata=None):
        user = info.context.user

        # Require authentication for creating audits
        if user.is_anonymous:
            raise PermissionDenied(_("unauthorized"))

        # Validate inputs
        if not (0.0 <= similarity_score <= 1.0):
            raise ValueError("similarity_score must be between 0.0 and 1.0")
        if not (0.0 <= threshold_used <= 1.0):
            raise ValueError("threshold_used must be between 0.0 and 1.0")

        # Consistency check
        expected_verified = similarity_score >= threshold_used
        if is_verified != expected_verified:
            raise ValueError(
                f"is_verified={is_verified} inconsistent with "
                f"similarity_score={similarity_score} >= threshold={threshold_used}"
            )

        from .models import ClaimFacialAudit
        from claim.models import Claim

        # Fetch claim
        try:
            claim = Claim.objects.get(uuid=claim_uuid, validity_to__isnull=True)
        except Claim.DoesNotExist:
            raise ValueError(f"Claim with UUID {claim_uuid} not found")

        # Fetch service if provided
        ## To do : Idea was to get last service entry in the claim.
        # But not realistic as the claim is not save before the Face recognition processing
        # Final process should be refined if we want to track the verification on
        # each services / department of the HF

        service = None
        if service_uuid:
            from medical.models import Service
            try:
                service = Service.objects.get(uuid=service_uuid, validity_to__isnull=True)
            except Service.DoesNotExist:
                raise ValueError(f"Service with UUID {service_uuid} not found")

        # Create audit record
        audit = ClaimFacialAudit.objects.create(
            claim=claim,
            service=service,
            similarity_score=similarity_score,
            threshold_used=threshold_used,
            is_verified=is_verified,
            step_name=step_name,
            device_id=device_id or "",
            metadata=metadata or {},
            user_created=user,
            user_updated=user,
        )

        return audit


# ---------------------------------------------------------------------------
# Multimodal identity + deduplication seam (docs/wb-biometric-dedup-seam.md §3.6)
# ---------------------------------------------------------------------------

class BiometricTemplateType(graphene.ObjectType):
    """Metadata only — never exposes vector/template plaintext or ciphertext."""
    id = graphene.String(required=True)
    subject_model = graphene.String(required=True)
    subject_id = graphene.String(required=True)
    modality = graphene.String(required=True)
    position = graphene.String()
    kind = graphene.String(required=True)
    quality = graphene.Float()
    provider = graphene.String(required=True)
    model_name = graphene.String(required=True)
    encrypted = graphene.Boolean(required=True)
    validity_from = graphene.DateTime()
    validity_to = graphene.DateTime()


class BiometricVerificationRecordType(graphene.ObjectType):
    """One BiometricVerification audit row."""
    id = graphene.String(required=True)
    subject_model = graphene.String(required=True)
    subject_id = graphene.String(required=True)
    modality = graphene.String(required=True)
    score = graphene.Float()
    threshold = graphene.Float(required=True)
    verified = graphene.Boolean(required=True)
    origin = graphene.String(required=True)
    fallback = graphene.Boolean(required=True)
    device_id = graphene.String()
    actor = graphene.String(required=True)
    created_at = graphene.DateTime(required=True)


class BiometricMatchType(graphene.ObjectType):
    """One identify() ranking result."""
    subject_model = graphene.String(required=True)
    subject_id = graphene.String(required=True)
    template_id = graphene.String(required=True)
    score = graphene.Float(required=True)


class BiometricVerifyResultType(graphene.ObjectType):
    """verifyBiometric mutation result."""
    verified = graphene.Boolean(required=True)
    confidence = graphene.Float()
    provider = graphene.String()
    modality = graphene.String()
    origin = graphene.String()
    threshold = graphene.Float()
    error = graphene.String()


class EnrolBiometricMutation(graphene.Mutation):
    """Extract (or accept a device template for) one sample and store it as the subject's active template."""

    class Arguments:
        subject_model = graphene.String(required=True)
        subject_id = graphene.String(required=True)
        modality = graphene.String(required=True)
        sample = graphene.String(required=True, description="Base64-encoded sample.")
        position = graphene.String(required=False)
        metadata = graphene.JSONString(required=False)

    Output = BiometricTemplateType

    @classmethod
    def mutate(cls, root, info, subject_model, subject_id, modality, sample, position=None, metadata=None):
        user = info.context.user
        _require_perms(user, BiometricVerificationConfig.gql_biometric_enrol_perms)

        from .services import enrol

        template = enrol(
            subject_model, subject_id, modality, _decode_sample(sample),
            position=position, actor=user.username, metadata=metadata,
        )
        return BiometricTemplateType(
            id=str(template.id), subject_model=template.subject_model, subject_id=template.subject_id,
            modality=template.modality, position=template.position, kind=template.kind,
            quality=template.quality, provider=template.provider, model_name=template.model_name,
            encrypted=template.encrypted, validity_from=template.validity_from, validity_to=template.validity_to,
        )


class VerifyBiometricMutation(graphene.Mutation):
    """Server path (sample given) or device path (deviceScore given)."""

    class Arguments:
        subject_model = graphene.String(required=True)
        subject_id = graphene.String(required=True)
        modality = graphene.String(required=True)
        sample = graphene.String(required=False, description="Base64-encoded sample (server path).")
        position = graphene.String(required=False)
        device_score = graphene.Float(required=False)
        fallback = graphene.Boolean(required=False)
        device_id = graphene.String(required=False)
        context = graphene.JSONString(required=False)

    Output = BiometricVerifyResultType

    @classmethod
    def mutate(cls, root, info, subject_model, subject_id, modality, sample=None, position=None,
               device_score=None, fallback=False, device_id="", context=None):
        user = info.context.user
        _require_perms(user, BiometricVerificationConfig.gql_biometric_verify_perms)

        from .services import verify

        result = verify(
            subject_model, subject_id, modality, sample=_decode_sample(sample), position=position,
            device_score=device_score, fallback=bool(fallback), context=context,
            device_id=device_id or "", actor=user.username,
        )
        return BiometricVerifyResultType(
            verified=result.verified, confidence=result.confidence, provider=result.provider,
            modality=result.modality, origin=result.origin, threshold=result.threshold, error=result.error,
        )


class RecordBiometricConsentMutation(graphene.Mutation):
    """Records a consent decision gating enrol() when REQUIRE_CONSENT is set."""

    class Arguments:
        subject_model = graphene.String(required=True)
        subject_id = graphene.String(required=True)
        modality = graphene.String(required=True)
        granted = graphene.Boolean(required=True)
        note = graphene.String(required=False)

    ok = graphene.Boolean(required=True)
    id = graphene.String()

    @classmethod
    def mutate(cls, root, info, subject_model, subject_id, modality, granted, note=None):
        user = info.context.user
        _require_perms(user, BiometricVerificationConfig.gql_biometric_enrol_perms)

        from .models import BiometricConsent

        consent = BiometricConsent.objects.create(
            subject_model=subject_model, subject_id=str(subject_id), modality=modality,
            granted=granted, recorded_by=user.username, note=note or "",
        )
        return cls(ok=True, id=str(consent.id))


# ---------------------------------------------------------------------------
# Root types — registered by the openIMIS schema aggregator
# ---------------------------------------------------------------------------

class Query(graphene.ObjectType):
    claim_risk_assessment = graphene.Field(
        ClaimRiskAssessmentType,
        claim_uuid=graphene.String(required=True),
        description="Get fraud risk assessment for a claim based on facial audits.",
    )

    claim_facial_audits = graphene.List(
        ClaimFacialAuditType,
        claim_uuid=graphene.String(required=True),
        description="Get all facial audits for a specific claim.",
    )

    @staticmethod
    def resolve_claim_risk_assessment(root, info, claim_uuid):
        user = info.context.user
        if user.is_anonymous:
            raise PermissionDenied(_("unauthorized"))

        from .services import BiometricService
        from claim.models import Claim

        # Fetch claim to validate it exists
        try:
            claim = Claim.objects.get(uuid=claim_uuid, validity_to__isnull=True)
        except Claim.DoesNotExist:
            raise ValueError(f"Claim with UUID {claim_uuid} not found")

        # Calculate risk score
        risk_data = BiometricService.calculate_global_risk_score(claim.id)

        return ClaimRiskAssessmentType(**risk_data)

    @staticmethod
    def resolve_claim_facial_audits(root, info, claim_uuid):
        user = info.context.user
        if user.is_anonymous:
            raise PermissionDenied(_("unauthorized"))

        from .models import ClaimFacialAudit
        from claim.models import Claim

        # Fetch claim to validate it exists
        try:
            claim = Claim.objects.get(uuid=claim_uuid, validity_to__isnull=True)
        except Claim.DoesNotExist:
            raise ValueError(f"Claim with UUID {claim_uuid} not found")

        # Fetch all facial audits for this claim, ordered by audit date
        # Note: ClaimFacialAudit uses HistoryModel which has is_deleted instead of validity_to
        audits = ClaimFacialAudit.objects.filter(
            claim=claim,
            is_deleted=False
        ).order_by('-audit_date')

        # Convert to ClaimFacialAuditType objects
        return [
            ClaimFacialAuditType(
                uuid=str(audit.uuid),
                claim_id=str(audit.claim.uuid),
                service_id=str(audit.service.uuid) if audit.service else None,
                similarity_score=audit.similarity_score,
                threshold_used=audit.threshold_used,
                is_verified=audit.is_verified,
                step_name=audit.step_name,
                device_id=audit.device_id,
                audit_date=audit.audit_date,
                metadata=audit.metadata,
            )
            for audit in audits
        ]

    identify_biometric = graphene.List(
        BiometricMatchType,
        modality=graphene.String(required=True),
        sample=graphene.String(required=True, description="Base64-encoded probe sample."),
        top_k=graphene.Int(required=False),
        exclude_subject=graphene.String(required=False),
        description="Rank the gallery for one modality against a probe sample.",
    )

    biometric_templates = graphene.List(
        BiometricTemplateType,
        subject_model=graphene.String(required=True),
        subject_id=graphene.String(required=True),
        description="Active template metadata for one subject (no plaintext).",
    )

    biometric_verifications = graphene.List(
        BiometricVerificationRecordType,
        subject_model=graphene.String(required=True),
        subject_id=graphene.String(required=True),
        description="Verification audit trail for one subject, newest first.",
    )

    @staticmethod
    def resolve_identify_biometric(root, info, modality, sample, top_k=5, exclude_subject=None):
        user = info.context.user
        _require_perms(user, BiometricVerificationConfig.gql_biometric_identify_perms)

        from .services import identify

        matches = identify(
            modality, sample=_decode_sample(sample), top_k=top_k or 5, exclude_subject=exclude_subject,
        )
        return [
            BiometricMatchType(
                subject_model=m.subject_model, subject_id=m.subject_id,
                template_id=m.template_id, score=m.score,
            )
            for m in matches
        ]

    @staticmethod
    def resolve_biometric_templates(root, info, subject_model, subject_id):
        user = info.context.user
        _require_perms(user, BiometricVerificationConfig.gql_biometric_read_perms)

        from .models import BiometricTemplate

        rows = BiometricTemplate.objects.filter(
            subject_model=subject_model, subject_id=str(subject_id), validity_to__isnull=True,
        )
        return [
            BiometricTemplateType(
                id=str(row.id), subject_model=row.subject_model, subject_id=row.subject_id,
                modality=row.modality, position=row.position, kind=row.kind, quality=row.quality,
                provider=row.provider, model_name=row.model_name, encrypted=row.encrypted,
                validity_from=row.validity_from, validity_to=row.validity_to,
            )
            for row in rows
        ]

    @staticmethod
    def resolve_biometric_verifications(root, info, subject_model, subject_id):
        user = info.context.user
        _require_perms(user, BiometricVerificationConfig.gql_biometric_read_perms)

        from .models import BiometricVerification as BiometricVerificationModel

        rows = BiometricVerificationModel.objects.filter(
            subject_model=subject_model, subject_id=str(subject_id),
        ).order_by("-created_at")
        return [
            BiometricVerificationRecordType(
                id=str(row.id), subject_model=row.subject_model, subject_id=row.subject_id,
                modality=row.modality, score=row.score, threshold=row.threshold, verified=row.verified,
                origin=row.origin, fallback=row.fallback, device_id=row.device_id,
                actor=row.actor, created_at=row.created_at,
            )
            for row in rows
        ]


class Mutation(graphene.ObjectType):
    verify_face = VerifyFaceMutation.Field()
    compute_insuree_embedding = ComputeInsureeEmbeddingMutation.Field()
    create_claim_facial_audit = CreateClaimFacialAuditMutation.Field()
    enrol_biometric = EnrolBiometricMutation.Field()
    verify_biometric = VerifyBiometricMutation.Field()
    record_biometric_consent = RecordBiometricConsentMutation.Field()
