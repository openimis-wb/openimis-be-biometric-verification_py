import graphene
from django.core.exceptions import PermissionDenied
from django.utils.translation import gettext as _

from .apps import BiometricVerificationConfig


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


class Mutation(graphene.ObjectType):
    verify_face = VerifyFaceMutation.Field()
    compute_insuree_embedding = ComputeInsureeEmbeddingMutation.Field()
    create_claim_facial_audit = CreateClaimFacialAuditMutation.Field()
