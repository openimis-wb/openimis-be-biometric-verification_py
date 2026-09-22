import base64

import graphene
from django.core.exceptions import PermissionDenied
from django.utils.translation import gettext as _

from .apps import BiometricConfig


def _decode_sample(sample):
    """Base64 sample, optionally with a data-URI prefix."""
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
        subject_id = graphene.String(required=True)
        modality = graphene.String(required=True)
        sample = graphene.String(required=True, description="Base64-encoded sample.")
        subject_model = graphene.String(required=False, description="Defaults to BIOMETRIC['SUBJECT_MODEL'].")
        position = graphene.String(required=False)
        metadata = graphene.JSONString(required=False)

    Output = BiometricTemplateType

    @classmethod
    def mutate(cls, root, info, subject_id, modality, sample, subject_model=None, position=None, metadata=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_enrol_perms)

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
        subject_id = graphene.String(required=True)
        modality = graphene.String(required=True)
        subject_model = graphene.String(required=False, description="Defaults to BIOMETRIC['SUBJECT_MODEL'].")
        sample = graphene.String(required=False, description="Base64-encoded sample (server path).")
        position = graphene.String(required=False)
        device_score = graphene.Float(required=False)
        fallback = graphene.Boolean(required=False)
        device_id = graphene.String(required=False)
        context = graphene.JSONString(required=False)

    Output = BiometricVerifyResultType

    @classmethod
    def mutate(cls, root, info, subject_id, modality, subject_model=None, sample=None, position=None,
               device_score=None, fallback=False, device_id="", context=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_verify_perms)

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
        subject_id = graphene.String(required=True)
        modality = graphene.String(required=True)
        granted = graphene.Boolean(required=True)
        subject_model = graphene.String(required=False, description="Defaults to BIOMETRIC['SUBJECT_MODEL'].")
        note = graphene.String(required=False)

    ok = graphene.Boolean(required=True)
    id = graphene.String()

    @classmethod
    def mutate(cls, root, info, subject_id, modality, granted, subject_model=None, note=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_enrol_perms)

        from .models import BiometricConsent

        subject_model = subject_model or BiometricConfig.subject_model
        consent = BiometricConsent.objects.create(
            subject_model=subject_model, subject_id=str(subject_id), modality=modality,
            granted=granted, recorded_by=user.username, note=note or "",
        )
        return cls(ok=True, id=str(consent.id))


# ---------------------------------------------------------------------------
# Root types — registered by the openIMIS schema aggregator
# ---------------------------------------------------------------------------

class Query(graphene.ObjectType):
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
        subject_id=graphene.String(required=True),
        subject_model=graphene.String(required=False, description="Defaults to BIOMETRIC['SUBJECT_MODEL']."),
        description="Active template metadata for one subject (no plaintext).",
    )

    biometric_verifications = graphene.List(
        BiometricVerificationRecordType,
        subject_id=graphene.String(required=True),
        subject_model=graphene.String(required=False, description="Defaults to BIOMETRIC['SUBJECT_MODEL']."),
        description="Verification audit trail for one subject, newest first.",
    )

    @staticmethod
    def resolve_identify_biometric(root, info, modality, sample, top_k=5, exclude_subject=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_identify_perms)

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
    def resolve_biometric_templates(root, info, subject_id, subject_model=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_read_perms)

        from .models import BiometricTemplate

        subject_model = subject_model or BiometricConfig.subject_model
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
    def resolve_biometric_verifications(root, info, subject_id, subject_model=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_read_perms)

        from .models import BiometricVerification as BiometricVerificationModel

        subject_model = subject_model or BiometricConfig.subject_model
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
    enrol_biometric = EnrolBiometricMutation.Field()
    verify_biometric = VerifyBiometricMutation.Field()
    record_biometric_consent = RecordBiometricConsentMutation.Field()
