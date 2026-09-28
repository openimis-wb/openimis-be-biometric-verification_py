import base64

import graphene
from django.core.exceptions import PermissionDenied
from django.utils.translation import gettext as _
from graphene_django import DjangoObjectType

from core import ExtendedConnection
from core.schema import OrderedDjangoFilterConnectionField

from .apps import BiometricConfig
from .models import BiometricAlert, BiometricAuditEvent


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

class BiometricQualityMeasureType(graphene.ObjectType):
    """One quality gate measure beside the limit it is judged against."""
    name = graphene.String(required=True)
    value = graphene.Float()
    limit = graphene.Float()
    kind = graphene.String(required=True)
    passed = graphene.Boolean(description="Null when the measure is recorded, not judged.")
    source = graphene.String()
    detail = graphene.String()


class BiometricQualityVerdictType(graphene.ObjectType):
    """The quality gate's verdict on an enrolled sample (docs/wb-biometric-dedup-seam.md §6.7)."""
    status = graphene.String(required=True, description="ACCEPTED | REFUSED | NOT_ASSESSED")
    mode = graphene.String(required=True)
    modality = graphene.String(required=True)
    reasons = graphene.List(graphene.NonNull(graphene.String), required=True)
    measures = graphene.List(graphene.NonNull(BiometricQualityMeasureType), required=True)
    version = graphene.Int()


def _verdict_type(verdict):
    """BiometricQualityVerdictType from a stored verdict dict; None for anything else."""
    if not isinstance(verdict, dict):
        return None
    return BiometricQualityVerdictType(
        status=verdict.get("status"),
        mode=verdict.get("mode"),
        modality=verdict.get("modality"),
        reasons=list(verdict.get("reasons") or []),
        measures=[
            BiometricQualityMeasureType(
                name=m.get("name"), value=m.get("value"), limit=m.get("limit"), kind=m.get("kind"),
                passed=m.get("passed"), source=m.get("source"), detail=m.get("detail"),
            )
            for m in (verdict.get("measures") or [])
            if isinstance(m, dict)
        ],
        version=verdict.get("version"),
    )


class BiometricImpersonationCandidateType(graphene.ObjectType):
    """One foreign subject the impersonation probe ranked at or above its threshold."""
    subject_model = graphene.String(required=True)
    subject_id = graphene.String(required=True)
    template_id = graphene.String(required=True)
    score = graphene.Float(required=True)
    suspect = graphene.Boolean(required=True)


class BiometricImpersonationProbeType(graphene.ObjectType):
    """
    The impersonation probe run inside verify() (docs/wb-biometric-dedup-seam.md §6.9).
    matchedSubjectModel, matchedSubjectId and candidates are null / empty
    unless the caller holds the identify rights.
    """
    status = graphene.String(required=True, description="ok | failed")
    suspected = graphene.Boolean(required=True)
    threshold = graphene.Float()
    margin = graphene.Float()
    claimed_score = graphene.Float()
    matched_subject_model = graphene.String()
    matched_subject_id = graphene.String()
    matched_score = graphene.Float()
    candidates = graphene.List(BiometricImpersonationCandidateType)
    error = graphene.String()
    latency_ms = graphene.Float()


def _impersonation_gql(source, user):
    """
    BiometricImpersonationProbeType from an ImpersonationProbe or from a
    BiometricVerification row whose impersonation_status is set; None for
    anything else. Other subjects' identities need the identify rights.
    """
    from .impersonation import ImpersonationProbe

    if isinstance(source, ImpersonationProbe):
        status = source.status
        suspected = source.suspected
        best_match = source.best_match or {}
        evidence = source.as_evidence()
    else:
        status = getattr(source, "impersonation_status", None)
        if not isinstance(status, str) or not status:
            return None
        suspected = bool(source.impersonation_suspected)
        best_match = {
            "subject_model": source.impersonation_subject_model or None,
            "subject_id": source.impersonation_subject_id or None,
            "score": source.impersonation_score,
        }
        evidence = source.impersonation_evidence if isinstance(source.impersonation_evidence, dict) else {}

    may_identify = bool(user.has_perms(BiometricConfig.gql_biometric_identify_perms))
    candidates = []
    if may_identify:
        candidates = [
            BiometricImpersonationCandidateType(
                subject_model=c.get("subject_model"), subject_id=c.get("subject_id"),
                template_id=c.get("template_id"), score=c.get("score"), suspect=c.get("suspect"),
            )
            for c in (evidence.get("candidates") or [])
            if isinstance(c, dict)
        ]
    return BiometricImpersonationProbeType(
        status=status,
        suspected=suspected,
        threshold=evidence.get("threshold"),
        margin=evidence.get("margin"),
        claimed_score=evidence.get("claimed_score"),
        matched_subject_model=best_match.get("subject_model") if may_identify else None,
        matched_subject_id=best_match.get("subject_id") if may_identify else None,
        matched_score=best_match.get("score"),
        candidates=candidates,
        error=evidence.get("error") or None,
        latency_ms=evidence.get("latency_ms"),
    )


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
    quality_verdict = graphene.Field(
        BiometricQualityVerdictType, description="Null for rows the quality gate never ran on.",
    )


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
    risk_profile = graphene.String(description="The named risk profile applied; empty for the base rules.")
    impersonation = graphene.Field(
        BiometricImpersonationProbeType, description="Null when the impersonation probe did not run.",
    )


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
    risk_profile = graphene.String(description="The named risk profile applied; empty for the base rules.")
    impersonation = graphene.Field(
        BiometricImpersonationProbeType, description="Null when the impersonation probe did not run.",
    )


def _verify_result_gql(result, user):
    """BiometricVerifyResultType from a services.verify() result."""
    return BiometricVerifyResultType(
        verified=result.verified, confidence=result.confidence, provider=result.provider,
        modality=result.modality, origin=result.origin, threshold=result.threshold, error=result.error,
        risk_profile=result.risk_profile if isinstance(result.risk_profile, str) else "",
        impersonation=_impersonation_gql(getattr(result, "impersonation", None), user),
    )


class BiometricMultimodalVerifyResultType(graphene.ObjectType):
    """verifyBiometricMultimodal result: the fused decision and each leg's verification."""
    outcome = graphene.String(required=True, description="accept | review | reject")
    score = graphene.Float(description="Fused score; null when no weighted leg has a score.")
    reasons = graphene.List(graphene.NonNull(graphene.String), required=True)
    risk_profile = graphene.String(description="The named risk profile applied; empty for the base rules.")
    legs = graphene.List(graphene.NonNull(BiometricVerifyResultType), required=True)


class BiometricVerifyLegInput(graphene.InputObjectType):
    """One modality of a multimodal verification: a sample (server path) or a device score."""
    modality = graphene.String(required=True)
    sample = graphene.String(required=False, description="Base64-encoded sample (server path).")
    position = graphene.String(required=False)
    device_score = graphene.Float(required=False)


class EnrolBiometricMutation(graphene.Mutation):
    """
    Extract (or accept a device template for) one sample and store it as the subject's active template.

    The result carries the quality gate's verdict as qualityVerdict. In enforce
    mode a refused sample returns enrolBiometric: null and errors[0].extensions
    = {code: "BIOMETRIC_QUALITY_REFUSED", verdict: {...}}; QualityRefusedError
    is not caught here, graphql-core copies its .extensions onto the error.
    """

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
            quality_verdict=_verdict_type(getattr(template, "quality_verdict", None)),
        )


class VerifyBiometricMutation(graphene.Mutation):
    """
    Server path (sample given) or device path (deviceScore given).

    The impersonation probe runs from configuration only
    (BIOMETRIC["IMPERSONATION_PROBE"]); no argument turns it off.

    An unknown or invalid riskProfile is a GraphQL error, never verified=false:
    UnknownRiskProfileError and RiskProfileError are not caught here.
    """

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
        risk_profile = graphene.String(
            required=False,
            description="Named risk profile from BIOMETRIC['RISK_PROFILES']; it may only raise the modality threshold.",
        )

    Output = BiometricVerifyResultType

    @classmethod
    def mutate(cls, root, info, subject_id, modality, subject_model=None, sample=None, position=None,
               device_score=None, fallback=False, device_id="", context=None, risk_profile=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_verify_perms)

        from .services import verify

        result = verify(
            subject_model, subject_id, modality, sample=_decode_sample(sample), position=position,
            device_score=device_score, fallback=bool(fallback), context=context,
            device_id=device_id or "", actor=user.username, risk_profile=risk_profile or None,
        )
        return _verify_result_gql(result, user)


class VerifyBiometricMultimodalMutation(graphene.Mutation):
    """
    Verifies each leg (one per modality) and fuses the leg scores under
    BIOMETRIC["FUSION"] and the optional risk profile (docs/wb-biometric-dedup-seam.md §6.11).

    The fusion rules come from configuration and the profile only: no argument
    sets weights, thresholds, floors, floor_decision or required. An unknown
    or invalid riskProfile, or an invalid leg list, is a GraphQL error and no
    verification is recorded.
    """

    class Arguments:
        subject_id = graphene.String(required=True)
        subject_model = graphene.String(required=False, description="Defaults to BIOMETRIC['SUBJECT_MODEL'].")
        legs = graphene.List(graphene.NonNull(BiometricVerifyLegInput), required=True)
        risk_profile = graphene.String(
            required=False, description="Named risk profile from BIOMETRIC['RISK_PROFILES']; it may only tighten.",
        )
        fallback = graphene.Boolean(required=False)
        device_id = graphene.String(required=False)
        context = graphene.JSONString(required=False)

    Output = BiometricMultimodalVerifyResultType

    @classmethod
    def mutate(cls, root, info, subject_id, legs, subject_model=None, risk_profile=None, fallback=False,
               device_id="", context=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_verify_perms)

        from .services import verify_multimodal

        service_legs = []
        for leg in legs:
            service_leg = {"modality": leg.modality}
            if leg.sample is not None:
                service_leg["sample"] = _decode_sample(leg.sample)
            if leg.device_score is not None:
                service_leg["device_score"] = leg.device_score
            if leg.position:
                service_leg["position"] = leg.position
            service_legs.append(service_leg)

        result = verify_multimodal(
            subject_model, subject_id, service_legs, fallback=bool(fallback), context=context,
            device_id=device_id or "", actor=user.username, risk_profile=risk_profile or None,
        )
        return BiometricMultimodalVerifyResultType(
            outcome=result.decision.outcome, score=result.decision.score, reasons=list(result.decision.reasons),
            risk_profile=result.decision.risk_profile, legs=[_verify_result_gql(leg, user) for leg in result.legs],
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
# Audit chain and alerts (docs/wb-biometric-dedup-seam.md §6.10)
# ---------------------------------------------------------------------------

# Payload / detail keys naming a subject other than the event's own; shown
# only to callers holding gql_biometric_identify_perms.
_MATCHED_IDENTITY_KEYS = ("matched_subject_model", "matched_subject_id", "matched_template_id")
_MATCH_IDENTITY_KEYS = ("subject_model", "subject_id", "template_id")


def _may_identify(info):
    user = getattr(info.context, "user", None)
    return bool(user is not None and user.has_perms(BiometricConfig.gql_biometric_identify_perms))


def _event_payload_for(action, payload, info):
    """The event payload, without other subjects' identities unless the caller may identify."""
    from .audit_chain import ACTION_IDENTIFY, ACTION_IMPERSONATION

    payload = dict(payload) if isinstance(payload, dict) else {}
    if _may_identify(info):
        return payload
    if action == ACTION_IDENTIFY and isinstance(payload.get("matches"), list):
        payload["matches"] = [
            {k: v for k, v in match.items() if k not in _MATCH_IDENTITY_KEYS} if isinstance(match, dict) else match
            for match in payload["matches"]
        ]
    elif action == ACTION_IMPERSONATION:
        for key in _MATCHED_IDENTITY_KEYS:
            payload.pop(key, None)
    return payload


class BiometricAuditEventGQLType(DjangoObjectType):
    """One hash-chained audit event. Holds identifiers, scores and counts, never biometric material."""

    payload = graphene.JSONString()

    class Meta:
        model = BiometricAuditEvent
        interfaces = (graphene.relay.Node,)
        connection_class = ExtendedConnection
        fields = (
            "id", "sequence", "action", "actor", "subject_model", "subject_id", "modality",
            "payload", "created_at", "prev_hash", "hash",
        )
        filter_fields = {
            "sequence": ["exact", "lt", "gt"],
            "action": ["exact", "startswith"],
            "actor": ["exact"],
            "subject_model": ["exact"],
            "subject_id": ["exact"],
            "modality": ["exact"],
            "created_at": ["gte", "lte"],
        }

    def resolve_payload(self, info):
        return _event_payload_for(self.action, self.payload, info)


class BiometricAlertGQLType(DjangoObjectType):
    """An alert raised by an audit rule. The dedupe key is internal and not exposed."""

    detail = graphene.JSONString()
    trigger_event_id = graphene.String()

    class Meta:
        model = BiometricAlert
        interfaces = (graphene.relay.Node,)
        connection_class = ExtendedConnection
        convert_choices_to_enum = False
        fields = (
            "id", "rule_kind", "severity", "state", "title", "detail", "subject_model", "subject_id",
            "occurrences", "triggered_at", "last_seen_at", "acknowledged_by", "acknowledged_at",
            "resolved_by", "resolved_at", "resolution_note",
        )
        filter_fields = {
            "subject_model": ["exact"],
            "subject_id": ["exact"],
            "triggered_at": ["gte", "lte"],
        }

    def resolve_detail(self, info):
        from .audit_rules import IMPERSONATION_SUSPECTED

        detail = dict(self.detail) if isinstance(self.detail, dict) else {}
        if self.rule_kind == IMPERSONATION_SUSPECTED and not _may_identify(info):
            for key in _MATCHED_IDENTITY_KEYS:
                detail.pop(key, None)
        return detail

    def resolve_trigger_event_id(self, info):
        return str(self.trigger_event_id) if self.trigger_event_id else None


class AcknowledgeBiometricAlertMutation(graphene.Mutation):
    """NEW -> ACKNOWLEDGED. An alert in any other state is a GraphQL error."""

    class Arguments:
        id = graphene.String(required=True, description="The alert's UUID.")

    Output = BiometricAlertGQLType

    @classmethod
    def mutate(cls, root, info, id):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_alert_perms)

        from .audit_rules import acknowledge_alert

        return acknowledge_alert(id, actor=user.username)


class ResolveBiometricAlertMutation(graphene.Mutation):
    """NEW or ACKNOWLEDGED -> RESOLVED with an optional note. A resolved alert is a GraphQL error."""

    class Arguments:
        id = graphene.String(required=True, description="The alert's UUID.")
        note = graphene.String(required=False)

    Output = BiometricAlertGQLType

    @classmethod
    def mutate(cls, root, info, id, note=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_alert_perms)

        from .audit_rules import resolve_alert

        return resolve_alert(id, actor=user.username, note=note or "")


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

    biometric_audit_events = OrderedDjangoFilterConnectionField(
        BiometricAuditEventGQLType,
        orderBy=graphene.List(of_type=graphene.String),
        description="Hash-chained biometric audit events, newest first.",
    )

    biometric_alerts = OrderedDjangoFilterConnectionField(
        BiometricAlertGQLType,
        orderBy=graphene.List(of_type=graphene.String),
        state=graphene.String(),
        severity=graphene.String(),
        ruleKind=graphene.String(),
        open=graphene.Boolean(description="True: NEW or ACKNOWLEDGED only."),
        description="Alerts raised by the audit rules, newest first.",
    )

    @staticmethod
    def resolve_biometric_audit_events(root, info, **kwargs):
        _require_perms(info.context.user, BiometricConfig.gql_biometric_audit_perms)
        return BiometricAuditEvent.objects.order_by("-sequence")

    @staticmethod
    def resolve_biometric_alerts(root, info, **kwargs):
        _require_perms(info.context.user, BiometricConfig.gql_biometric_audit_perms)

        qs = BiometricAlert.objects.order_by("-triggered_at", "-id")
        if kwargs.get("state"):
            qs = qs.filter(state=kwargs["state"])
        if kwargs.get("severity"):
            qs = qs.filter(severity=kwargs["severity"])
        if kwargs.get("ruleKind"):
            qs = qs.filter(rule_kind=kwargs["ruleKind"])
        if kwargs.get("open") is True:
            qs = qs.filter(state__in=BiometricAlert.OPEN_STATES)
        elif kwargs.get("open") is False:
            qs = qs.exclude(state__in=BiometricAlert.OPEN_STATES)
        return qs

    @staticmethod
    def resolve_identify_biometric(root, info, modality, sample, top_k=5, exclude_subject=None):
        user = info.context.user
        _require_perms(user, BiometricConfig.gql_biometric_identify_perms)

        from .services import identify

        matches = identify(
            modality, sample=_decode_sample(sample), top_k=top_k or 5, exclude_subject=exclude_subject,
            actor=user.username,
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

        from .audit_chain import ACTION_TEMPLATE_LIST, audit_enabled, record_event
        from .models import BiometricTemplate

        subject_model = subject_model or BiometricConfig.subject_model
        rows = list(BiometricTemplate.objects.filter(
            subject_model=subject_model, subject_id=str(subject_id), validity_to__isnull=True,
        ))
        if audit_enabled():
            record_event(
                ACTION_TEMPLATE_LIST, actor=user.username, subject_model=subject_model, subject_id=str(subject_id),
                payload={"template_ids": [str(row.id) for row in rows]},
            )
        return [
            BiometricTemplateType(
                id=str(row.id), subject_model=row.subject_model, subject_id=row.subject_id,
                modality=row.modality, position=row.position, kind=row.kind, quality=row.quality,
                provider=row.provider, model_name=row.model_name, encrypted=row.encrypted,
                validity_from=row.validity_from, validity_to=row.validity_to,
                quality_verdict=_verdict_type(getattr(row, "quality_verdict", None)),
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
                actor=row.actor, created_at=row.created_at, risk_profile=row.risk_profile,
                impersonation=_impersonation_gql(row, user),
            )
            for row in rows
        ]


class Mutation(graphene.ObjectType):
    enrol_biometric = EnrolBiometricMutation.Field()
    verify_biometric = VerifyBiometricMutation.Field()
    verify_biometric_multimodal = VerifyBiometricMultimodalMutation.Field()
    record_biometric_consent = RecordBiometricConsentMutation.Field()
    acknowledge_biometric_alert = AcknowledgeBiometricAlertMutation.Field()
    resolve_biometric_alert = ResolveBiometricAlertMutation.Field()
