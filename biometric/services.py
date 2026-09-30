import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, FrozenSet, List, Optional

from .providers.base import VerificationResult
from .quality import QualityRefusedError  # noqa: F401  (re-exported for callers)
from .risk_profiles import RiskProfileError, UnknownRiskProfileError  # noqa: F401  (re-exported for callers)

logger = logging.getLogger(__name__)

# BiometricTemplate.metadata key holding the provider's preprocessing tag (§6.13);
# written by enrol() only, never taken from the caller.
PREPROCESSING_KEY = "preprocessing"
# Why a template was left out of a comparison: its tag differs from the provider's.
PREPROCESSING_MISMATCH = "preprocessing_mismatch"


def provider_preprocessing(provider) -> str:
    """The provider's preprocessing tag; "" when it declares none."""
    value = getattr(provider, "preprocessing", "")
    return value if isinstance(value, str) else ""


def template_preprocessing(row) -> str:
    """The tag stored on a BiometricTemplate; "" when the row carries none."""
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    value = metadata.get(PREPROCESSING_KEY, "")
    return value if isinstance(value, str) else ""


def comparable_preprocessing(row, provider) -> bool:
    """True when the row was extracted under the provider's current preprocessing."""
    return template_preprocessing(row) == provider_preprocessing(provider)


def log_preprocessing_skips(log, skipped, modality, where, level=logging.INFO):
    """Logs how many templates a comparison left out for PREPROCESSING_MISMATCH; silent for none."""
    if skipped:
        log.log(level, "%s: %d %s template(s) skipped: %s", where, skipped, modality, PREPROCESSING_MISMATCH)


class DevicePathRefusedError(ValueError):
    """A device score was given for a modality whose provider matches on the server."""

    def __init__(self, modality):
        # graphql-core copies .extensions onto the GraphQL error it reports.
        self.extensions = {"code": "BIOMETRIC_DEVICE_PATH_REFUSED"}
        super().__init__(
            f"Modality '{modality}' is matched on the server; a device score is accepted only for a "
            "device_reported provider."
        )


class ConsentRequiredError(PermissionError):
    """Raised by enrol() when REQUIRE_CONSENT is set and the latest consent for the modality is not a grant."""


@dataclass(frozen=True)
class Match:
    """One identify() result — a candidate gallery hit."""
    subject_model: str
    subject_id: str
    template_id: str
    score: float


@dataclass(frozen=True)
class Decision:
    """fuse() result."""
    outcome: str                 # "accept" | "review" | "reject"
    score: Optional[float]
    reasons: List[str] = field(default_factory=list)
    risk_profile: str = ""       # the named profile the rules were resolved under, "" for the base


@dataclass(frozen=True)
class MultimodalVerification:
    """verify_multimodal() result: the fused decision and each leg's verify() result, in leg order."""
    decision: Decision
    legs: List[VerificationResult]
    decision_id: Optional[str] = None   # the BiometricMultimodalDecision row that stores the decision


def enrol(subject_model=None, subject_id=None, modality=None, sample=None, *, position=None, actor,
          metadata=None, device_template=None):
    """
    Extract (or accept a device-supplied) template and store it as the active
    one for (subject, modality, position, provider, model_name). Any previous
    active row on that same key is superseded (validity_to=now), never updated.

    The quality gate (biometric/quality.py, §6.7) assesses the sample after
    extraction; its verdict is stored on the row as quality_verdict. In enforce
    mode a REFUSED verdict raises QualityRefusedError before anything is
    superseded or written.

    With BIOMETRIC["AUDIT"] enabled (§6.10), the supersede, the insert and a
    template.enrol audit event share one transaction. A sample refused in
    enforce mode records a template.enrol_refused event (verdict and provider
    metadata, never the sample or the vector), committed before
    QualityRefusedError is raised; a caller transaction that rolls back on
    that error discards it.

    metadata["preprocessing"] holds the provider's preprocessing tag (§6.13),
    also for a device template; a caller's value for that key is dropped.

    subject_model defaults to BIOMETRIC["SUBJECT_MODEL"] when omitted (§6.3).
    """
    from django.utils import timezone

    from . import crypto
    from .apps import BiometricConfig
    from .audit_chain import ACTION_ENROL, ACTION_ENROL_REFUSED, audited_block, record_event
    from .models import BiometricConsent, BiometricTemplate
    from .quality import REFUSED, assess
    from .quality import mode as quality_mode
    from .registry import ProviderRegistry

    if subject_id is None or modality is None or sample is None:
        raise TypeError("enrol() requires subject_id, modality and sample.")

    subject_model = subject_model or BiometricConfig.subject_model
    subject_id = str(subject_id)
    position = position or ""
    metadata = dict(metadata or {})

    if BiometricConfig.require_consent:
        # The latest decision for (subject, modality) wins: a later refusal revokes.
        latest = BiometricConsent.objects.filter(
            subject_model=subject_model,
            subject_id=subject_id,
            modality=modality,
        ).order_by("-recorded_at").values_list("granted", flat=True).first()
        if not latest:
            raise ConsentRequiredError(
                f"No granted consent for modality '{modality}' on {subject_model}:{subject_id}."
            )

    provider = ProviderRegistry.get_provider(modality)
    extracted = device_template if device_template is not None else provider.extract(sample, position=position)
    model_name = getattr(provider, "model_name", "")
    stored_metadata = {**extracted.metadata, **metadata}
    stored_metadata.pop(PREPROCESSING_KEY, None)
    if provider_preprocessing(provider):
        stored_metadata[PREPROCESSING_KEY] = provider_preprocessing(provider)

    verdict = assess(
        modality, sample, extracted, server_extracted=device_template is None, mode_value=quality_mode(),
    )
    if verdict.mode == "enforce" and verdict.status == REFUSED:
        logger.info("enrol(): %s sample refused by the quality gate: %s", modality, verdict.reasons)
        # The event commits with this block; the error is raised after it.
        with audited_block():
            record_event(
                ACTION_ENROL_REFUSED, actor=actor, subject_model=subject_model, subject_id=subject_id,
                modality=modality,
                payload={
                    "position": position,
                    "provider": provider.provider_name,
                    "model_name": model_name,
                    "kind": provider.kind,
                    "quality": extracted.quality,
                    "device_template": device_template is not None,
                    "quality_status": verdict.status,
                    "quality_mode": verdict.mode,
                    "quality_reasons": list(verdict.reasons),
                    "quality_measures": [m.as_dict() for m in verdict.measures],
                },
            )
        raise QualityRefusedError(verdict)

    key = BiometricConfig.template_key
    crypto.warn_if_unencrypted(key)
    encrypted = key is not None

    vector = crypto.encrypt_vector(extracted.vector, key) if extracted.vector is not None else None
    template_bytes = crypto.encrypt_bytes(extracted.template, key) if extracted.template is not None else None
    template_iso_bytes = (
        crypto.encrypt_bytes(extracted.template_iso, key) if extracted.template_iso is not None else None
    )

    now = timezone.now()
    with audited_block():
        superseded = []
        # Superseded row-by-row (not a queryset .update()) so post_save fires for
        # each one — biometric_pgvector relies on it to drop the stale side row.
        for stale in BiometricTemplate.objects.filter(
            subject_model=subject_model, subject_id=subject_id, modality=modality,
            position=position, provider=provider.provider_name, model_name=model_name,
            validity_to__isnull=True,
        ):
            stale.validity_to = now
            stale.save(update_fields=["validity_to"])
            superseded.append(str(stale.id))

        template = BiometricTemplate.objects.create(
            subject_model=subject_model, subject_id=subject_id, modality=modality,
            position=position, kind=provider.kind,
            vector=vector, template=template_bytes, template_iso=template_iso_bytes,
            encrypted=encrypted, quality=extracted.quality,
            provider=provider.provider_name, model_name=model_name,
            metadata=stored_metadata,
            quality_verdict=verdict.as_dict(),
        )
        record_event(
            ACTION_ENROL, actor=actor, subject_model=subject_model, subject_id=subject_id, modality=modality,
            payload={
                "template_id": str(template.id),
                "position": position,
                "provider": provider.provider_name,
                "model_name": model_name,
                "kind": provider.kind,
                "quality": extracted.quality,
                "encrypted": encrypted,
                "device_template": device_template is not None,
                "superseded": superseded,
                "quality_status": verdict.status,
                "quality_reasons": list(verdict.reasons),
            },
        )
    return template


def verify(subject_model=None, subject_id=None, modality=None, *, sample=None, position=None,
           device_score=None, fallback=False, context=None, device_id="", actor, risk_profile=None,
           device_template=None):
    """
    Server path: extract the probe, compare with every active template of the
    subject for this modality (and position, if given), keep the best score.
    Device path (device_score given): nothing is extracted, the reported score
    is checked against the modality threshold directly. Both record a
    BiometricVerification row. The device path is open only to a modality
    whose provider is a DeviceReportedMatcher; any other raises
    DevicePathRefusedError before anything is written.

    risk_profile names a BIOMETRIC["RISK_PROFILES"] entry (§6.8); it can only
    raise the modality threshold. The row records the profile name and the
    effective threshold.

    When BIOMETRIC["IMPERSONATION_PROBE"] is enabled for the modality, the
    server path also ranks the extracted probe against the whole gallery
    (biometric/impersonation.py, §6.9). Its outcome is recorded on the row and
    returned as result.impersonation; score, threshold and verified are never
    changed by it. A suspicion emits biometric.impersonation_suspected after
    commit. On the device path the probe ranks device_template (an Extracted
    carrying the device's vector or template) only when
    IMPERSONATION_PROBE["device_path"] is true; an enabled probe that does not
    run records its reason as impersonation_skip_reason. device_template on
    the server path raises ValueError. An unknown name raises UnknownRiskProfileError, and a
    malformed or looser profile RiskProfileError, before extraction and before
    any row is written.

    The server path compares only templates recorded under the provider's
    preprocessing (§6.13); a template left out sets template_skip_reason to
    PREPROCESSING_MISMATCH on the row and the result.

    With BIOMETRIC["AUDIT"] enabled (§6.10), the row and a verify audit event
    (plus an impersonation.suspected event on a suspicion) share one
    transaction, opened after the probe.

    subject_model defaults to BIOMETRIC["SUBJECT_MODEL"] when omitted (§6.3).
    """
    from . import crypto
    from .apps import BiometricConfig
    from .audit_chain import ACTION_VERIFY, audited_block, record_event
    from .models import BiometricTemplate
    from .models import BiometricVerification as BiometricVerificationModel
    from .registry import ProviderRegistry

    if subject_id is None or modality is None:
        raise TypeError("verify() requires subject_id and modality.")
    if device_template is not None and device_score is None:
        raise ValueError("verify(): device_template applies to the device path only (device_score).")

    subject_model = subject_model or BiometricConfig.subject_model
    subject_id = str(subject_id)
    context = dict(context or {})
    modality_cfg = BiometricConfig.modalities.get(modality, {})

    provider = ProviderRegistry.get_provider(modality)
    if device_score is not None:
        _check_device_path(modality, provider)
    threshold = modality_cfg.get("threshold")
    if threshold is None:
        threshold = provider.default_threshold
    if risk_profile:
        from .risk_profiles import verify_threshold

        threshold = verify_threshold(risk_profile, modality, threshold)

    probe = None
    skip_reason = ""
    template_skip_reason = ""
    if device_score is not None:
        origin = "device"
        score = device_score
        verified = score >= threshold

        from .impersonation import device_path_probe

        probe, skip_reason = device_path_probe(subject_model, subject_id, modality, provider, device_template)
    else:
        origin = "server"
        if sample is None:
            raise ValueError("verify() server path requires sample.")
        extracted = provider.extract(sample, position=position)

        qs = BiometricTemplate.objects.filter(
            subject_model=subject_model, subject_id=subject_id, modality=modality,
            validity_to__isnull=True,
        )
        if position:
            qs = qs.filter(position=position)

        key = BiometricConfig.template_key
        best = None
        skipped = 0
        for row in qs:
            if not comparable_preprocessing(row, provider):
                skipped += 1
                continue
            try:
                row_key = crypto.row_key(row.encrypted, key)
                if row.kind == "embedding":
                    stored = crypto.decrypt_vector(row.vector, row_key)
                    candidate = provider.similarity(extracted.vector, stored)
                else:
                    stored = crypto.decrypt_bytes(row.template, row_key)
                    candidate = provider.match(extracted.template, stored)
            except crypto.TemplateKeyError:
                logger.error("verify(): template %s is marked encrypted and does not decrypt", row.id)
                raise
            except Exception:
                logger.debug("verify(): skipping incomparable template %s", row.id, exc_info=True)
                continue
            if best is None or candidate > best:
                best = candidate

        score = best
        verified = score is not None and score >= threshold
        log_preprocessing_skips(logger, skipped, modality, "verify()", level=logging.WARNING)
        template_skip_reason = PREPROCESSING_MISMATCH if skipped else ""

        from .impersonation import maybe_probe

        probe = maybe_probe(subject_model, subject_id, modality, provider, extracted)

    impersonation_fields = {}
    if probe is not None:
        best_match = probe.best_match or {}
        impersonation_fields = {
            "impersonation_status": probe.status,
            "impersonation_suspected": probe.suspected,
            "impersonation_subject_model": best_match.get("subject_model", ""),
            "impersonation_subject_id": best_match.get("subject_id", ""),
            "impersonation_score": best_match.get("score"),
            "impersonation_evidence": probe.as_evidence(),
        }

    # The probe above runs before this block, so the audit lock is never held during it.
    with audited_block():
        row = BiometricVerificationModel.objects.create(
            subject_model=subject_model, subject_id=subject_id, modality=modality,
            score=score, threshold=threshold, verified=verified, origin=origin,
            fallback=fallback, context=context, device_id=device_id or "", actor=actor,
            risk_profile=risk_profile or "", impersonation_skip_reason=skip_reason,
            template_skip_reason=template_skip_reason, **impersonation_fields,
        )
        record_event(
            ACTION_VERIFY, actor=actor, subject_model=subject_model, subject_id=subject_id, modality=modality,
            payload={
                "verification_id": str(row.id),
                "verified": bool(verified),
                "score": score,
                "threshold": threshold,
                "origin": origin,
                "fallback": bool(fallback),
                "device_id": device_id or "",
                "position": position or "",
                "risk_profile": risk_profile or "",
                "impersonation_status": probe.status if probe is not None else "",
                "impersonation_suspected": bool(probe is not None and probe.suspected),
                "impersonation_skip_reason": skip_reason,
                "template_skip_reason": template_skip_reason,
            },
        )
        if probe is not None and probe.suspected:
            record_impersonation_suspected(
                subject_model, subject_id,
                modality=modality,
                verification_id=str(row.id),
                matched_subject_model=probe.best_match["subject_model"],
                matched_subject_id=probe.best_match["subject_id"],
                matched_template_id=probe.best_match["template_id"],
                matched_score=probe.best_match["score"],
                claimed_score=probe.claimed_score,
                threshold=probe.threshold,
                margin=probe.margin,
                actor=actor,
            )

    if probe is not None and probe.suspected:
        from django.db import transaction

        from . import signals

        logger.warning("verify(): impersonation suspected on verification %s (%s)", row.id, modality)
        signal_payload = {
            "verification_id": str(row.id),
            "subject_model": subject_model,
            "subject_id": subject_id,
            "modality": modality,
            "matched_subject_model": probe.best_match["subject_model"],
            "matched_subject_id": probe.best_match["subject_id"],
            "matched_template_id": probe.best_match["template_id"],
            "matched_score": probe.best_match["score"],
            "claimed_score": probe.claimed_score,
            "threshold": probe.threshold,
            "margin": probe.margin,
            "actor": actor,
            "device_id": device_id or "",
            "context": dict(context),
        }
        transaction.on_commit(lambda: signals.emit_impersonation_suspected(**signal_payload))

    return VerificationResult(
        verification_id=str(row.id),
        verified=verified,
        confidence=score,
        provider=getattr(provider, "provider_name", None),
        modality=modality,
        origin=origin,
        threshold=threshold,
        risk_profile=risk_profile or "",
        impersonation=probe,
        impersonation_skip_reason=skip_reason,
        template_skip_reason=template_skip_reason,
    )


def _check_device_path(modality, provider):
    """DevicePathRefusedError unless the modality's provider matches on the device."""
    from .providers.device_reported import DeviceReportedMatcher

    if not isinstance(provider, DeviceReportedMatcher):
        raise DevicePathRefusedError(modality)


_LEG_KEYS = frozenset({"modality", "sample", "position", "device_score", "device_template"})


def _check_legs(legs):
    """ValueError unless legs is a non-empty list of distinct modalities, each with a sample or a device score."""
    if not isinstance(legs, (list, tuple)) or not legs:
        raise ValueError("verify_multimodal() requires a non-empty list of legs.")
    seen = set()
    for leg in legs:
        if not isinstance(leg, dict):
            raise ValueError("Each leg must be a dict.")
        unknown = sorted(set(leg) - _LEG_KEYS)
        if unknown:
            raise ValueError(f"Unknown leg keys {unknown}; allowed: {sorted(_LEG_KEYS)}.")
        modality = leg.get("modality")
        if not isinstance(modality, str) or not modality:
            raise ValueError("Each leg needs a modality.")
        if modality in seen:
            raise ValueError(f"Modality '{modality}' appears in more than one leg.")
        seen.add(modality)
        has_sample = leg.get("sample") is not None
        has_score = leg.get("device_score") is not None
        if has_sample == has_score:
            raise ValueError(f"Leg '{modality}' needs exactly one of sample and device_score.")
        if leg.get("device_template") is not None and not has_score:
            raise ValueError(f"Leg '{modality}': device_template goes with device_score only.")


def verify_multimodal(subject_model=None, subject_id=None, legs=None, *, fallback=False, context=None,
                      device_id="", actor, risk_profile=None):
    """
    Verify one subject on several modalities and fuse the leg scores.

    legs is a list of {"modality", "sample" | "device_score", "position"?,
    "device_template"?}, one per modality; device_template goes with
    device_score only. Each leg runs verify() with the same risk_profile,
    fallback, context, device_id and actor, and records its own
    BiometricVerification row. fuse() then combines the leg scores under the
    configured BIOMETRIC["FUSION"] rules and the same profile, so every
    profile key applies: thresholds, floors, floor_decision, required and
    modality_thresholds. Callers pass no fusion rule of their own.

    The legs, the modalities' providers (a device-score leg needs a
    device_reported provider) and the profile are checked before any leg runs; a leg that fails later (e.g. no face in its sample) leaves
    the rows of the legs before it and stores no decision.

    The fused decision is stored as a BiometricMultimodalDecision row listing
    the leg verification ids, and recorded as a verify.multimodal audit event
    in the same block when BIOMETRIC["AUDIT"] is enabled (§6.10). The event
    names the claimed subject only; a leg's impersonation match stays on that
    leg's row and events.
    """
    from .apps import BiometricConfig
    from .audit_chain import ACTION_VERIFY_MULTIMODAL, audited_block, record_event
    from .models import BiometricMultimodalDecision
    from .registry import ProviderRegistry

    if subject_id is None:
        raise TypeError("verify_multimodal() requires subject_id.")
    _check_legs(legs)
    for leg in legs:
        provider = ProviderRegistry.get_provider(leg["modality"])
        if leg.get("device_score") is not None:
            _check_device_path(leg["modality"], provider)
    if risk_profile:
        from .risk_profiles import base_rules, resolve

        resolve(risk_profile, base_rules(fusion=BiometricConfig.fusion, modalities=BiometricConfig.modalities))

    results = [
        verify(
            subject_model, subject_id, leg["modality"],
            sample=leg.get("sample"), position=leg.get("position"), device_score=leg.get("device_score"),
            device_template=leg.get("device_template"), fallback=fallback, context=context,
            device_id=device_id, actor=actor, risk_profile=risk_profile,
        )
        for leg in legs
    ]
    decision = fuse({result.modality: result.confidence for result in results}, risk_profile=risk_profile)

    subject_model = subject_model or BiometricConfig.subject_model
    subject_id = str(subject_id)
    modalities = [result.modality for result in results]
    verification_ids = [result.verification_id for result in results]
    with audited_block():
        row = BiometricMultimodalDecision.objects.create(
            subject_model=subject_model, subject_id=subject_id, outcome=decision.outcome, score=decision.score,
            reasons=list(decision.reasons), risk_profile=decision.risk_profile, modalities=modalities,
            verification_ids=verification_ids, fallback=bool(fallback), device_id=device_id or "", actor=actor,
        )
        record_event(
            ACTION_VERIFY_MULTIMODAL, actor=actor, subject_model=subject_model, subject_id=subject_id,
            payload={
                "decision_id": str(row.id),
                "outcome": decision.outcome,
                "score": decision.score,
                "reasons": list(decision.reasons),
                "risk_profile": decision.risk_profile,
                "modalities": modalities,
                "verification_ids": verification_ids,
                "fallback": bool(fallback),
                "device_id": device_id or "",
            },
        )
    return MultimodalVerification(decision=decision, legs=results, decision_id=str(row.id))


def record_impersonation_suspected(subject_model, subject_id, *, modality, verification_id,
                                   matched_subject_model, matched_subject_id, matched_template_id,
                                   matched_score, claimed_score, threshold, margin, actor):
    """
    Records an impersonation.suspected audit event for the claimed subject
    (docs/wb-biometric-dedup-seam.md §6.10). Decides nothing: the caller has
    already found the suspicion. Returns the event, or None when audit is off.
    """
    from .audit_chain import ACTION_IMPERSONATION, record_event

    return record_event(
        ACTION_IMPERSONATION, actor=actor, subject_model=subject_model, subject_id=str(subject_id),
        modality=modality,
        payload={
            "verification_id": str(verification_id),
            "matched_subject_model": matched_subject_model,
            "matched_subject_id": str(matched_subject_id),
            "matched_template_id": str(matched_template_id) if matched_template_id is not None else None,
            "matched_score": matched_score,
            "claimed_score": claimed_score,
            "threshold": threshold,
            "margin": margin,
        },
    )


def _identify_gallery_queryset(provider, modality, scope, exclude_subject, kind):
    from .models import BiometricTemplate

    qs = BiometricTemplate.objects.filter(
        modality=modality,
        provider=provider.provider_name,
        model_name=getattr(provider, "model_name", ""),
        kind=kind,
        validity_to__isnull=True,
    )
    if exclude_subject is not None:
        qs = qs.exclude(subject_id=str(exclude_subject))
    if scope:
        for key, value in scope.items():
            qs = qs.filter(**{f"metadata__{key}": value})
    return qs


def _comparable_gallery(provider, modality, scope, exclude_subject, kind):
    """The gallery rows recorded under the provider's preprocessing; the others are counted and logged."""
    rows = []
    skipped = 0
    for row in _identify_gallery_queryset(provider, modality, scope, exclude_subject, kind):
        if comparable_preprocessing(row, provider):
            rows.append(row)
        else:
            skipped += 1
    log_preprocessing_skips(logger, skipped, modality, "identify()")
    return rows


class Gallery:
    """
    The comparable gallery of one modality's provider and model (numpy or
    template path), decrypted once and ranked against any number of probes.
    rank() drops exclude_subject's rows before the top_k cut, as identify()
    drops them from its query.
    """

    def __init__(self, provider, modality, scope=None, exclude_subject=None):
        from . import crypto
        from .apps import BiometricConfig

        self.provider = provider
        self.kind = provider.kind
        self.rows = _comparable_gallery(provider, modality, scope, exclude_subject, self.kind)
        key = BiometricConfig.template_key
        if self.kind == "embedding":
            self.stored = [crypto.decrypt_vector(row.vector, crypto.row_key(row.encrypted, key)) for row in self.rows]
        else:
            self.stored = [crypto.decrypt_bytes(row.template, crypto.row_key(row.encrypted, key)) for row in self.rows]
        self._index = {str(row.id): position for position, row in enumerate(self.rows)}
        self._unit = None

    def stored_value(self, template_id):
        """The decrypted vector or template of a gallery row; KeyError when the row is not in the gallery."""
        return self.stored[self._index[str(template_id)]]

    def rank(self, *, vector=None, template=None, top_k=5, exclude_subject=None) -> List[Match]:
        positions = [
            i for i, row in enumerate(self.rows)
            if exclude_subject is None or row.subject_id != str(exclude_subject)
        ]
        if not positions:
            return []
        if self.kind == "embedding":
            return self._rank_embedding(vector, top_k, positions)
        scored = [(self.provider.match(template, self.stored[i]), self.rows[i]) for i in positions]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Match(subject_model=row.subject_model, subject_id=row.subject_id, template_id=str(row.id), score=score)
            for score, row in scored[:top_k]
        ]

    def _rank_embedding(self, probe_vector, top_k, positions):
        """gallery @ probe after L2 normalisation."""
        import numpy as np

        if self._unit is None:
            gallery = np.array(self.stored, dtype=float)
            norms = np.linalg.norm(gallery, axis=1)
            norms[norms == 0] = 1.0
            self._unit = gallery / norms[:, None]
        probe = np.array(probe_vector, dtype=float)
        probe_norm = np.linalg.norm(probe) or 1.0

        similarities = self._unit[positions] @ (probe / probe_norm)
        order = np.argsort(-similarities)[:top_k]
        return [
            Match(
                subject_model=self.rows[positions[i]].subject_model,
                subject_id=self.rows[positions[i]].subject_id,
                template_id=str(self.rows[positions[i]].id),
                score=float(similarities[i]),
            )
            for i in order
        ]


def _identify_numpy(provider, modality, probe_vector, top_k, scope, exclude_subject):
    """Portable path: gallery @ probe after L2 normalisation. Always available."""
    return Gallery(provider, modality, scope, exclude_subject).rank(vector=probe_vector, top_k=top_k)


def _identify_pgvector(provider, modality, probe_vector, top_k, scope, exclude_subject):
    """
    VECTOR_INDEX == "pgvector" path: queries biometric_pgvector's side table
    (docs/wb-biometric-dedup-seam.md §6.2). Refuses before importing anything
    from that app or the pgvector package — the app, not the package, is the
    gate, so this stays importable when biometric_pgvector is not installed.

    Same cast expression as the partial HNSW index
    (embedding::vector(N)) vector_cosine_ops, so the planner can use it once
    biometric_vector_index --model --dim has created one; falls back to an
    exact sequential scan otherwise. similarity = 1 - cosine distance.
    """
    import json

    from django.apps import apps
    from django.core.exceptions import ImproperlyConfigured

    if not apps.is_installed("biometric_pgvector"):
        raise ImproperlyConfigured(
            "BIOMETRIC['VECTOR_INDEX'] is 'pgvector' but the 'biometric_pgvector' "
            "app is not installed."
        )

    from django.db import connection, transaction

    from biometric_pgvector.apps import BiometricPgvectorConfig

    model_name = getattr(provider, "model_name", "")
    dim = len(probe_vector)
    probe_literal = "[" + ",".join(repr(float(x)) for x in probe_vector) + "]"

    # The preprocessing filter sits before LIMIT, so a skipped row never takes a top_k place.
    where = [
        "bvi.modality = %s", "bvi.provider = %s", "bvi.model_name = %s",
        "bt.validity_to IS NULL",
        f"COALESCE(bt.metadata->>'{PREPROCESSING_KEY}', '') = %s",
    ]
    params = [modality, provider.provider_name, model_name, provider_preprocessing(provider)]

    if exclude_subject is not None:
        where.append("bt.subject_id != %s")
        params.append(str(exclude_subject))

    if scope:
        for key, value in scope.items():
            where.append("bt.metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))

    where_sql = " AND ".join(where)
    sql = f"""
        SELECT bt.subject_model, bt.subject_id, bvi.template_id,
               1 - ((bvi.embedding::vector({dim})) <=> %s::vector({dim})) AS score
        FROM biometric_vector_index bvi
        JOIN biometric_template bt ON bt.id = bvi.template_id
        WHERE {where_sql}
        ORDER BY (bvi.embedding::vector({dim})) <=> %s::vector({dim})
        LIMIT %s
    """

    with transaction.atomic():
        with connection.cursor() as cursor:
            # hnsw.ef_search takes a plain literal, not a bind parameter.
            cursor.execute(f"SET LOCAL hnsw.ef_search = {int(BiometricPgvectorConfig.hnsw_ef_search)}")
            cursor.execute(sql, [probe_literal, *params, probe_literal, top_k])
            rows = cursor.fetchall()

    return [
        Match(subject_model=subject_model, subject_id=subject_id, template_id=str(template_id), score=float(score))
        for subject_model, subject_id, template_id, score in rows
    ]


def _identify_template(provider, modality, probe_template, top_k, scope, exclude_subject):
    return Gallery(provider, modality, scope, exclude_subject).rank(template=probe_template, top_k=top_k)


def identify(modality, *, sample=None, vector=None, template=None, top_k=5,
             scope=None, exclude_subject=None, actor=None):
    """
    Rank the gallery (active templates for the modality's configured provider
    and model, filtered by scope on metadata keys) against one probe. Only
    templates recorded under the provider's preprocessing are ranked (§6.13).

    With actor given, the ranking is recorded as an identify audit event
    (§6.10); internal callers (the impersonation probe, the deduplication
    scan) pass none and record nothing.
    """
    from .apps import BiometricConfig
    from .registry import ProviderRegistry

    provider = ProviderRegistry.get_provider(modality)

    if sample is not None:
        extracted = provider.extract(sample)
        probe_vector, probe_template = extracted.vector, extracted.template
    else:
        probe_vector, probe_template = vector, template

    if provider.kind == "embedding":
        if BiometricConfig.vector_index == "pgvector":
            matches = _identify_pgvector(provider, modality, probe_vector, top_k, scope, exclude_subject)
        else:
            matches = _identify_numpy(provider, modality, probe_vector, top_k, scope, exclude_subject)
    else:
        matches = _identify_template(provider, modality, probe_template, top_k, scope, exclude_subject)

    if actor is not None:
        from .audit_chain import ACTION_IDENTIFY, record_event

        if sample is not None:
            probe_kind = "sample"
        elif vector is not None:
            probe_kind = "vector"
        else:
            probe_kind = "template"
        record_event(
            ACTION_IDENTIFY, actor=actor, modality=modality,
            payload={
                "top_k": top_k,
                "scope": scope,
                "exclude_subject": str(exclude_subject) if exclude_subject is not None else None,
                "probe": probe_kind,
                "matches": [
                    {
                        "subject_model": m.subject_model, "subject_id": m.subject_id,
                        "template_id": m.template_id, "score": m.score,
                    }
                    for m in matches
                ],
            },
        )
    return matches


_OUTCOME_RANK = {"reject": 0, "review": 1, "accept": 2}


def fuse(scores: Dict[str, Optional[float]], *, weights=None, thresholds=None,
         floors=None, floor_decision=None, required: FrozenSet[str] = frozenset(),
         risk_profile=None):
    """
    Weighted mean of present, positively-weighted legs, each normalised to its
    modality threshold (leg_score / leg_threshold, so 1.0 == at threshold).
    Rules only ever tighten the outcome: a missing required leg caps it at
    "review"; a leg below its floor caps it at floor_decision; otherwise the
    fused score is banded against thresholds["accept"]/["review"].

    risk_profile names a BIOMETRIC["RISK_PROFILES"] entry (§6.8) merged over
    the rules resolved above; it can only tighten them, and the Decision
    carries its name. An unknown name raises UnknownRiskProfileError, and a
    malformed or looser profile RiskProfileError, before any scoring.
    """
    from .apps import BiometricConfig

    fusion_cfg = BiometricConfig.fusion
    weights = weights if weights is not None else fusion_cfg.get("weights", {})
    thresholds = thresholds if thresholds is not None else fusion_cfg.get("thresholds", {})
    floors = floors if floors is not None else fusion_cfg.get("floors", {})
    floor_decision = floor_decision or fusion_cfg.get("floor_decision", "review")
    modality_thresholds = BiometricConfig.modalities

    leg_overrides = {}
    if risk_profile:
        from .risk_profiles import FusionRules, resolve

        base = FusionRules(
            thresholds={"accept": thresholds.get("accept", 1.0), "review": thresholds.get("review", 0.0)},
            floors=dict(floors),
            floor_decision=floor_decision,
            required=frozenset(required),
            modality_thresholds={
                m: c.get("threshold") for m, c in modality_thresholds.items() if c.get("threshold")
            },
        )
        rules = resolve(risk_profile, base)
        thresholds = rules.thresholds
        floors = rules.floors
        floor_decision = rules.floor_decision
        required = rules.required
        leg_overrides = rules.modality_thresholds

    reasons = []

    forced_review = False
    for modality in required:
        if scores.get(modality) is None:
            forced_review = True
            reasons.append(f"required modality '{modality}' has no score")

    floor_breach = False
    for modality, floor in floors.items():
        score = scores.get(modality)
        if score is not None and score < floor:
            floor_breach = True
            reasons.append(f"'{modality}' score {score} below floor {floor}")

    weighted_sum = 0.0
    weight_total = 0.0
    for modality, score in scores.items():
        if score is None:
            continue
        weight = weights.get(modality, 0.0)
        if weight <= 0:
            continue
        leg_threshold = modality_thresholds.get(modality, {}).get("threshold") or 1.0
        normalised = score / leg_threshold
        if leg_overrides.get(modality):
            # The lower of the two normalisations, so a raised threshold never
            # lifts a negative leg score.
            normalised = min(normalised, score / leg_overrides[modality])
        weighted_sum += normalised * weight
        weight_total += weight

    fused_score = weighted_sum / weight_total if weight_total > 0 else None

    if fused_score is None:
        outcome = "reject"
        reasons.append("no scored modality")
    elif fused_score >= thresholds.get("accept", 1.0):
        outcome = "accept"
    elif fused_score >= thresholds.get("review", 0.0):
        outcome = "review"
    else:
        outcome = "reject"

    rank = _OUTCOME_RANK[outcome]
    if floor_breach:
        rank = min(rank, _OUTCOME_RANK[floor_decision])
    if forced_review:
        rank = min(rank, _OUTCOME_RANK["review"])
    outcome = next(name for name, value in _OUTCOME_RANK.items() if value == rank)

    return Decision(outcome=outcome, score=fused_score, reasons=reasons, risk_profile=risk_profile or "")


def consolidate(subject_model=None, kept_id=None, retired_id=None, *, actor):
    """
    Re-point retired's active templates to kept. Where kept already holds an
    active row on the same (modality, position, provider, model_name), the
    retired row is superseded instead of moved. Writes one access log entry.
    Bound to deduplication.subject_merged (signals.py).

    subject_model defaults to BIOMETRIC["SUBJECT_MODEL"] when omitted (§6.3).
    """
    from django.db import transaction
    from django.utils import timezone

    from .apps import BiometricConfig
    from .audit_chain import ACTION_CONSOLIDATE, record_event
    from .models import BiometricAccessLog, BiometricTemplate

    if kept_id is None or retired_id is None:
        raise TypeError("consolidate() requires kept_id and retired_id.")

    subject_model = subject_model or BiometricConfig.subject_model
    kept_id = str(kept_id)
    retired_id = str(retired_id)
    counts = {}

    with transaction.atomic():
        retired_templates = list(
            BiometricTemplate.objects.select_for_update().filter(
                subject_model=subject_model, subject_id=retired_id, validity_to__isnull=True,
            )
        )
        now = timezone.now()
        for row in retired_templates:
            collides = BiometricTemplate.objects.filter(
                subject_model=subject_model, subject_id=kept_id, modality=row.modality,
                position=row.position, provider=row.provider, model_name=row.model_name,
                validity_to__isnull=True,
            ).exists()
            if collides:
                row.validity_to = now
                row.save(update_fields=["validity_to"])
            else:
                row.subject_id = kept_id
                row.date_updated = now
                row.save(update_fields=["subject_id", "date_updated"])
            counts[row.modality] = counts.get(row.modality, 0) + 1

        if retired_templates:
            template_ids = [str(row.id) for row in retired_templates]
            BiometricAccessLog.objects.create(
                subject_model=subject_model, subject_id=kept_id, actor=actor,
                purpose="consolidate", template_ids=template_ids,
            )
            record_event(
                ACTION_CONSOLIDATE, actor=actor, subject_model=subject_model, subject_id=kept_id,
                payload={"retired_id": retired_id, "counts": dict(counts), "template_ids": template_ids},
            )

    return counts


def templates_of(subject_model=None, subject_id=None, *, modality=None, actor, purpose="read"):
    """
    The only sanctioned plaintext read path: decrypts active templates for one
    subject and writes a BiometricAccessLog entry naming exactly which rows
    were read, by whom, and why.

    subject_model defaults to BIOMETRIC["SUBJECT_MODEL"] when omitted (§6.3).
    """
    from . import crypto
    from .apps import BiometricConfig
    from .audit_chain import ACTION_TEMPLATE_READ, audited_block, record_event
    from .models import BiometricAccessLog, BiometricTemplate

    if subject_id is None:
        raise TypeError("templates_of() requires subject_id.")

    subject_model = subject_model or BiometricConfig.subject_model
    subject_id = str(subject_id)
    key = BiometricConfig.template_key

    qs = BiometricTemplate.objects.filter(
        subject_model=subject_model, subject_id=subject_id, validity_to__isnull=True,
    )
    if modality:
        qs = qs.filter(modality=modality)

    rows = list(qs)
    results = []
    for row in rows:
        row_key = crypto.row_key(row.encrypted, key)
        results.append({
            "id": str(row.id),
            "modality": row.modality,
            "position": row.position,
            "kind": row.kind,
            "vector": crypto.decrypt_vector(row.vector, row_key) if row.kind == "embedding" else None,
            "template": crypto.decrypt_bytes(row.template, row_key) if row.kind == "template" else None,
            "template_iso": crypto.decrypt_bytes(row.template_iso, row_key) if row.template_iso is not None else None,
            "quality": row.quality,
            "provider": row.provider,
            "model_name": row.model_name,
        })

    template_ids = [r["id"] for r in results]
    with audited_block():
        BiometricAccessLog.objects.create(
            subject_model=subject_model, subject_id=subject_id, actor=actor,
            purpose=purpose, template_ids=template_ids,
        )
        record_event(
            ACTION_TEMPLATE_READ, actor=actor, subject_model=subject_model, subject_id=subject_id,
            modality=modality or "",
            payload={"purpose": purpose, "template_ids": template_ids, "modality": modality or None},
        )
    return results


def _erase(stale, *, reason, actor):
    """
    Delete the given stale BiometricTemplate rows, tombstoning per subject.
    With audit enabled, one template.purge event per tombstone is recorded in
    the same transaction.
    """
    from .audit_chain import ACTION_PURGE, audited_block, record_event
    from .models import BiometricErasure, BiometricTemplate

    if not stale:
        return None

    by_subject: Dict[tuple, Dict[str, int]] = {}
    for row in stale:
        subject_key = (row.subject_model, row.subject_id)
        by_subject.setdefault(subject_key, {})
        by_subject[subject_key][row.modality] = by_subject[subject_key].get(row.modality, 0) + 1

    with audited_block():
        BiometricTemplate.objects.filter(id__in=[row.id for row in stale]).delete()

        tombstones = [
            BiometricErasure.objects.create(
                subject_model=subject_model, subject_id=subject_id,
                modalities=list(erased.keys()), erased=erased,
                reason=reason, erased_by=actor,
            )
            for (subject_model, subject_id), erased in by_subject.items()
        ]
        for tombstone in tombstones:
            record_event(
                ACTION_PURGE, actor=actor, subject_model=tombstone.subject_model, subject_id=tombstone.subject_id,
                payload={"erasure_id": str(tombstone.id), "reason": reason, "erased": dict(tombstone.erased)},
            )
    return tombstones[-1]


def current_retention_policy():
    """The BiometricRetentionPolicy row purge() applies, or None when there is none."""
    from .models import BiometricRetentionPolicy

    return BiometricRetentionPolicy.objects.first()


def purge(now=None, *, actor="retention"):
    """
    Erase templates past retention. Two independent, off-by-default passes:

    1. Superseded templates past template_retention_days, when purge_enabled.
    2. Still-active templates past active_template_retention_days, when
       purge_active_enabled (§6.3) — run after pass 1, tombstoned with
       reason="ACTIVE_AGE".

    Returns the last BiometricErasure tombstone written across both passes,
    or None if neither pass erased anything (including when there is no
    policy row at all).
    """
    from django.utils import timezone

    from .models import BiometricTemplate

    policy = current_retention_policy()
    if policy is None:
        return None

    now = now or timezone.now()
    tombstone = None

    if policy.purge_enabled and policy.template_retention_days is not None:
        cutoff = now - timedelta(days=policy.template_retention_days)
        stale = list(
            BiometricTemplate.objects.filter(validity_to__isnull=False, validity_to__lt=cutoff)
        )
        result = _erase(stale, reason="retention", actor=actor)
        if result is not None:
            tombstone = result

    if policy.purge_active_enabled and policy.active_template_retention_days is not None:
        active_cutoff = now - timedelta(days=policy.active_template_retention_days)
        stale_active = list(
            BiometricTemplate.objects.filter(validity_to__isnull=True, validity_from__lt=active_cutoff)
        )
        result = _erase(stale_active, reason="ACTIVE_AGE", actor=actor)
        if result is not None:
            tombstone = result

    return tombstone
