"""
Impersonation probe (docs/wb-biometric-dedup-seam.md §6.9).

A 1:N search run inside the server path of services.verify(): the probe
already extracted for the 1:1 comparison is ranked against the whole gallery
of its modality, and a foreign subject scoring at or above the probe
threshold is reported as a possible impersonation. The probe never changes
the 1:1 score, threshold or verdict.

On the device-reported path (a device score, no sample) the probe ranks what
the device extracted, and only when "device_path" is true: device_path_probe()
returns the probe, or the reason an enabled probe did not run.

Configured under BIOMETRIC["IMPERSONATION_PROBE"]; nested keys are lowercase:

    {"enabled": False, "modalities": ["face"], "top_k": 5,
     "thresholds": {}, "margin": None, "device_path": False}
"""

import logging
import traceback
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
FAILURE_MESSAGE = "impersonation probe failed"

PROBE_DEFAULTS = {
    "enabled": False,
    "modalities": ["face"],
    "top_k": 5,
    "thresholds": {},
    "margin": None,
    "device_path": False,
}

# Why an enabled probe did not run on the device path, in the order they are checked.
SKIP_PROVIDER_MATCHES_ON_DEVICE = "provider_matches_on_device"   # the provider has no server-side match()
SKIP_NO_DEVICE_TEMPLATE = "no_device_template"                   # the device sent no vector / template
SKIP_DEVICE_PATH_DISABLED = "device_path_disabled"               # IMPERSONATION_PROBE["device_path"] is off
SKIP_REASONS = (SKIP_PROVIDER_MATCHES_ON_DEVICE, SKIP_NO_DEVICE_TEMPLATE, SKIP_DEVICE_PATH_DISABLED)


def probe_settings() -> Dict[str, Any]:
    """
    BIOMETRIC["IMPERSONATION_PROBE"] over PROBE_DEFAULTS. A stored module
    configuration or a settings override may carry only some keys; the
    missing ones take their default here.
    """
    from .apps import BiometricConfig

    configured = BiometricConfig.impersonation_probe
    return {**PROBE_DEFAULTS, **(configured if isinstance(configured, dict) else {})}


def probe_threshold(modality: str, provider, settings: Dict[str, Any]) -> float:
    """
    First value set among thresholds[modality], DEDUP_THRESHOLD[modality],
    MODALITIES[modality]["threshold"], then provider.default_threshold.
    A risk profile never changes it.
    """
    from .apps import BiometricConfig

    candidates = (
        (settings.get("thresholds") or {}).get(modality),
        (BiometricConfig.dedup_threshold or {}).get(modality),
        (BiometricConfig.modalities or {}).get(modality, {}).get("threshold"),
    )
    for value in candidates:
        if value is not None:
            return float(value)
    return float(provider.default_threshold)


@dataclass(frozen=True)
class ImpersonationProbe:
    """Outcome of one probe; status is "ok" or "failed", and a failed probe is never suspected."""
    status: str
    suspected: bool
    threshold: Optional[float]
    margin: Optional[float]
    top_k: int
    claimed_score: Optional[float]
    best_match: Optional[Dict[str, Any]]
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    latency_ms: Optional[float] = None

    def as_evidence(self) -> Dict[str, Any]:
        """JSON-safe copy stored on BiometricVerification.impersonation_evidence."""
        return {
            "threshold": self.threshold,
            "margin": self.margin,
            "top_k": self.top_k,
            "claimed_score": self.claimed_score,
            "candidates": [dict(candidate) for candidate in self.candidates],
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


def maybe_probe(subject_model, subject_id, modality, provider, extracted) -> Optional[ImpersonationProbe]:
    """
    Run the probe for a server-path verify() of (subject_model, subject_id).
    None when the probe is disabled or the modality is not listed. Any
    exception yields a "failed" probe, never a raised error.
    """
    settings = probe_settings()
    if not settings.get("enabled") or modality not in (settings.get("modalities") or []):
        return None

    threshold = margin = None
    top_k = PROBE_DEFAULTS["top_k"]
    try:
        threshold = probe_threshold(modality, provider, settings)
        margin = settings.get("margin")
        margin = float(margin) if margin is not None else None
        top_k = int(settings.get("top_k") or PROBE_DEFAULTS["top_k"])
        return _run(subject_model, subject_id, modality, provider, extracted, threshold, margin, top_k)
    except Exception as exc:
        # An exception's message can quote stored ciphertext or the probe vector
        # (a failed decrypt returns the raw stored value; psycopg2 interpolates
        # parameters into its error text). Only the class name and the stack
        # frames are logged and recorded, never the message or the traceback's
        # final line.
        error = failure_error(exc)
        logger.warning(
            "Impersonation probe failed for modality %s: %s\n%s",
            modality, error, "".join(traceback.format_tb(exc.__traceback__)),
        )
        return ImpersonationProbe(
            status=STATUS_FAILED, suspected=False, threshold=threshold, margin=margin, top_k=top_k,
            claimed_score=None, best_match=None, candidates=[], error=error,
        )


def device_path_probe(subject_model, subject_id, modality, provider, device_template
                      ) -> Tuple[Optional[ImpersonationProbe], str]:
    """
    (probe, skip reason) for a device-path verify() of (subject_model, subject_id).
    (None, "") when the probe is disabled or the modality is not listed.
    Otherwise the first reason that applies, in SKIP_REASONS order: a
    DeviceReportedMatcher cannot rank the gallery on the server; the device
    supplied no vector (embedding kind) or template (template kind);
    "device_path" is off. When none applies, maybe_probe() ranks the device's
    vector or template exactly as it ranks a server extraction.
    """
    from .providers.device_reported import DeviceReportedMatcher

    settings = probe_settings()
    if not settings.get("enabled") or modality not in (settings.get("modalities") or []):
        return None, ""
    if isinstance(provider, DeviceReportedMatcher):
        return None, SKIP_PROVIDER_MATCHES_ON_DEVICE
    supplied = None
    if device_template is not None:
        supplied = device_template.vector if provider.kind == "embedding" else device_template.template
    if supplied is None:
        return None, SKIP_NO_DEVICE_TEMPLATE
    if not settings.get("device_path"):
        return None, SKIP_DEVICE_PATH_DISABLED
    return maybe_probe(subject_model, subject_id, modality, provider, device_template), ""


def failure_error(exc: BaseException) -> str:
    """The error recorded for a failed probe: the exception class name and a fixed message."""
    return f"{type(exc).__name__}: {FAILURE_MESSAGE}"


def _run(subject_model, subject_id, modality, provider, extracted, threshold, margin, top_k):
    """
    The claimed subject is dropped from identify()'s ranking by its
    (subject_model, subject_id) pair. identify() is asked for top_k + n rows,
    n being the claimed subject's active templates on the gallery key, so
    top_k foreign rows are still returned when all n rank first.
    """
    from django.db import transaction

    from .models import BiometricTemplate
    from .services import identify

    # A savepoint: a database error here rolls back to it and leaves the
    # caller's transaction usable for the verification row.
    with transaction.atomic():
        claimed_templates = BiometricTemplate.objects.filter(
            subject_model=subject_model, subject_id=subject_id, modality=modality,
            provider=provider.provider_name, model_name=getattr(provider, "model_name", ""),
            kind=provider.kind, validity_to__isnull=True,
        ).count()
        started = perf_counter()
        matches = identify(
            modality, vector=extracted.vector, template=extracted.template, top_k=top_k + claimed_templates,
        )
    latency_ms = (perf_counter() - started) * 1000.0

    claimed_score = None
    best_by_subject: Dict[tuple, Any] = {}
    for match in matches:
        key = (match.subject_model, match.subject_id)
        if key == (subject_model, subject_id):
            if claimed_score is None or match.score > claimed_score:
                claimed_score = float(match.score)
            continue
        kept = best_by_subject.get(key)
        if kept is None or match.score > kept.score:
            best_by_subject[key] = match

    foreign = sorted(best_by_subject.values(), key=lambda m: m.score, reverse=True)
    candidates = []
    for match in foreign:
        score = float(match.score)
        if score < threshold:
            continue
        suspect = margin is None or claimed_score is None or score >= claimed_score - margin
        candidates.append({
            "subject_model": match.subject_model,
            "subject_id": match.subject_id,
            "template_id": str(match.template_id),
            "score": score,
            "suspect": suspect,
        })
        if len(candidates) >= top_k:
            break

    best_match = next((candidate for candidate in candidates if candidate["suspect"]), None)
    return ImpersonationProbe(
        status=STATUS_OK, suspected=best_match is not None, threshold=threshold, margin=margin,
        top_k=top_k, claimed_score=claimed_score, best_match=best_match, candidates=candidates,
        latency_ms=latency_ms,
    )
