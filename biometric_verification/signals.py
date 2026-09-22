"""
Django signals for biometric verification module.

Automatically updates Claim risk scores when facial audits are created/modified.
"""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from core.service_signals import ServiceSignalBindType
from core.signals import bind_service_signal

from .legacy import HAS_CLAIM

logger = logging.getLogger(__name__)


if HAS_CLAIM:
    from .models import ClaimFacialAudit

    @receiver(post_save, sender=ClaimFacialAudit)
    def update_claim_risk_score_on_audit(sender, instance, created, **kwargs):
        """
        Signal handler: Automatically recalculate claim fraud risk score
        whenever a ClaimFacialAudit is created or updated.

        Stores the risk assessment in claim.json_ext['biometric_risk_assessment'].

        Args:
            sender: The model class (ClaimFacialAudit)
            instance: The ClaimFacialAudit instance that was saved
            created: Boolean - True if this is a new record
            **kwargs: Additional signal arguments

        Note: ClaimFacialAudit is an immutable audit trail - no soft-delete.
        """
        from .services import BiometricService

        claim_id = instance.claim_id
        action = "created" if created else "updated"

        try:
            # Calculate the global risk score for this claim
            risk_data = BiometricService.calculate_global_risk_score(claim_id)

            # Store risk assessment in Claim.json_ext
            from claim.models import Claim

            claim = Claim.objects.get(id=claim_id)

            # Initialize json_ext if it doesn't exist
            if claim.json_ext is None:
                claim.json_ext = {}

            # Store the complete risk assessment
            claim.json_ext['biometric_risk_assessment'] = {
                'risk_score': risk_data['risk_score'],
                'risk_level': risk_data['risk_level'],
                'audit_count': risk_data['audit_count'],
                'failed_audits': risk_data['failed_audits'],
                'score_variance': risk_data['score_variance'],
                'avg_similarity': risk_data['avg_similarity'],
                'last_updated': instance.audit_date.isoformat(),
            }

            claim.save()

            logger.info(
                f"Audit {action} for Claim {claim_id} - "
                f"Risk score updated: {risk_data['risk_score']} "
                f"({risk_data['risk_level']}) - stored in json_ext"
            )

        except Exception as e:
            logger.error(
                f"Failed to update risk score for Claim {claim_id} "
                f"after audit {action}: {e}",
                exc_info=True
            )


# ---------------------------------------------------------------------------
# Deduplication seam (docs/wb-biometric-dedup-seam.md §2.2, §3.6)
#
# bind_service_signals() is auto-discovered and invoked by
# openIMIS/signal_binding/apps.py for every app in OPENIMIS_APPS. Binding
# here does not require the deduplication package to be installed — the
# signal is only ever fired if that module later registers and emits it.
# ---------------------------------------------------------------------------

def on_subject_merged(**kwargs):
    """
    AFTER handler for deduplication.subject_merged — calls consolidate().

    The payload is `subject_model, kept_id, retired_id, actor, policy` as
    flat kwargs (contract §2.2). Also tolerates the shape produced by
    core.signals.register_service_signal's automatic before/after wrapper
    (`data=[args, kwargs]`, `result`), in case the emitting side chooses
    that mechanism instead of sending the signal directly.
    """
    try:
        payload = _extract_subject_merged_payload(kwargs)
        if payload is None:
            logger.warning(
                "deduplication.subject_merged: could not read subject_model/"
                "kept_id/retired_id/actor from signal kwargs: %s", list(kwargs)
            )
            return

        from .services import consolidate

        consolidate(
            payload["subject_model"],
            payload["kept_id"],
            payload["retired_id"],
            actor=payload.get("actor", "deduplication"),
        )
    except Exception:
        logger.exception("consolidate() failed for deduplication.subject_merged")


def _extract_subject_merged_payload(kwargs):
    """Read the subject_merged fields from flat kwargs, or from a data/result wrapper."""
    if "subject_model" in kwargs and "kept_id" in kwargs and "retired_id" in kwargs:
        return kwargs

    for candidate in (kwargs.get("result"), kwargs.get("data")):
        if isinstance(candidate, dict) and "subject_model" in candidate:
            return candidate
        if isinstance(candidate, (list, tuple)):
            for part in candidate:
                if isinstance(part, dict) and "subject_model" in part:
                    return part
    return None


def bind_service_signals():
    bind_service_signal(
        "deduplication.subject_merged",
        on_subject_merged,
        bind_type=ServiceSignalBindType.AFTER,
    )
