"""
Deduplication seam (docs/wb-biometric-dedup-seam.md §2.2, §3.6).

bind_service_signals() is auto-discovered and invoked by
openIMIS/signal_binding/apps.py for every app in OPENIMIS_APPS. Binding
here does not require the deduplication package to be installed — the
signal is only ever fired if that module later registers and emits it.
"""

import logging

from core.service_signals import ServiceSignalBindType
from core.signals import bind_service_signal

logger = logging.getLogger(__name__)


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
