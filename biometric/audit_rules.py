"""
Alert rules over the biometric audit chain (docs/wb-biometric-dedup-seam.md §6.10).

Rules run from transaction.on_commit after an event is recorded
(audit_chain.record_event). Each rule reads the events table only and raises
or refreshes a BiometricAlert. A failing rule is logged and never reaches the
committed biometric operation; the other rules still run. Events whose action
starts with "alert." (triage) are never evaluated, so triage cannot raise
alerts.

Rules are configured under BIOMETRIC["AUDIT"]["rules"]; each kind's dict
overrides DEFAULT_RULES key by key.
"""

import logging
from datetime import timedelta

from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError, transaction
from django.utils import timezone

from . import audit_chain

logger = logging.getLogger(__name__)

FAILED_VERIFICATIONS = "FAILED_VERIFICATIONS"
IMPERSONATION_SUSPECTED = "IMPERSONATION_SUSPECTED"
ACCESS_BURST = "ACCESS_BURST"

DEFAULT_RULES = {
    FAILED_VERIFICATIONS: {
        "enabled": True,
        "severity": "MEDIUM",
        "threshold": 3,
        "window_minutes": 60,
        "per_modality": True,
    },
    IMPERSONATION_SUSPECTED: {
        "enabled": True,
        "severity": "HIGH",
    },
    ACCESS_BURST: {
        "enabled": True,
        "severity": "HIGH",
        "max_events": 200,
        "window_minutes": 60,
        "actions": [
            audit_chain.ACTION_TEMPLATE_READ, audit_chain.ACTION_TEMPLATE_LIST, audit_chain.ACTION_IDENTIFY,
        ],
    },
}

SEVERITIES = ("LOW", "MEDIUM", "HIGH")

AUDIT_KEYS = ("enabled", "rules")


def _bool(value):
    if not isinstance(value, bool):
        return "expected true or false"
    return None


def _severity(value):
    if value not in SEVERITIES:
        return f"expected one of {', '.join(SEVERITIES)}"
    return None


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return "expected a positive integer"
    return None


def _actions(value):
    if not isinstance(value, list) or not value:
        return "expected a non-empty list of audit actions"
    for item in value:
        if not isinstance(item, str) or item not in audit_chain.ACTIONS:
            return f"unknown audit action {item!r}"
        if item.startswith(audit_chain.ALERT_ACTION_PREFIX):
            return f"alert triage action {item!r} cannot be watched"
    return None


# Every parameter a rule reads, with the check its value must pass.
PARAM_VALIDATORS = {
    "enabled": _bool,
    "severity": _severity,
    "threshold": _positive_int,
    "window_minutes": _positive_int,
    "max_events": _positive_int,
    "per_modality": _bool,
    "actions": _actions,
}


def _param_errors(kind, params):
    errors = []
    defaults = DEFAULT_RULES[kind]
    for name, value in params.items():
        if name not in defaults:
            errors.append(f"rules.{kind}: unknown parameter {name!r}")
            continue
        problem = PARAM_VALIDATORS[name](value)
        if problem:
            errors.append(f"rules.{kind}.{name} is {value!r}: {problem}")
    return errors


def validate_audit_config(cfg) -> list:
    """Every problem in a BIOMETRIC["AUDIT"] dict, as messages; never raises."""
    if cfg is None:
        return []
    if not isinstance(cfg, dict):
        return [f"expected a dict, got {type(cfg).__name__}"]
    errors = []
    for key in cfg:
        if key not in AUDIT_KEYS:
            errors.append(f"unknown key {key!r}; expected one of {', '.join(AUDIT_KEYS)}")
    if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
        errors.append(f"enabled is {cfg['enabled']!r}: expected true or false")
    rules = cfg.get("rules", {})
    if rules is None:
        return errors
    if not isinstance(rules, dict):
        errors.append(f"rules is {type(rules).__name__}: expected a dict")
        return errors
    for kind, override in rules.items():
        if kind not in DEFAULT_RULES:
            errors.append(f"rules: unknown rule kind {kind!r}; expected one of {', '.join(DEFAULT_RULES)}")
            continue
        if not isinstance(override, dict):
            errors.append(f"rules.{kind} is {type(override).__name__}: expected a dict")
            continue
        errors.extend(_param_errors(kind, override))
    return errors


def rule_params(kind) -> dict:
    """DEFAULT_RULES[kind] overridden key by key; raises ImproperlyConfigured when the result is invalid."""
    from .apps import BiometricConfig

    audit = BiometricConfig.audit if isinstance(BiometricConfig.audit, dict) else {}
    rules = audit.get("rules") if isinstance(audit.get("rules"), dict) else {}
    override = rules.get(kind) or {}
    if not isinstance(override, dict):
        raise ImproperlyConfigured(f"BIOMETRIC['AUDIT'] rules.{kind} must be a dict")
    params = {**DEFAULT_RULES[kind], **override}
    errors = _param_errors(kind, params)
    if errors:
        raise ImproperlyConfigured("BIOMETRIC['AUDIT']: " + "; ".join(errors))
    return params


def _bump(alert, *, detail, event, now):
    alert.occurrences += 1
    alert.last_seen_at = now
    alert.detail = detail
    alert.trigger_event = event
    alert.save(update_fields=["occurrences", "last_seen_at", "detail", "trigger_event"])
    return alert


def raise_alert(*, rule_kind, dedupe_key, severity, title, detail, event, subject_model="", subject_id=""):
    """Opens an alert, or bumps the open one (NEW or ACKNOWLEDGED) with the same key."""
    from .models import BiometricAlert

    now = timezone.now()

    def _open():
        return (
            BiometricAlert.objects.select_for_update()
            .filter(rule_kind=rule_kind, dedupe_key=dedupe_key, state__in=BiometricAlert.OPEN_STATES)
            .first()
        )

    with transaction.atomic():
        existing = _open()
        if existing is not None:
            return _bump(existing, detail=detail, event=event, now=now)
        try:
            with transaction.atomic():
                return BiometricAlert.objects.create(
                    rule_kind=rule_kind, severity=severity, title=title[:200], detail=detail,
                    dedupe_key=dedupe_key, subject_model=subject_model or "", subject_id=subject_id or "",
                    trigger_event=event, triggered_at=now, last_seen_at=now,
                )
        except IntegrityError:
            # A concurrent creator opened the alert first (biometric_alert_one_open_per_key).
            existing = _open()
            if existing is None:
                raise
            return _bump(existing, detail=detail, event=event, now=now)


def _window(event, params):
    return event.created_at - timedelta(minutes=params["window_minutes"]), event.created_at


def check_failed_verifications(event, params):
    """Alerts when a subject collects `threshold` failed verifies within the window."""
    from .models import BiometricAuditEvent

    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.action != audit_chain.ACTION_VERIFY or payload.get("verified") is not False:
        return None

    since, until = _window(event, params)
    failures = BiometricAuditEvent.objects.filter(
        action=audit_chain.ACTION_VERIFY,
        subject_model=event.subject_model,
        subject_id=event.subject_id,
        payload__verified=False,
        created_at__gte=since,
        created_at__lte=until,
    )
    per_modality = params["per_modality"]
    if per_modality:
        failures = failures.filter(modality=event.modality)

    attempts = failures.count()
    if attempts < params["threshold"]:
        return None

    actors = sorted({a for a in failures.order_by().values_list("actor", flat=True).distinct() if a})
    device_ids = sorted({
        d for d in failures.order_by().values_list("payload__device_id", flat=True).distinct()
        if isinstance(d, str) and d
    })
    dedupe_key = f"failed_verifications:{event.subject_model}:{event.subject_id}"
    if per_modality:
        dedupe_key += f":{event.modality}"

    return raise_alert(
        rule_kind=FAILED_VERIFICATIONS,
        dedupe_key=dedupe_key,
        severity=params["severity"],
        title=f"{attempts} failed biometric verifications within {params['window_minutes']} minutes",
        detail={
            "attempts": attempts,
            "threshold": params["threshold"],
            "window_minutes": params["window_minutes"],
            "modality": event.modality if per_modality else None,
            "actors": actors,
            "device_ids": device_ids,
        },
        event=event,
        subject_model=event.subject_model,
        subject_id=event.subject_id,
    )


def check_impersonation_suspected(event, params):
    """Alerts on every impersonation.suspected event; the alert's subject is the claimed subject."""
    if event.action != audit_chain.ACTION_IMPERSONATION:
        return None

    payload = event.payload if isinstance(event.payload, dict) else {}
    matched_subject_model = payload.get("matched_subject_model") or ""
    matched_subject_id = payload.get("matched_subject_id") or ""
    return raise_alert(
        rule_kind=IMPERSONATION_SUSPECTED,
        dedupe_key=(
            f"impersonation:{event.subject_model}:{event.subject_id}:"
            f"{matched_subject_model}:{matched_subject_id}"
        ),
        severity=params["severity"],
        title=f"Possible impersonation on a {event.modality or 'biometric'} verification",
        detail={
            "verification_id": payload.get("verification_id"),
            "modality": event.modality,
            "matched_subject_model": matched_subject_model,
            "matched_subject_id": matched_subject_id,
            "matched_template_id": payload.get("matched_template_id"),
            "matched_score": payload.get("matched_score"),
            "claimed_score": payload.get("claimed_score"),
            "threshold": payload.get("threshold"),
            "margin": payload.get("margin"),
        },
        event=event,
        subject_model=event.subject_model,
        subject_id=event.subject_id,
    )


def check_access_burst(event, params):
    """Alerts when one actor records `max_events` watched actions within the window."""
    from .models import BiometricAuditEvent

    actions = params["actions"]
    actor = event.actor or ""
    if event.action not in actions or not actor.strip():
        return None

    since, until = _window(event, params)
    events = BiometricAuditEvent.objects.filter(
        actor=actor, action__in=actions, created_at__gte=since, created_at__lte=until,
    )
    count = events.count()
    if count < params["max_events"]:
        return None

    distinct_subjects = (
        events.exclude(subject_id="").order_by().values("subject_model", "subject_id").distinct().count()
    )
    return raise_alert(
        rule_kind=ACCESS_BURST,
        dedupe_key=f"access_burst:{actor}",
        severity=params["severity"],
        title=f"{count} biometric reads by one account within {params['window_minutes']} minutes",
        detail={
            "actor": actor,
            "events": count,
            "max_events": params["max_events"],
            "window_minutes": params["window_minutes"],
            "actions": list(actions),
            "distinct_subjects": distinct_subjects,
        },
        event=event,
    )


CHECKS = {
    FAILED_VERIFICATIONS: check_failed_verifications,
    IMPERSONATION_SUSPECTED: check_impersonation_suspected,
    ACCESS_BURST: check_access_burst,
}


def evaluate_event(event_id) -> list:
    """Runs every enabled rule against one recorded event; returns the alerts raised or bumped."""
    from .models import BiometricAuditEvent

    event = BiometricAuditEvent.objects.filter(pk=event_id).first()
    if event is None or event.action.startswith(audit_chain.ALERT_ACTION_PREFIX):
        return []

    alerts = []
    for kind, check in CHECKS.items():
        try:
            params = rule_params(kind)
            if params.get("enabled") is not True:
                continue
            # A savepoint per rule: a database error in one rule leaves the
            # connection usable for the next.
            with transaction.atomic():
                alert = check(event, params)
        except Exception:
            logger.exception("biometric audit rule %s failed on event %s", kind, event_id)
            continue
        if alert is not None:
            alerts.append(alert)
    return alerts


def _triage(alert_id, *, actor, action, apply, note=""):
    from .models import BiometricAlert

    with transaction.atomic():
        alert = BiometricAlert.objects.select_for_update().get(pk=alert_id)
        apply(alert)
        audit_chain.record_event(
            action,
            actor=actor,
            subject_model=alert.subject_model,
            subject_id=alert.subject_id,
            payload={"alert_id": str(alert.id), "rule_kind": alert.rule_kind, "note": note or ""},
        )
    return alert


def acknowledge_alert(alert_id, *, actor):
    """NEW -> ACKNOWLEDGED, recorded as an alert.acknowledge event."""
    return _triage(
        alert_id, actor=actor, action=audit_chain.ACTION_ALERT_ACK,
        apply=lambda alert: alert.acknowledge(actor),
    )


def resolve_alert(alert_id, *, actor, note=""):
    """NEW or ACKNOWLEDGED -> RESOLVED with the note, recorded as an alert.resolve event."""
    return _triage(
        alert_id, actor=actor, action=audit_chain.ACTION_ALERT_RESOLVE, note=note,
        apply=lambda alert: alert.resolve(actor, note),
    )
