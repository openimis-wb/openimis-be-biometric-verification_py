"""
Hash-chained audit events (docs/wb-biometric-dedup-seam.md §6.10).

Each BiometricAuditEvent carries its position in the chain (sequence), the
previous event's hash (prev_hash) and its own hash:

    hash = sha256(prev_hash || canonical_event(row))

with GENESIS_HASH as the prev_hash of sequence 1. Appends serialize on a
Postgres transaction-level advisory lock, so sequence order is append order.
prev_hash and sequence are UNIQUE: an append that escapes the lock raises
IntegrityError instead of forking the chain. verify_chain() walks the rows
by sequence and names the first divergence.

What the chain proves: altering, deleting or reordering a stored row
changes its recomputed hash or breaks the link of the row after it.

What it does not prove:
- Deleting the newest rows (tail truncation) leaves a coherent chain.
  Only a head (sequence, hash) recorded outside this database reveals it;
  biometric_audit_verify prints the head on every run; check_anchor() takes
  a recorded head back and requires the row at that sequence to still hold
  that hash. Rows appended after a recorded head are covered once a newer
  head is recorded.
- Someone holding both the database and the application can recompute a
  coherent chain from any point.
- created_at is asserted by this server. No timestamp authority signs it.

The feature is off unless BIOMETRIC["AUDIT"]["enabled"] is True; while off,
record_event() writes nothing, takes no lock and runs no query.
"""

import contextlib
import datetime
import hashlib
import json
import logging
import math
import numbers
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

from django.db import connection, transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

# Action names shared by the producers (services.py, schema.py) and the rules
# (audit_rules.py).
ACTION_ENROL = "template.enrol"
ACTION_ENROL_REFUSED = "template.enrol_refused"
ACTION_VERIFY = "verify"
ACTION_VERIFY_MULTIMODAL = "verify.multimodal"
ACTION_IDENTIFY = "identify"
ACTION_IMPERSONATION = "impersonation.suspected"
ACTION_CONSOLIDATE = "template.consolidate"
ACTION_PURGE = "template.purge"
ACTION_TEMPLATE_READ = "template.read"
ACTION_TEMPLATE_LIST = "template.list"
ACTION_ALERT_ACK = "alert.acknowledge"
ACTION_ALERT_RESOLVE = "alert.resolve"
ALERT_ACTION_PREFIX = "alert."

ACTIONS = frozenset({
    ACTION_ENROL,
    ACTION_ENROL_REFUSED,
    ACTION_VERIFY,
    ACTION_VERIFY_MULTIMODAL,
    ACTION_IDENTIFY,
    ACTION_IMPERSONATION,
    ACTION_CONSOLIDATE,
    ACTION_PURGE,
    ACTION_TEMPLATE_READ,
    ACTION_TEMPLATE_LIST,
    ACTION_ALERT_ACK,
    ACTION_ALERT_RESOLVE,
})

GENESIS_HASH = "0" * 64

# Fixed key for pg_advisory_xact_lock; unique among this deployment's advisory locks.
CHAIN_LOCK_KEY = 0x42494F4155444954

# Payload keys that name biometric material; refused at any depth.
FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "sample", "vector", "template", "template_iso", "embedding", "image", "frame", "probe_vector",
})

# A list holding more numbers than this is taken for a vector under another key.
MAX_NUMERIC_LIST = 16

# Floats at or above this magnitude serialise in exponent form without a
# fraction; Postgres jsonb returns them as integers.
_EXPONENT_FLOAT_MIN = 1e16


def audit_enabled() -> bool:
    """True only when BIOMETRIC["AUDIT"]["enabled"] is the boolean True."""
    from .apps import BiometricConfig

    audit = BiometricConfig.audit
    return isinstance(audit, dict) and audit.get("enabled") is True


def audited_block():
    """transaction.atomic() when audit is enabled, a no-op context otherwise."""
    if audit_enabled():
        return transaction.atomic()
    return contextlib.nullcontext()


def _clean(value, path):
    if hasattr(value, "dtype") and hasattr(value, "tolist"):
        # NumPy scalars become Python scalars, arrays become lists.
        value = value.tolist()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"audit payload value at {path} is binary; biometric material is never recorded")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(f"audit payload string at {path} contains a NUL character")
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"audit payload number at {path} is not finite")
        if number == 0.0:
            # jsonb has no negative zero: -0.0 reads back as 0.0.
            return 0.0
        if abs(number) >= _EXPONENT_FLOAT_MIN and number.is_integer():
            return int(number)
        return number
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key = str(key)
            if key.lower() in FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"audit payload key {path}.{key} names biometric material")
            if "\x00" in key:
                raise ValueError(f"audit payload key at {path} contains a NUL character")
            cleaned[key] = _clean(item, f"{path}.{key}")
        return cleaned
    if isinstance(value, (list, tuple)):
        numeric = sum(1 for item in value if isinstance(item, numbers.Real) and not isinstance(item, bool))
        if numeric > MAX_NUMERIC_LIST:
            raise ValueError(
                f"audit payload list at {path} holds {numeric} numbers; a vector is never recorded"
            )
        return [_clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"audit payload value at {path} has unsupported type {type(value).__name__}")


def sanitize_payload(payload) -> dict:
    """
    JSON-safe copy of payload, identical to what Postgres jsonb returns.
    Raises ValueError for a forbidden key, a numeric list longer than
    MAX_NUMERIC_LIST, NaN or infinity; TypeError for binary values.
    """
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("audit payload must be a dict")
    clean = _clean(payload, "payload")
    return json.loads(json.dumps(clean, sort_keys=True, allow_nan=False))


def _canonical_time(value) -> str:
    if timezone.is_aware(value):
        value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return value.isoformat(timespec="microseconds")


def canonical_event(event) -> str:
    """Byte-stable serialisation of the fields the hash covers."""
    return json.dumps(
        {
            "id": str(event.id),
            "sequence": event.sequence,
            "action": event.action,
            "actor": event.actor,
            "subject_model": event.subject_model,
            "subject_id": event.subject_id,
            "modality": event.modality,
            "payload": event.payload,
            "created_at": _canonical_time(event.created_at),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compute_hash(event, prev_hash: str) -> str:
    return hashlib.sha256(prev_hash.encode("utf-8") + canonical_event(event).encode("utf-8")).hexdigest()


def _lock_chain() -> None:
    if connection.vendor != "postgresql":
        return
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", [CHAIN_LOCK_KEY])


def record_event(action, *, actor, subject_model="", subject_id="", modality="", payload=None):
    """
    Appends one event to the chain and schedules the alert rules for after
    commit. Returns the saved BiometricAuditEvent, or None when audit is off.
    The advisory lock is held until the outermost transaction commits.
    """
    if not audit_enabled():
        return None
    if action not in ACTIONS:
        raise ValueError(f"unknown biometric audit action {action!r}")

    from .models import BiometricAuditEvent

    clean = sanitize_payload(payload)

    with transaction.atomic():
        _lock_chain()
        head = BiometricAuditEvent.objects.order_by("-sequence").values("sequence", "hash").first()
        prev_hash = head["hash"] if head else GENESIS_HASH
        event = BiometricAuditEvent(
            id=uuid.uuid4(),
            sequence=(head["sequence"] + 1) if head else 1,
            action=action,
            actor=str(actor or ""),
            subject_model=str(subject_model or ""),
            subject_id=str(subject_id or ""),
            modality=str(modality or ""),
            payload=clean,
            created_at=timezone.now(),
            prev_hash=prev_hash,
        )
        event.hash = compute_hash(event, prev_hash)
        event.save(force_insert=True)

        event_id = event.id

        def _evaluate():
            from .audit_rules import evaluate_event

            evaluate_event(event_id)

        transaction.on_commit(_evaluate)
    return event


def chain_head() -> Tuple[int, str]:
    """(last sequence, head hash), or (0, GENESIS_HASH) for an empty chain."""
    from .models import BiometricAuditEvent

    head = BiometricAuditEvent.objects.order_by("-sequence").values("sequence", "hash").first()
    if head is None:
        return 0, GENESIS_HASH
    return head["sequence"], head["hash"]


@dataclass(frozen=True)
class Divergence:
    """First point where the stored rows stop forming the chain."""
    kind: str       # "missing_event" | "broken_link" | "altered_row"
    sequence: int
    detail: str

    def __str__(self):
        return f"{self.kind} at sequence {self.sequence}: {self.detail}"


@dataclass(frozen=True)
class ChainReport:
    """checked counts the rows verified before the divergence, or all rows when intact."""
    checked: int
    head_sequence: int
    head_hash: str
    divergence: Optional[Divergence]

    @property
    def ok(self) -> bool:
        return self.divergence is None


def verify_chain(*, batch_size: int = 1000) -> ChainReport:
    """
    Walks the events by sequence from GENESIS_HASH and returns the first
    missing_event (a sequence gap, including a deleted first row),
    broken_link (prev_hash differs from the previous row's hash) or
    altered_row (the recomputed hash differs from the stored one).
    """
    from .models import BiometricAuditEvent

    expected_seq = 1
    expected_prev = GENESIS_HASH
    checked = 0
    divergence = None
    for row in BiometricAuditEvent.objects.order_by("sequence").iterator(chunk_size=batch_size):
        if row.sequence != expected_seq:
            divergence = Divergence(
                "missing_event", expected_seq,
                f"expected sequence {expected_seq}, found {row.sequence}",
            )
            break
        if row.prev_hash != expected_prev:
            divergence = Divergence(
                "broken_link", row.sequence,
                f"prev_hash {row.prev_hash} does not match the previous hash {expected_prev}",
            )
            break
        recomputed = compute_hash(row, row.prev_hash)
        if recomputed != row.hash:
            divergence = Divergence(
                "altered_row", row.sequence,
                f"stored hash {row.hash} does not match the recomputed hash {recomputed}",
            )
            break
        checked += 1
        expected_seq += 1
        expected_prev = row.hash

    head_sequence, head_hash = chain_head()
    return ChainReport(checked=checked, head_sequence=head_sequence, head_hash=head_hash, divergence=divergence)


def check_anchor(report: ChainReport, *, sequence: Optional[int] = None,
                 head_hash: Optional[str] = None) -> Optional[str]:
    """
    Checks a head (sequence, hash) recorded outside this database against an
    intact report from verify_chain(). The chain may have grown since: the
    row at the recorded sequence must still exist and hold the recorded hash.
    With only a hash, some row must hold it; with only a sequence, the chain
    must reach it. Sequence 0 stands for the empty chain and GENESIS_HASH.
    Returns None when the anchor holds, otherwise the reason it does not.
    """
    from .models import BiometricAuditEvent

    if sequence is not None and sequence < 0:
        return f"recorded head sequence {sequence} is negative"
    if sequence is not None and sequence > report.head_sequence:
        return (
            f"chain ends at sequence {report.head_sequence}, before the recorded head "
            f"sequence {sequence}: the tail was truncated or rewritten"
        )
    if head_hash is None:
        return None
    if sequence is None:
        if head_hash == GENESIS_HASH or BiometricAuditEvent.objects.filter(hash=head_hash).exists():
            return None
        return f"no event holds the recorded head hash {head_hash}: the tail was truncated or rewritten"
    if sequence == 0:
        stored = GENESIS_HASH
    else:
        stored = BiometricAuditEvent.objects.filter(sequence=sequence).values_list("hash", flat=True).first()
    if stored != head_hash:
        return (
            f"event at sequence {sequence} has hash {stored}, recorded head hash is {head_hash}: "
            "the tail was truncated or rewritten"
        )
    return None


def record_chain_check(*, actor, batch_size: int = 1000):
    """
    Runs verify_chain() and stores its outcome as a BiometricAuditChainCheck
    row, returned. The row is not an audit event, so the head it records is
    still the head once it is written.
    """
    return store_chain_check(verify_chain(batch_size=batch_size), actor=actor)


def store_chain_check(report: ChainReport, *, actor):
    """Stores a verify_chain() report as a BiometricAuditChainCheck row, returned; appends no event."""
    from .models import BiometricAuditChainCheck

    divergence = report.divergence
    return BiometricAuditChainCheck.objects.create(
        checked_by=str(actor or ""),
        ok=report.ok,
        checked=report.checked,
        head_sequence=report.head_sequence,
        head_hash=report.head_hash,
        divergence_kind=divergence.kind if divergence else "",
        divergence_sequence=divergence.sequence if divergence else None,
        divergence_detail=divergence.detail if divergence else "",
    )


def latest_chain_check():
    """The most recent BiometricAuditChainCheck, or None when the chain was never checked this way."""
    from .models import BiometricAuditChainCheck

    return BiometricAuditChainCheck.objects.order_by("-checked_at", "-id").first()
