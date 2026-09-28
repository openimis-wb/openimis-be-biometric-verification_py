"""
Hash-chained audit events (docs/wb-biometric-dedup-seam.md §6.10): recording,
payload sanitisation, the verifier's divergence kinds, the ORM guards, the
fork backstop, concurrent appends and the biometric_audit_verify command.
"""

import datetime
import json
import threading
import uuid
from io import StringIO

import numpy as np
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, connection, transaction
from django.test import SimpleTestCase, TestCase, TransactionTestCase
from django.test.utils import CaptureQueriesContext

from biometric import audit_chain
from biometric.apps import BiometricConfig
from biometric.audit_chain import (
    ACTION_TEMPLATE_READ,
    ACTION_VERIFY,
    ACTIONS,
    GENESIS_HASH,
    audited_block,
    canonical_event,
    chain_head,
    compute_hash,
    record_event,
    sanitize_payload,
    verify_chain,
)
from biometric.models import BiometricAuditEvent

AUDIT_ON = {"enabled": True, "rules": {}}


AUDIT_SETTINGS = ("audit", "gql_biometric_audit_perms", "gql_biometric_alert_perms")


def restore_audit_settings_on_cleanup(test):
    """Registers a cleanup restoring the audit settings; it runs even when setUp fails."""
    for name in AUDIT_SETTINGS:
        test.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))


class AuditConfigMixin:
    """Restores the audit settings each test changes."""

    def setUp(self):
        restore_audit_settings_on_cleanup(self)
        super().setUp()


def _record(n, **kwargs):
    events = []
    for index in range(n):
        events.append(record_event(
            ACTION_VERIFY, actor="tester", subject_model="individual.Individual",
            subject_id=f"s{index}", modality="face", payload={"verified": True, "index": index, **kwargs},
        ))
    return events


class TestSanitizePayload(SimpleTestCase):

    def test_forbidden_keys_refused_at_any_depth(self):
        for key in ("vector", "sample", "template", "embedding", "frame", "image", "template_iso", "probe_vector"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    sanitize_payload({"outer": {"inner": [{key: 1}]}})

    def test_forbidden_key_match_ignores_case(self):
        with self.assertRaises(ValueError):
            sanitize_payload({"Vector": [0.1]})

    def test_bytes_refused(self):
        for value in (b"x", bytearray(b"x"), memoryview(b"x")):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    sanitize_payload({"blob": value})

    def test_long_numeric_list_refused(self):
        with self.assertRaises(ValueError):
            sanitize_payload({"scores": [0.5] * 17})
        self.assertEqual(sanitize_payload({"scores": [0.5] * 16})["scores"], [0.5] * 16)

    def test_numpy_array_is_a_numeric_list(self):
        with self.assertRaises(ValueError):
            sanitize_payload({"x": np.zeros(128)})

    def test_nan_and_infinity_refused(self):
        for value in (float("nan"), float("inf"), float("-inf"), np.float32("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    sanitize_payload({"score": value})

    def test_normalises_uuid_datetime_and_numpy_scalars(self):
        ident = uuid.uuid4()
        when = datetime.datetime(2026, 1, 2, 3, 4, 5, 678901)
        clean = sanitize_payload({
            "id": ident, "when": when, "day": datetime.date(2026, 1, 2),
            "f32": np.float32(0.5), "f64": np.float64(0.25), "i64": np.int64(7), "flag": np.bool_(True),
            "tuple": (1, 2),
        })
        self.assertEqual(clean, {
            "id": str(ident), "when": when.isoformat(), "day": "2026-01-02",
            "f32": 0.5, "f64": 0.25, "i64": 7, "flag": True, "tuple": [1, 2],
        })

    def test_large_integral_float_becomes_int(self):
        self.assertEqual(sanitize_payload({"big": 1e20})["big"], 10 ** 20)
        self.assertIsInstance(sanitize_payload({"big": 1e20})["big"], int)

    def test_negative_zero_becomes_zero(self):
        clean = sanitize_payload({"s": -0.0, "l": [-0.0], "n": np.float32(-0.0), "d": {"x": np.float64(-0.0)}})
        self.assertEqual(clean, {"s": 0.0, "l": [0.0], "n": 0.0, "d": {"x": 0.0}})
        dumped = json.dumps(clean)
        self.assertNotIn("-0", dumped)

    def test_none_payload_is_empty(self):
        self.assertEqual(sanitize_payload(None), {})

    def test_non_dict_payload_refused(self):
        with self.assertRaises(TypeError):
            sanitize_payload([1, 2])


class TestHashDefinition(SimpleTestCase):

    def _event(self, **overrides):
        values = dict(
            id=uuid.UUID("12345678-1234-5678-1234-567812345678"), sequence=1, action=ACTION_VERIFY,
            actor="a", subject_model="individual.Individual", subject_id="s1", modality="face",
            payload={"b": 1, "a": 0.1}, created_at=datetime.datetime(2026, 1, 2, 3, 4, 5, 6),
            prev_hash=GENESIS_HASH,
        )
        values.update(overrides)
        return BiometricAuditEvent(**values)

    def test_canonical_form_is_sorted_and_compact(self):
        self.assertEqual(
            canonical_event(self._event()),
            '{"action":"verify","actor":"a","created_at":"2026-01-02T03:04:05.000006",'
            '"id":"12345678-1234-5678-1234-567812345678","modality":"face","payload":{"a":0.1,"b":1},'
            '"sequence":1,"subject_id":"s1","subject_model":"individual.Individual"}',
        )

    def test_hash_is_sha256_of_prev_hash_then_canonical(self):
        import hashlib

        event = self._event()
        expected = hashlib.sha256((GENESIS_HASH + canonical_event(event)).encode()).hexdigest()
        self.assertEqual(compute_hash(event, GENESIS_HASH), expected)

    def test_aware_time_hashes_as_naive_utc(self):
        naive = self._event(created_at=datetime.datetime(2026, 1, 2, 3, 4, 5))
        aware = self._event(created_at=datetime.datetime(
            2026, 1, 2, 4, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=1)),
        ))
        self.assertEqual(canonical_event(naive), canonical_event(aware))

    def test_negative_zero_hashes_as_its_jsonb_read_back(self):
        written = self._event(payload=sanitize_payload({"score": -0.0}))
        read_back = self._event(payload={"score": 0.0})
        self.assertEqual(compute_hash(written, GENESIS_HASH), compute_hash(read_back, GENESIS_HASH))

    def test_every_covered_field_changes_the_hash(self):
        base = compute_hash(self._event(), GENESIS_HASH)
        for field, value in (
            ("sequence", 2), ("action", ACTION_TEMPLATE_READ), ("actor", "b"), ("subject_model", "x.Y"),
            ("subject_id", "s2"), ("modality", "fingerprint"), ("payload", {"b": 2}),
            ("created_at", datetime.datetime(2026, 1, 2, 3, 4, 5, 7)), ("id", uuid.uuid4()),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(compute_hash(self._event(**{field: value}), GENESIS_HASH), base)
        self.assertNotEqual(compute_hash(self._event(), "1" * 64), base)


class TestDisabledIsInert(AuditConfigMixin, SimpleTestCase):
    """SimpleTestCase refuses database queries, so any query here fails the test."""

    def test_default_config_is_off(self):
        self.assertFalse(BiometricConfig.audit.get("enabled"))
        self.assertFalse(audit_chain.audit_enabled())

    def test_record_event_returns_none_without_queries(self):
        self.assertIsNone(record_event(ACTION_VERIFY, actor="a", payload={"verified": True}))

    def test_only_boolean_true_enables(self):
        for value in (1, "true", "yes", None):
            with self.subTest(value=value):
                BiometricConfig.audit = {"enabled": value}
                self.assertFalse(audit_chain.audit_enabled())
                self.assertIsNone(record_event(ACTION_VERIFY, actor="a"))
        BiometricConfig.audit = None
        self.assertFalse(audit_chain.audit_enabled())

    def test_audited_block_is_nullcontext_when_off(self):
        import contextlib

        self.assertIsInstance(audited_block(), contextlib.nullcontext)
        BiometricConfig.audit = dict(AUDIT_ON)
        self.assertIsInstance(audited_block(), transaction.Atomic)

    def test_unknown_action_raises_before_any_query(self):
        BiometricConfig.audit = dict(AUDIT_ON)
        with self.assertRaises(ValueError):
            record_event("verify.forged", actor="a")

    def test_actions_constant(self):
        self.assertEqual(
            ACTIONS,
            frozenset({
                "template.enrol", "template.enrol_refused", "verify", "identify", "impersonation.suspected",
                "template.consolidate", "template.purge", "template.read", "template.list", "alert.acknowledge",
                "alert.resolve",
            }),
        )


class TestRecordEvent(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)

    def test_disabled_writes_no_row(self):
        BiometricConfig.audit = {"enabled": False, "rules": {}}
        self.assertIsNone(record_event(ACTION_VERIFY, actor="a", payload={"verified": True}))
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)

    def test_first_event_links_to_genesis(self):
        event = record_event(ACTION_VERIFY, actor="a", subject_id="s1", payload={"verified": True})

        self.assertEqual(event.sequence, 1)
        self.assertEqual(event.prev_hash, GENESIS_HASH)
        row = BiometricAuditEvent.objects.get(pk=event.pk)
        self.assertEqual(compute_hash(row, row.prev_hash), row.hash)
        self.assertEqual(row.hash, event.hash)

    def test_events_link_in_sequence(self):
        events = _record(5)

        rows = list(BiometricAuditEvent.objects.order_by("sequence"))
        self.assertEqual([row.sequence for row in rows], [1, 2, 3, 4, 5])
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(current.prev_hash, previous.hash)
        report = verify_chain()
        self.assertTrue(report.ok)
        self.assertEqual(report.checked, 5)
        self.assertEqual(chain_head(), (5, events[-1].hash))

    def test_payload_round_trip_still_verifies(self):
        record_event(
            ACTION_VERIFY, actor="agent-é", subject_id="s1",
            payload={
                "small": 1e-7, "tenth": 0.1, "third": 1 / 3, "big": 1e20, "int": 12,
                "text": "Ngaoundéré — œuvre ✓", "nested": {"a": [1, {"b": None}], "z": {"y": "x"}},
                "none": None, "naive": datetime.datetime(2026, 1, 2, 3, 4, 5, 678901), "verified": False,
            },
        )
        row = BiometricAuditEvent.objects.get()
        self.assertEqual(compute_hash(row, row.prev_hash), row.hash)
        self.assertTrue(verify_chain().ok)

    def test_refused_payload_writes_nothing_and_consumes_no_sequence(self):
        refusals = (
            ({"outer": {"vector": [0.1]}}, ValueError),
            ({"outer": {"sample": "x"}}, ValueError),
            ({"template": "x"}, ValueError),
            ({"a": [{"embedding": 1}]}, ValueError),
            ({"frame": 1}, ValueError),
            ({"raw": b"\x00\x01"}, TypeError),
            ({"values": [0.25] * 17}, ValueError),
            ({"score": float("nan")}, ValueError),
        )
        for payload, error in refusals:
            with self.subTest(payload=payload):
                with self.assertRaises(error):
                    record_event(ACTION_VERIFY, actor="a", payload=payload)
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)

        event = record_event(ACTION_VERIFY, actor="a", payload={"verified": True})
        self.assertEqual(event.sequence, 1)

    def test_unknown_action_raises(self):
        with self.assertRaises(ValueError):
            record_event("template.forged", actor="a")
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)

    def test_takes_the_advisory_lock_on_postgres(self):
        if connection.vendor != "postgresql":
            self.skipTest("advisory lock is Postgres-only")
        with CaptureQueriesContext(connection) as queries:
            record_event(ACTION_VERIFY, actor="a", payload={"verified": True})
        self.assertTrue(any("pg_advisory_xact_lock" in q["sql"] for q in queries.captured_queries))

    def test_rules_are_scheduled_on_commit(self):
        with self.captureOnCommitCallbacks(execute=False) as callbacks:
            record_event(ACTION_VERIFY, actor="a", payload={"verified": True})
        self.assertEqual(len(callbacks), 1)


class TestVerifyChainTamper(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)
        self.events = _record(5)

    def _sql(self, statement, params=None):
        with connection.cursor() as cursor:
            cursor.execute(statement, params or [])

    def _divergence(self):
        report = verify_chain()
        self.assertFalse(report.ok)
        return report.divergence

    def test_intact_chain_is_ok(self):
        report = verify_chain(batch_size=2)
        self.assertTrue(report.ok)
        self.assertEqual((report.head_sequence, report.head_hash), (5, self.events[-1].hash))

    def test_payload_update_is_altered_row(self):
        self._sql("UPDATE biometric_audit_event SET payload = %s::jsonb WHERE sequence = 2", ['{"verified": false}'])
        divergence = self._divergence()
        self.assertEqual((divergence.kind, divergence.sequence), ("altered_row", 2))
        self.assertEqual(verify_chain().checked, 1)

    def test_hash_update_is_altered_row(self):
        self._sql("UPDATE biometric_audit_event SET hash = %s WHERE sequence = 3", ["f" * 64])
        divergence = self._divergence()
        self.assertEqual((divergence.kind, divergence.sequence), ("altered_row", 3))

    def test_prev_hash_update_is_broken_link(self):
        self._sql("UPDATE biometric_audit_event SET prev_hash = %s WHERE sequence = 3", ["e" * 64])
        divergence = self._divergence()
        self.assertEqual((divergence.kind, divergence.sequence), ("broken_link", 3))

    def test_deleted_middle_row_is_missing_event(self):
        self._sql("DELETE FROM biometric_audit_event WHERE sequence = 3")
        divergence = self._divergence()
        self.assertEqual((divergence.kind, divergence.sequence), ("missing_event", 3))

    def test_deleted_first_row_is_missing_event(self):
        self._sql("DELETE FROM biometric_audit_event WHERE sequence = 1")
        divergence = self._divergence()
        self.assertEqual((divergence.kind, divergence.sequence), ("missing_event", 1))

    def test_tail_truncation_passes_the_walk_but_moves_the_head(self):
        recorded_head = chain_head()
        self._sql("DELETE FROM biometric_audit_event WHERE sequence = 5")

        self.assertTrue(verify_chain().ok)
        self.assertNotEqual(chain_head(), recorded_head)
        self.assertEqual(chain_head(), (4, self.events[3].hash))


class TestAppendOnlyGuards(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)
        self.event = record_event(ACTION_VERIFY, actor="a", payload={"verified": True})

    def test_saving_an_existing_event_raises(self):
        row = BiometricAuditEvent.objects.get(pk=self.event.pk)
        row.actor = "someone-else"
        with self.assertRaises(PermissionError):
            row.save()

    def test_queryset_update_and_delete_raise(self):
        with self.assertRaises(PermissionError):
            BiometricAuditEvent.objects.filter(pk=self.event.pk).update(actor="x")
        with self.assertRaises(PermissionError):
            BiometricAuditEvent.objects.filter(pk=self.event.pk).delete()

    def test_instance_delete_raises(self):
        with self.assertRaises(PermissionError):
            BiometricAuditEvent.objects.get(pk=self.event.pk).delete()
        self.assertEqual(BiometricAuditEvent.objects.count(), 1)

    def test_duplicate_prev_hash_is_refused(self):
        forged = BiometricAuditEvent(
            id=uuid.uuid4(), sequence=99, action=ACTION_VERIFY, actor="a", payload={},
            prev_hash=self.event.prev_hash, hash="a" * 64,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                forged.save(force_insert=True)

    def test_duplicate_sequence_is_refused(self):
        forged = BiometricAuditEvent(
            id=uuid.uuid4(), sequence=self.event.sequence, action=ACTION_VERIFY, actor="a", payload={},
            prev_hash="b" * 64, hash="a" * 64,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                forged.save(force_insert=True)


class TestConcurrentAppends(AuditConfigMixin, TransactionTestCase):
    """Real commits from four connections; the flush touches the biometric tables only."""

    available_apps = ["biometric"]

    def test_forty_appends_from_four_threads_form_one_chain(self):
        BiometricConfig.audit = dict(AUDIT_ON)
        errors = []

        def worker(thread_index):
            try:
                for index in range(10):
                    record_event(
                        ACTION_VERIFY, actor=f"thread-{thread_index}", subject_id=f"s{thread_index}-{index}",
                        payload={"verified": True},
                    )
            except Exception as exc:  # collected and asserted below
                errors.append(exc)
            finally:
                connection.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        sequences = list(BiometricAuditEvent.objects.order_by("sequence").values_list("sequence", flat=True))
        self.assertEqual(sequences, list(range(1, 41)))
        report = verify_chain()
        self.assertTrue(report.ok)
        self.assertEqual(report.checked, 40)


class TestVerifyCommand(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)

    def _run(self, *args):
        out = StringIO()
        call_command("biometric_audit_verify", *args, stdout=out)
        return out.getvalue()

    def test_empty_chain_is_intact(self):
        output = self._run()
        self.assertIn("biometric audit chain intact over 0 event(s)", output)
        self.assertIn(f"head sequence 0 hash {GENESIS_HASH}", output)

    def test_intact_chain_prints_the_head(self):
        events = _record(3)
        output = self._run("--batch-size", "2")
        self.assertIn("intact over 3 event(s)", output)
        self.assertIn(f"head sequence 3 hash {events[-1].hash}", output)

    def test_matching_expected_head_and_sequence_pass(self):
        events = _record(3)
        output = self._run("--expected-head", events[-1].hash, "--expected-sequence", "3")
        self.assertIn("intact over 3 event(s)", output)

    def test_anchor_holds_after_the_chain_grows(self):
        events = _record(3)
        _record(2)
        output = self._run("--expected-head", events[-1].hash, "--expected-sequence", "3")
        self.assertIn("intact over 5 event(s)", output)
        self.assertIn(f"head sequence 5 hash {chain_head()[1]}", output)

    def test_anchor_hash_alone_or_sequence_alone_holds_after_growth(self):
        events = _record(3)
        _record(1)
        self.assertIn("intact over 4 event(s)", self._run("--expected-head", events[-1].hash))
        self.assertIn("intact over 4 event(s)", self._run("--expected-sequence", "3"))

    def test_empty_chain_anchor_holds_on_any_chain(self):
        _record(2)
        output = self._run("--expected-head", GENESIS_HASH, "--expected-sequence", "0")
        self.assertIn("intact over 2 event(s)", output)
        self.assertIn("intact over 2 event(s)", self._run("--expected-head", GENESIS_HASH))

    def test_anchor_at_sequence_zero_needs_the_genesis_hash(self):
        events = _record(1)
        with self.assertRaises(CommandError) as ctx:
            self._run("--expected-head", events[0].hash, "--expected-sequence", "0")
        self.assertIn("truncated or rewritten", str(ctx.exception))

    def test_negative_anchor_sequence_raises(self):
        with self.assertRaises(CommandError):
            self._run("--expected-sequence", "-1")

    def test_anchor_with_the_hash_of_another_sequence_raises(self):
        events = _record(3)
        with self.assertRaises(CommandError) as ctx:
            self._run("--expected-head", events[1].hash, "--expected-sequence", "3")
        self.assertIn("event at sequence 3", str(ctx.exception))

    def test_truncated_then_regrown_tail_caught_by_anchor(self):
        events = _record(3)
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM biometric_audit_event WHERE sequence = 3")
        _record(2)
        self.assertIn("intact over 4 event(s)", self._run())
        with self.assertRaises(CommandError) as ctx:
            self._run("--expected-head", events[-1].hash, "--expected-sequence", "3")
        self.assertIn("truncated or rewritten", str(ctx.exception))
        with self.assertRaises(CommandError):
            self._run("--expected-head", events[-1].hash)

    def test_tampered_chain_raises(self):
        _record(3)
        with connection.cursor() as cursor:
            cursor.execute("UPDATE biometric_audit_event SET actor = 'x' WHERE sequence = 2")
        with self.assertRaises(CommandError) as ctx:
            self._run()
        message = str(ctx.exception)
        self.assertIn("diverges after 1 event(s)", message)
        self.assertIn("altered_row at sequence 2", message)

    def test_mismatched_expected_head_raises(self):
        _record(2)
        with self.assertRaises(CommandError) as ctx:
            self._run("--expected-head", "0" * 63 + "1")
        self.assertIn("truncated or rewritten", str(ctx.exception))

    def test_mismatched_expected_sequence_raises(self):
        _record(2)
        with self.assertRaises(CommandError) as ctx:
            self._run("--expected-sequence", "3")
        self.assertIn("truncated or rewritten", str(ctx.exception))

    def test_tail_truncation_caught_by_expected_head(self):
        events = _record(3)
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM biometric_audit_event WHERE sequence = 3")
        self.assertIn("intact over 2 event(s)", self._run())
        with self.assertRaises(CommandError):
            self._run("--expected-head", events[-1].hash, "--expected-sequence", "3")

