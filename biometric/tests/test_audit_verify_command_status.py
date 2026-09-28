"""
The biometric_audit_verify command stores its chain walk through
audit_chain.store_chain_check(), the service verifyBiometricAuditChain uses,
so a scheduled run updates biometricAuditChainStatus
(docs/wb-biometric-dedup-seam.md §6.12). Exit codes are unchanged.
"""

from io import StringIO

from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test import TestCase

from biometric.apps import BiometricConfig
from biometric.audit_chain import GENESIS_HASH, chain_head
from biometric.models import BiometricAuditChainCheck, BiometricAuditEvent
from biometric.tests.test_admin_schema import AUDIT_PERMS, STATUS_QUERY, _AdminConfigMixin, _execute, _User
from biometric.tests.test_audit_chain import AUDIT_ON, AuditConfigMixin, _record

COMMAND_ACTOR = "biometric_audit_verify"


class TestCommandStoresTheCheck(_AdminConfigMixin, AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)

    def _run(self, *args):
        out = StringIO()
        call_command("biometric_audit_verify", *args, stdout=out)
        return out.getvalue()

    def _status(self):
        result = _execute(STATUS_QUERY, _User(perms=AUDIT_PERMS))
        self.assertIsNone(result.errors, result.errors)
        return result.data["biometricAuditChainStatus"]

    def test_an_intact_run_is_the_status(self):
        events = _record(3)

        output = self._run()

        self.assertEqual(BiometricAuditChainCheck.objects.count(), 1)
        self.assertIn(f"head sequence 3 hash {events[-1].hash}", output)
        status = self._status()
        self.assertEqual(
            (status["ok"], status["checkedBy"], status["checked"], status["headSequence"], status["headHash"]),
            (True, COMMAND_ACTOR, 3, 3, events[-1].hash),
        )
        self.assertEqual((status["divergenceKind"], status["divergenceSequence"]), ("", None))
        self.assertEqual(BiometricAuditEvent.objects.count(), 3)

    def test_a_diverging_run_is_stored_then_fails(self):
        _record(3)
        with connection.cursor() as cursor:
            cursor.execute("UPDATE biometric_audit_event SET actor = 'x' WHERE sequence = 2")

        with self.assertRaises(CommandError) as ctx:
            self._run()

        self.assertIn("altered_row at sequence 2", str(ctx.exception))
        self.assertEqual(BiometricAuditChainCheck.objects.count(), 1)
        status = self._status()
        self.assertFalse(status["ok"])
        self.assertEqual((status["divergenceKind"], status["divergenceSequence"]), ("altered_row", 2))
        self.assertEqual((status["checked"], status["headSequence"]), (1, 3))
        self.assertEqual(status["checkedBy"], COMMAND_ACTOR)

    def test_an_anchor_mismatch_stores_the_intact_walk_and_fails(self):
        events = _record(3)
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM biometric_audit_event WHERE sequence = 3")

        with self.assertRaises(CommandError) as ctx:
            self._run("--expected-head", events[-1].hash, "--expected-sequence", "3")

        self.assertIn("truncated or rewritten", str(ctx.exception))
        self.assertEqual(BiometricAuditChainCheck.objects.count(), 1)
        check = BiometricAuditChainCheck.objects.get()
        self.assertEqual((check.ok, check.checked, check.head_sequence), (True, 2, 2))
        self.assertEqual(check.head_hash, chain_head()[1])

    def test_each_run_stores_a_row_and_the_latest_is_the_status(self):
        _record(1)
        self._run()
        _record(1)
        self._run("--batch-size", "1")

        self.assertEqual(BiometricAuditChainCheck.objects.count(), 2)
        self.assertEqual(self._status()["headSequence"], 2)

    def test_an_empty_chain_run_is_stored(self):
        self._run()

        status = self._status()
        self.assertIsNotNone(status)
        self.assertEqual((status["ok"], status["checked"], status["headSequence"]), (True, 0, 0))
        self.assertEqual(status["headHash"], GENESIS_HASH)
