"""
Alert rules over the audit chain (docs/wb-biometric-dedup-seam.md §6.10):
FAILED_VERIFICATIONS, IMPERSONATION_SUSPECTED and ACCESS_BURST, their
configuration, isolation of a failing rule, the one-open-alert constraint and
the triage transitions. Rules run through captureOnCommitCallbacks(execute=True);
time is set by patching biometric.audit_chain.timezone.now.
"""

import datetime
from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import IntegrityError, connection, transaction
from django.test import SimpleTestCase, TestCase

from biometric import audit_rules
from biometric.apps import _SETTINGS_KEY_MAP, DEFAULT_CFG, BiometricConfig
from biometric.audit_chain import (
    ACTION_ALERT_ACK,
    ACTION_ALERT_RESOLVE,
    ACTION_IDENTIFY,
    ACTION_IMPERSONATION,
    ACTION_TEMPLATE_READ,
    ACTION_VERIFY,
    record_event,
)
from biometric.audit_rules import (
    ACCESS_BURST,
    DEFAULT_RULES,
    FAILED_VERIFICATIONS,
    IMPERSONATION_SUSPECTED,
    acknowledge_alert,
    evaluate_event,
    raise_alert,
    resolve_alert,
    rule_params,
    validate_audit_config,
)
from biometric.models import BiometricAlert, BiometricAuditEvent
from biometric.tests.test_audit_chain import AuditConfigMixin

SUBJECT_MODEL = "individual.Individual"
T0 = datetime.datetime(2026, 9, 1, 12, 0, 0)


class _RulesTestCase(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        self.now = T0
        patcher = patch("biometric.audit_chain.timezone.now", side_effect=lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _rules(self, **rules):
        BiometricConfig.audit = {"enabled": True, "rules": rules}

    def _at(self, minutes):
        self.now = T0 + datetime.timedelta(minutes=minutes)

    def _record(self, action, **kwargs):
        with self.captureOnCommitCallbacks(execute=True):
            return record_event(action, **kwargs)

    def _verify(self, subject_id="s1", verified=False, modality="face", actor="agent", device_id=""):
        return self._record(
            ACTION_VERIFY, actor=actor, subject_model=SUBJECT_MODEL, subject_id=subject_id, modality=modality,
            payload={"verified": verified, "score": 0.1, "threshold": 0.3, "device_id": device_id},
        )

    def _impersonation(self, claimed="alice", matched="bob", matched_model=SUBJECT_MODEL):
        return self._record(
            ACTION_IMPERSONATION, actor="agent", subject_model=SUBJECT_MODEL, subject_id=claimed, modality="face",
            payload={
                "verification_id": "v-1", "matched_subject_model": matched_model, "matched_subject_id": matched,
                "matched_template_id": "t-9", "matched_score": 0.9, "claimed_score": 0.2,
                "threshold": 0.62, "margin": None,
            },
        )

    def _read(self, actor, subject_id="s1"):
        return self._record(
            ACTION_TEMPLATE_READ, actor=actor, subject_model=SUBJECT_MODEL, subject_id=subject_id,
            payload={"purpose": "read", "template_ids": []},
        )


class TestFailedVerifications(_RulesTestCase):

    def test_threshold_occurrences_and_last_seen(self):
        self._verify(device_id="tab-1")
        self._at(1)
        self._verify(actor="agent-2", device_id="tab-2")
        self.assertEqual(BiometricAlert.objects.count(), 0)

        self._at(2)
        third = self._verify()
        alert = BiometricAlert.objects.get()
        self.assertEqual((alert.rule_kind, alert.state, alert.severity), (FAILED_VERIFICATIONS, "NEW", "MEDIUM"))
        self.assertEqual(alert.occurrences, 1)
        self.assertEqual((alert.subject_model, alert.subject_id), (SUBJECT_MODEL, "s1"))
        self.assertEqual(alert.trigger_event_id, third.id)
        self.assertEqual(alert.dedupe_key, f"failed_verifications:{SUBJECT_MODEL}:s1:face")
        self.assertEqual(alert.detail, {
            "attempts": 3, "threshold": 3, "window_minutes": 60, "modality": "face",
            "actors": ["agent", "agent-2"], "device_ids": ["tab-1", "tab-2"],
        })
        first_seen = alert.last_seen_at

        self._at(5)
        fourth = self._verify()
        alert.refresh_from_db()
        self.assertEqual(BiometricAlert.objects.count(), 1)
        self.assertEqual(alert.occurrences, 2)
        self.assertGreater(alert.last_seen_at, first_seen)
        self.assertEqual(alert.trigger_event_id, fourth.id)
        self.assertEqual(alert.detail["attempts"], 4)

    def test_window_success_and_other_subjects_are_not_counted(self):
        self._verify()
        self._at(30)
        self._verify()
        self._at(91)
        self._verify()
        self.assertEqual(BiometricAlert.objects.count(), 0)

        self._at(92)
        self._verify(verified=True)
        self._verify(subject_id="s2")
        self._verify(subject_id="s2")
        self.assertEqual(BiometricAlert.objects.count(), 0)

    def test_per_modality_keeps_modalities_apart(self):
        self._verify(modality="face")
        self._verify(modality="face")
        self._verify(modality="fingerprint")
        self.assertEqual(BiometricAlert.objects.count(), 0)

    def test_per_modality_off_merges_modalities(self):
        self._rules(FAILED_VERIFICATIONS={"per_modality": False})
        self._verify(modality="face")
        self._verify(modality="face")
        self._verify(modality="fingerprint")
        alert = BiometricAlert.objects.get()
        self.assertEqual(alert.dedupe_key, f"failed_verifications:{SUBJECT_MODEL}:s1")
        self.assertIsNone(alert.detail["modality"])

    def test_acknowledged_alert_is_bumped_and_resolved_alert_reopens(self):
        for _ in range(3):
            self._verify()
        alert = BiometricAlert.objects.get()
        acknowledge_alert(alert.id, actor="supervisor")

        self._verify()
        alert.refresh_from_db()
        self.assertEqual((alert.state, alert.occurrences), ("ACKNOWLEDGED", 2))

        resolve_alert(alert.id, actor="supervisor", note="checked")
        self._verify()
        self.assertEqual(BiometricAlert.objects.count(), 2)
        reopened = BiometricAlert.objects.exclude(pk=alert.pk).get()
        self.assertEqual((reopened.state, reopened.occurrences), ("NEW", 1))


class TestImpersonationSuspected(_RulesTestCase):

    def test_same_pair_bumps_other_match_opens_another(self):
        self._impersonation()
        alert = BiometricAlert.objects.get()
        self.assertEqual((alert.rule_kind, alert.severity), (IMPERSONATION_SUSPECTED, "HIGH"))
        self.assertEqual((alert.subject_model, alert.subject_id), (SUBJECT_MODEL, "alice"))
        self.assertEqual(alert.detail, {
            "verification_id": "v-1", "modality": "face", "matched_subject_model": SUBJECT_MODEL,
            "matched_subject_id": "bob", "matched_template_id": "t-9", "matched_score": 0.9,
            "claimed_score": 0.2, "threshold": 0.62, "margin": None,
        })

        self._impersonation()
        alert.refresh_from_db()
        self.assertEqual(alert.occurrences, 2)

        self._impersonation(matched="carol")
        self.assertEqual(BiometricAlert.objects.count(), 2)

    def test_same_id_under_another_model_is_a_separate_alert(self):
        self._impersonation(matched="bob")
        self._impersonation(matched="bob", matched_model="other.Subject")
        self.assertEqual(BiometricAlert.objects.count(), 2)


class TestAccessBurst(_RulesTestCase):

    def setUp(self):
        super().setUp()
        self._rules(ACCESS_BURST={"max_events": 3})

    def test_per_actor_count(self):
        self._read("A", "s1")
        self._read("A", "s2")
        self._read("B", "s1")
        self._read("B", "s1")
        self.assertEqual(BiometricAlert.objects.count(), 0)

        self._record(ACTION_IDENTIFY, actor="A", payload={"matches": []})
        alert = BiometricAlert.objects.get()
        self.assertEqual((alert.rule_kind, alert.dedupe_key, alert.subject_id), (ACCESS_BURST, "access_burst:A", ""))
        self.assertEqual(alert.detail["events"], 3)
        self.assertEqual(alert.detail["distinct_subjects"], 2)
        self.assertEqual(alert.detail["actor"], "A")

    def test_blank_actor_never_alerts(self):
        for _ in range(4):
            self._read("")
        self.assertEqual(BiometricAlert.objects.count(), 0)

    def test_unwatched_actions_are_not_counted(self):
        for _ in range(4):
            self._verify(actor="A", verified=True)
        self.assertEqual(BiometricAlert.objects.count(), 0)


class TestRuleConfiguration(_RulesTestCase):

    def test_disabled_rule_raises_nothing(self):
        self._rules(FAILED_VERIFICATIONS={"enabled": False})
        for _ in range(4):
            self._verify()
        self.assertEqual(BiometricAlert.objects.count(), 0)

    def test_override_changes_only_its_key(self):
        self._rules(FAILED_VERIFICATIONS={"threshold": 2})
        self.assertEqual(rule_params(FAILED_VERIFICATIONS), {**DEFAULT_RULES[FAILED_VERIFICATIONS], "threshold": 2})
        self.assertEqual(rule_params(ACCESS_BURST), DEFAULT_RULES[ACCESS_BURST])

        self._verify()
        self._verify()
        self.assertEqual(BiometricAlert.objects.get().detail["threshold"], 2)

    def test_invalid_rule_is_isolated(self):
        self._rules(FAILED_VERIFICATIONS={"threshold": 0})
        with self.assertRaises(ImproperlyConfigured):
            rule_params(FAILED_VERIFICATIONS)
        with self.assertLogs("biometric.audit_rules", level="ERROR"):
            self._impersonation()
        self.assertEqual(BiometricAlert.objects.get().rule_kind, IMPERSONATION_SUSPECTED)


class TestValidateAuditConfig(SimpleTestCase):

    def test_valid_configs(self):
        self.assertEqual(validate_audit_config(None), [])
        self.assertEqual(validate_audit_config({}), [])
        self.assertEqual(validate_audit_config(DEFAULT_CFG["audit"]), [])
        self.assertEqual(validate_audit_config({
            "enabled": True,
            "rules": {
                FAILED_VERIFICATIONS: {"threshold": 5, "severity": "HIGH", "per_modality": False},
                IMPERSONATION_SUSPECTED: {"enabled": False},
                ACCESS_BURST: {"max_events": 50, "actions": ["template.read"]},
            },
        }), [])

    def test_every_invalid_case_is_reported(self):
        cases = {
            "unknown kind": {"rules": {"FAILED_LOGINS": {}}},
            "unknown param": {"rules": {FAILED_VERIFICATIONS: {"limit": 3}}},
            "zero threshold": {"rules": {FAILED_VERIFICATIONS: {"threshold": 0}}},
            "bool threshold": {"rules": {FAILED_VERIFICATIONS: {"threshold": True}}},
            "string window": {"rules": {ACCESS_BURST: {"window_minutes": "60"}}},
            "unknown severity": {"rules": {IMPERSONATION_SUSPECTED: {"severity": "CRITICAL"}}},
            "alert action watched": {"rules": {ACCESS_BURST: {"actions": ["alert.acknowledge"]}}},
            "unknown action": {"rules": {ACCESS_BURST: {"actions": ["template.export"]}}},
            "empty actions": {"rules": {ACCESS_BURST: {"actions": []}}},
            "non-bool enabled": {"enabled": "yes"},
            "non-bool rule enabled": {"rules": {ACCESS_BURST: {"enabled": 1}}},
            "unknown top-level key": {"enabled": False, "max_page_size": 500},
            "rules not a dict": {"rules": ["FAILED_VERIFICATIONS"]},
            "rule not a dict": {"rules": {FAILED_VERIFICATIONS: 3}},
            "config not a dict": ["enabled"],
        }
        for name, cfg in cases.items():
            with self.subTest(name):
                errors = validate_audit_config(cfg)
                self.assertTrue(errors)
                self.assertTrue(all(isinstance(e, str) for e in errors))

    def test_ready_check_logs_without_raising(self):
        saved = BiometricConfig.audit
        try:
            BiometricConfig.audit = {"enabled": "yes", "rules": {"NOPE": {}}}
            with self.assertLogs("biometric.apps", level="ERROR") as logs:
                BiometricConfig._check_audit_config()
            self.assertEqual(len(logs.records), 2)
            self.assertTrue(all("BIOMETRIC['AUDIT']" in r.getMessage() for r in logs.records))
        finally:
            BiometricConfig.audit = saved


class TestAuditConfigDefaults(SimpleTestCase):

    def test_defaults(self):
        self.assertEqual(DEFAULT_CFG["audit"], {"enabled": False, "rules": {}})
        self.assertIs(DEFAULT_CFG["audit"]["enabled"], False)
        self.assertEqual(DEFAULT_CFG["gql_biometric_audit_perms"], ["174005"])
        self.assertEqual(DEFAULT_CFG["gql_biometric_alert_perms"], ["174006"])
        self.assertEqual(BiometricConfig.gql_biometric_audit_perms, ["174005"])
        self.assertEqual(BiometricConfig.gql_biometric_alert_perms, ["174006"])

    def test_settings_key_map(self):
        self.assertEqual(_SETTINGS_KEY_MAP["AUDIT"], "audit")
        self.assertEqual(_SETTINGS_KEY_MAP["GQL_BIOMETRIC_AUDIT_PERMS"], "gql_biometric_audit_perms")
        self.assertEqual(_SETTINGS_KEY_MAP["GQL_BIOMETRIC_ALERT_PERMS"], "gql_biometric_alert_perms")

    def test_existing_keys_unchanged(self):
        self.assertEqual(DEFAULT_CFG["gql_biometric_enrol_perms"], ["174001"])
        self.assertEqual(DEFAULT_CFG["gql_biometric_verify_perms"], ["174002"])
        self.assertEqual(DEFAULT_CFG["gql_biometric_identify_perms"], ["174003"])
        self.assertEqual(DEFAULT_CFG["gql_biometric_read_perms"], ["174004"])
        self.assertEqual(DEFAULT_CFG["subject_model"], "individual.Individual")
        self.assertIs(DEFAULT_CFG["impersonation_probe"]["enabled"], False)
        self.assertEqual(DEFAULT_CFG["risk_profiles"], {})
        self.assertEqual(DEFAULT_CFG["quality"]["mode"], "advisory")


class TestIsolationAndSkips(_RulesTestCase):

    def test_raising_check_is_logged_and_the_others_still_run(self):
        def broken(event, params):
            raise RuntimeError("rule bug")

        with patch.dict(audit_rules.CHECKS, {FAILED_VERIFICATIONS: broken}):
            with self.assertLogs("biometric.audit_rules", level="ERROR"):
                event = self._impersonation()
        self.assertTrue(BiometricAuditEvent.objects.filter(pk=event.pk).exists())
        self.assertEqual(BiometricAlert.objects.get().rule_kind, IMPERSONATION_SUSPECTED)

    def test_database_error_in_one_check_leaves_the_connection_usable(self):
        def broken(event, params):
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM biometric_no_such_table")

        with patch.dict(audit_rules.CHECKS, {FAILED_VERIFICATIONS: broken}):
            with self.assertLogs("biometric.audit_rules", level="ERROR"):
                self._impersonation()
        self.assertEqual(BiometricAlert.objects.count(), 1)

    def test_alert_actions_are_never_evaluated(self):
        self._rules(ACCESS_BURST={"max_events": 1})
        ack = self._record(ACTION_ALERT_ACK, actor="A", payload={"alert_id": "x", "rule_kind": "R", "note": ""})
        resolve = self._record(ACTION_ALERT_RESOLVE, actor="A", payload={"alert_id": "x", "rule_kind": "R", "note": ""})
        self.assertEqual(evaluate_event(ack.id), [])
        self.assertEqual(evaluate_event(resolve.id), [])
        self.assertEqual(BiometricAlert.objects.count(), 0)

    def test_missing_event_returns_empty(self):
        import uuid

        self.assertEqual(evaluate_event(uuid.uuid4()), [])

    def test_nothing_without_the_commit(self):
        for _ in range(3):
            record_event(
                ACTION_VERIFY, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1", modality="face",
                payload={"verified": False},
            )
        self.assertEqual(BiometricAuditEvent.objects.count(), 3)
        self.assertEqual(BiometricAlert.objects.count(), 0)


class TestAlertModel(_RulesTestCase):

    def setUp(self):
        super().setUp()
        self.event = record_event(ACTION_VERIFY, actor="a", subject_model=SUBJECT_MODEL, subject_id="s1",
                                  payload={"verified": False})

    def _alert(self, **kwargs):
        values = dict(rule_kind="R", severity="LOW", title="t", dedupe_key="k", trigger_event=self.event)
        values.update(kwargs)
        return BiometricAlert.objects.create(**values)

    def test_one_open_alert_per_key(self):
        self._alert()
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                self._alert()
        self._alert(dedupe_key="other")
        resolved = self._alert(dedupe_key="closed", state="RESOLVED")
        self._alert(dedupe_key="closed")
        self.assertEqual(resolved.state, "RESOLVED")

    def test_raise_alert_bumps_the_existing_open_alert(self):
        existing = self._alert(state="ACKNOWLEDGED")
        self._at(10)
        bumped = raise_alert(
            rule_kind="R", dedupe_key="k", severity="HIGH", title="again", detail={"n": 2}, event=self.event,
        )
        self.assertEqual(bumped.pk, existing.pk)
        self.assertEqual(BiometricAlert.objects.count(), 1)
        existing.refresh_from_db()
        self.assertEqual((existing.occurrences, existing.detail, existing.state), (2, {"n": 2}, "ACKNOWLEDGED"))
        self.assertEqual(existing.last_seen_at, T0 + datetime.timedelta(minutes=10))

    def test_raise_alert_recovers_from_a_concurrent_creator(self):
        existing = self._alert()
        real_open = BiometricAlert.objects.select_for_update

        calls = []

        def first_miss():
            qs = real_open()
            calls.append(1)
            return qs.none() if len(calls) == 1 else qs

        with patch.object(BiometricAlert.objects, "select_for_update", side_effect=first_miss):
            bumped = raise_alert(
                rule_kind="R", dedupe_key="k", severity="HIGH", title="t", detail={}, event=self.event,
            )
        self.assertEqual(bumped.pk, existing.pk)
        existing.refresh_from_db()
        self.assertEqual(existing.occurrences, 2)

    def test_acknowledge_then_resolve(self):
        alert = self._alert(subject_model=SUBJECT_MODEL, subject_id="s1")
        self._at(3)
        with self.captureOnCommitCallbacks(execute=True):
            acked = acknowledge_alert(alert.id, actor="supervisor")
        self.assertEqual((acked.state, acked.acknowledged_by), ("ACKNOWLEDGED", "supervisor"))
        self.assertIsNotNone(acked.acknowledged_at)
        event = BiometricAuditEvent.objects.get(action=ACTION_ALERT_ACK)
        self.assertEqual((event.actor, event.subject_id), ("supervisor", "s1"))
        self.assertEqual(event.payload, {"alert_id": str(alert.id), "rule_kind": "R", "note": ""})

        with self.assertRaises(ValidationError):
            acknowledge_alert(alert.id, actor="supervisor")

        resolved = resolve_alert(alert.id, actor="lead", note="false positive")
        self.assertEqual((resolved.state, resolved.resolved_by, resolved.resolution_note),
                         ("RESOLVED", "lead", "false positive"))
        self.assertIsNotNone(resolved.resolved_at)
        event = BiometricAuditEvent.objects.get(action=ACTION_ALERT_RESOLVE)
        self.assertEqual(event.payload["note"], "false positive")

        with self.assertRaises(ValidationError):
            resolve_alert(alert.id, actor="lead")

    def test_resolve_from_new(self):
        alert = self._alert()
        resolved = resolve_alert(alert.id, actor="lead")
        self.assertEqual(resolved.state, "RESOLVED")
        self.assertEqual(resolved.acknowledged_by, "")

    def test_invalid_transition_records_no_event(self):
        alert = self._alert(state="RESOLVED")
        before = BiometricAuditEvent.objects.count()
        with self.assertRaises(ValidationError):
            acknowledge_alert(alert.id, actor="x")
        self.assertEqual(BiometricAuditEvent.objects.count(), before)
