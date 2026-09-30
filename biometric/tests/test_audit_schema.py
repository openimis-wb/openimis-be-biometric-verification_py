"""
GraphQL surface of the audit chain (docs/wb-biometric-dedup-seam.md §6.10):
biometricAuditEvents and biometricAlerts connections, the triage mutations,
identity stripping, and the audit hooks on identifyBiometric / biometricTemplates.
"""

import base64
import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import graphene
from django.core.exceptions import PermissionDenied, ValidationError
from django.test import SimpleTestCase, TestCase

from biometric import audit_rules
from biometric.apps import BiometricConfig
from biometric.audit_chain import (
    ACTION_ALERT_ACK,
    ACTION_ENROL,
    ACTION_IDENTIFY,
    ACTION_IMPERSONATION,
    ACTION_TEMPLATE_LIST,
    ACTION_TEMPLATE_READ,
    ACTION_VERIFY,
    record_event,
)
from biometric.models import BiometricAlert, BiometricAuditEvent, BiometricTemplate
from biometric.schema import BiometricAlertGQLType, BiometricAuditEventGQLType, Mutation, Query
from biometric.tests.test_audit_chain import AuditConfigMixin

SUBJECT_MODEL = "individual.Individual"
T0 = datetime.datetime(2026, 9, 1, 12, 0, 0)
AUDIT_PERMS = ["174005"]
ALERT_PERMS = ["174006"]
IDENTIFY_PERMS = ["174003"]


class _User:
    """has_perms() answers from a fixed set; enough for the resolvers and core's connection field."""

    def __init__(self, perms=(), username="auditor", anonymous=False):
        self.perms = set(perms)
        self.username = username
        self.is_anonymous = anonymous
        self.is_authenticated = not anonymous
        self.id = None if anonymous else 1

    def has_perms(self, perms):
        # core.models.User.has_perms: an empty list passes, otherwise any one listed right is enough.
        return not perms or any(p in self.perms for p in perms)


def _execute(query, user):
    schema = graphene.Schema(query=Query, mutation=Mutation)
    return schema.execute(query, context_value=SimpleNamespace(user=user, headers={}))


def _original(result):
    return [getattr(e, "original_error", e) for e in (result.errors or [])]


EVENTS_QUERY = """
query {
  biometricAuditEvents(%s) {
    totalCount
    pageInfo { hasNextPage hasPreviousPage }
    edges { node { sequence action actor subjectModel subjectId modality payload createdAt prevHash hash } }
  }
}
"""

ALERTS_QUERY = """
query {
  biometricAlerts(%s) {
    totalCount
    edges { node { ruleKind severity state title detail subjectId occurrences triggerEventId } }
  }
}
"""


class TestAuditQueriesDenied(AuditConfigMixin, SimpleTestCase):

    def test_anonymous_and_missing_perms_are_denied(self):
        users = {
            "anonymous": _User(anonymous=True),
            "no audit right": _User(perms=["174004", "174003", "174006"]),
        }
        for name, user in users.items():
            for query in (EVENTS_QUERY % "first: 5", ALERTS_QUERY % "first: 5"):
                with self.subTest(name, query=query[:40]):
                    result = _execute(query, user)
                    self.assertTrue(result.errors)
                    self.assertIsInstance(_original(result)[0], PermissionDenied)

    def test_resolvers_check_the_audit_right(self):
        info = MagicMock()
        info.context.user = _User(perms=[])
        with self.assertRaises(PermissionDenied):
            Query.resolve_biometric_audit_events(None, info)
        with self.assertRaises(PermissionDenied):
            Query.resolve_biometric_alerts(None, info)


class TestAuditTypes(SimpleTestCase):

    def test_event_type_exposes_no_biometric_material(self):
        fields = set(BiometricAuditEventGQLType._meta.fields)
        self.assertEqual(fields, {
            "id", "sequence", "action", "actor", "subject_model", "subject_id", "modality",
            "payload", "created_at", "prev_hash", "hash",
        })
        self.assertFalse(fields & {"vector", "template", "template_iso", "sample", "embedding"})

    def test_alert_type_hides_the_dedupe_key(self):
        fields = set(BiometricAlertGQLType._meta.fields)
        self.assertNotIn("dedupe_key", fields)
        self.assertNotIn("trigger_event", fields)
        self.assertIn("trigger_event_id", fields)


class _AuditSchemaTestCase(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        BiometricConfig.gql_biometric_audit_perms = list(AUDIT_PERMS)
        BiometricConfig.gql_biometric_alert_perms = list(ALERT_PERMS)
        self.now = T0
        patcher = patch("biometric.audit_chain.timezone.now", side_effect=lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _at(self, minutes):
        self.now = T0 + datetime.timedelta(minutes=minutes)


class TestAuditEventsConnection(_AuditSchemaTestCase):

    def setUp(self):
        super().setUp()
        self._at(0)
        record_event(ACTION_ENROL, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1", modality="face",
                     payload={"template_id": "t1"})
        self._at(5)
        record_event(ACTION_VERIFY, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1", modality="face",
                     payload={"verified": True})
        self._at(10)
        record_event(ACTION_TEMPLATE_READ, actor="reader", subject_model=SUBJECT_MODEL, subject_id="s2",
                     payload={"purpose": "read", "template_ids": []})
        self._at(15)
        record_event(ACTION_IDENTIFY, actor="investigator", modality="face", payload={
            "top_k": 1, "probe": "sample",
            "matches": [{"subject_model": SUBJECT_MODEL, "subject_id": "s9", "template_id": "t9", "score": 0.8}],
        })
        self._at(20)
        record_event(ACTION_IMPERSONATION, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1",
                     modality="face", payload={
                         "verification_id": "v1", "matched_subject_model": SUBJECT_MODEL, "matched_subject_id": "s9",
                         "matched_template_id": "t9", "matched_score": 0.8, "claimed_score": 0.1,
                         "threshold": 0.62, "margin": None,
                     })
        self.auditor = _User(perms=AUDIT_PERMS)

    def _nodes(self, args, user=None):
        result = _execute(EVENTS_QUERY % args, user or self.auditor)
        self.assertIsNone(result.errors, result.errors)
        return result.data["biometricAuditEvents"]

    def test_newest_first_with_count_and_page_info(self):
        data = self._nodes("first: 2")
        self.assertEqual(data["totalCount"], 5)
        self.assertTrue(data["pageInfo"]["hasNextPage"])
        self.assertEqual([e["node"]["sequence"] for e in data["edges"]], [5, 4])

    def test_filters(self):
        cases = {
            'action_Startswith: "template."': [3, 1],
            'subjectId: "s1"': [5, 2, 1],
            'actor: "agent"': [5, 2, 1],
            'createdAt_Gte: "2026-09-01T12:10:00"': [5, 4, 3],
            'createdAt_Lte: "2026-09-01T12:05:00"': [2, 1],
            "sequence_Lt: 3": [2, 1],
            'action: "verify"': [2],
            'modality: "face", subjectModel: "%s"' % SUBJECT_MODEL: [5, 2, 1],
        }
        for args, expected in cases.items():
            with self.subTest(args):
                data = self._nodes(args + ", first: 10")
                self.assertEqual([e["node"]["sequence"] for e in data["edges"]], expected)

    def test_page_size_is_capped(self):
        result = _execute(EVENTS_QUERY % "first: 101", self.auditor)
        self.assertTrue(result.errors)

    def test_cross_subject_identity_needs_the_identify_right(self):
        without = {e["node"]["action"]: e["node"]["payload"] for e in self._nodes("first: 10")["edges"]}
        import json

        identify_payload = json.loads(without[ACTION_IDENTIFY])
        self.assertEqual(identify_payload["matches"], [{"score": 0.8}])
        suspicion = json.loads(without[ACTION_IMPERSONATION])
        self.assertNotIn("matched_subject_id", suspicion)
        self.assertNotIn("matched_subject_model", suspicion)
        self.assertNotIn("matched_template_id", suspicion)
        self.assertEqual(suspicion["matched_score"], 0.8)

        full = {
            e["node"]["action"]: json.loads(e["node"]["payload"])
            for e in self._nodes("first: 10", _User(perms=AUDIT_PERMS + IDENTIFY_PERMS))["edges"]
        }
        self.assertEqual(full[ACTION_IDENTIFY]["matches"][0]["subject_id"], "s9")
        self.assertEqual(full[ACTION_IMPERSONATION]["matched_subject_id"], "s9")

    def test_stored_payload_is_not_modified_by_stripping(self):
        self._nodes("first: 10")
        stored = BiometricAuditEvent.objects.get(action=ACTION_IMPERSONATION)
        self.assertEqual(stored.payload["matched_subject_id"], "s9")


class TestAlertsConnection(_AuditSchemaTestCase):

    def setUp(self):
        super().setUp()
        self.event = record_event(ACTION_IMPERSONATION, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1",
                                  payload={"matched_subject_id": "s9"})
        self._alert("FAILED_VERIFICATIONS", "k1", state="NEW", severity="MEDIUM", subject_id="s1")
        self._alert("FAILED_VERIFICATIONS", "k2", state="ACKNOWLEDGED", severity="MEDIUM", subject_id="s2")
        self._alert("ACCESS_BURST", "k3", state="RESOLVED", severity="HIGH")
        self._alert("IMPERSONATION_SUSPECTED", "k4", state="NEW", severity="HIGH", subject_id="s1", detail={
            "verification_id": "v1", "modality": "face", "matched_subject_model": SUBJECT_MODEL,
            "matched_subject_id": "s9", "matched_template_id": "t9", "matched_score": 0.8,
            "claimed_score": 0.1, "threshold": 0.62, "margin": None,
        })
        self.auditor = _User(perms=AUDIT_PERMS)

    def _alert(self, rule_kind, key, **kwargs):
        self._at(len(key) + BiometricAlert.objects.count())
        return BiometricAlert.objects.create(
            rule_kind=rule_kind, dedupe_key=key, title=f"{rule_kind} {key}", trigger_event=self.event,
            triggered_at=self.now, last_seen_at=self.now, **kwargs,
        )

    def _kinds(self, args, user=None):
        result = _execute(ALERTS_QUERY % args, user or self.auditor)
        self.assertIsNone(result.errors, result.errors)
        return [(e["node"]["ruleKind"], e["node"]["state"]) for e in result.data["biometricAlerts"]["edges"]]

    def test_filters(self):
        self.assertEqual(len(self._kinds("first: 10")), 4)
        self.assertEqual(sorted(self._kinds("open: true, first: 10")), sorted([
            ("FAILED_VERIFICATIONS", "NEW"), ("FAILED_VERIFICATIONS", "ACKNOWLEDGED"), ("IMPERSONATION_SUSPECTED", "NEW"),
        ]))
        self.assertEqual(self._kinds('state: "RESOLVED", first: 10'), [("ACCESS_BURST", "RESOLVED")])
        self.assertEqual(len(self._kinds('ruleKind: "FAILED_VERIFICATIONS", first: 10')), 2)
        self.assertEqual(len(self._kinds('severity: "HIGH", first: 10')), 2)
        self.assertEqual(len(self._kinds('subjectId: "s1", first: 10')), 2)

    def test_newest_first_and_trigger_event_id(self):
        result = _execute(ALERTS_QUERY % "first: 10", self.auditor)
        nodes = [e["node"] for e in result.data["biometricAlerts"]["edges"]]
        self.assertEqual(nodes[0]["ruleKind"], "IMPERSONATION_SUSPECTED")
        self.assertEqual(nodes[0]["triggerEventId"], str(self.event.id))

    def test_impersonation_detail_needs_the_identify_right(self):
        import json

        def detail(user):
            result = _execute(ALERTS_QUERY % 'ruleKind: "IMPERSONATION_SUSPECTED", first: 1', user)
            return json.loads(result.data["biometricAlerts"]["edges"][0]["node"]["detail"])

        stripped = detail(self.auditor)
        self.assertNotIn("matched_subject_id", stripped)
        self.assertNotIn("matched_template_id", stripped)
        self.assertEqual(stripped["matched_score"], 0.8)
        self.assertEqual(detail(_User(perms=AUDIT_PERMS + IDENTIFY_PERMS))["matched_subject_id"], "s9")

    def test_dedupe_key_is_not_queryable(self):
        result = _execute("query { biometricAlerts(first: 1) { edges { node { dedupeKey } } } }", self.auditor)
        self.assertTrue(result.errors)


ACK = 'mutation { acknowledgeBiometricAlert(id: "%s") { state acknowledgedBy } }'
RESOLVE = 'mutation { resolveBiometricAlert(id: "%s", note: "%s") { state resolvedBy resolutionNote } }'


class TestTriageMutations(_AuditSchemaTestCase):

    def setUp(self):
        super().setUp()
        event = record_event(ACTION_VERIFY, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1",
                             payload={"verified": False})
        self.alert = BiometricAlert.objects.create(
            rule_kind="FAILED_VERIFICATIONS", severity="MEDIUM", title="t", dedupe_key="k",
            subject_model=SUBJECT_MODEL, subject_id="s1", trigger_event=event,
        )
        self.supervisor = _User(perms=ALERT_PERMS, username="supervisor")

    def test_denied_without_the_alert_right(self):
        for user in (_User(anonymous=True), _User(perms=AUDIT_PERMS)):
            for mutation in (ACK % self.alert.id, RESOLVE % (self.alert.id, "n")):
                result = _execute(mutation, user)
                self.assertIsInstance(_original(result)[0], PermissionDenied)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.state, "NEW")

    def test_acknowledge_delegates_with_the_username(self):
        with patch("biometric.audit_rules.acknowledge_alert", wraps=audit_rules.acknowledge_alert) as spy:
            result = _execute(ACK % self.alert.id, self.supervisor)
        self.assertIsNone(result.errors, result.errors)
        spy.assert_called_once_with(str(self.alert.id), actor="supervisor")
        self.assertEqual(result.data["acknowledgeBiometricAlert"], {"state": "ACKNOWLEDGED", "acknowledgedBy": "supervisor"})
        self.assertEqual(BiometricAuditEvent.objects.get(action=ACTION_ALERT_ACK).actor, "supervisor")

    def test_resolve_delegates_and_invalid_transition_is_an_error(self):
        with patch("biometric.audit_rules.resolve_alert", wraps=audit_rules.resolve_alert) as spy:
            result = _execute(RESOLVE % (self.alert.id, "false positive"), self.supervisor)
        self.assertIsNone(result.errors, result.errors)
        spy.assert_called_once_with(str(self.alert.id), actor="supervisor", note="false positive")
        self.assertEqual(result.data["resolveBiometricAlert"], {
            "state": "RESOLVED", "resolvedBy": "supervisor", "resolutionNote": "false positive",
        })

        again = _execute(RESOLVE % (self.alert.id, "x"), self.supervisor)
        self.assertIsInstance(_original(again)[0], ValidationError)
        ack = _execute(ACK % self.alert.id, self.supervisor)
        self.assertIsInstance(_original(ack)[0], ValidationError)


class TestIdentifyForwardsActor(SimpleTestCase):

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.identify")
    def test_actor_is_the_username(self, mock_identify, mock_cfg):
        mock_cfg.gql_biometric_identify_perms = []
        mock_identify.return_value = []
        info = MagicMock()
        info.context.user = _User(username="investigator")

        Query.resolve_identify_biometric(None, info, modality="face", sample=base64.b64encode(b"p").decode())

        self.assertEqual(mock_identify.call_args.kwargs["actor"], "investigator")


class TestTemplatesListEvent(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        self.row = BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="s1", modality="face", kind="embedding",
            vector=[0.1, 0.2], provider="fake_embedding", model_name="",
        )
        self._read_perms = BiometricConfig.gql_biometric_read_perms
        self.addCleanup(setattr, BiometricConfig, "gql_biometric_read_perms", self._read_perms)
        BiometricConfig.gql_biometric_read_perms = []
        self.info = MagicMock()
        self.info.context.user = _User(username="reader")

    def test_records_a_list_event_when_enabled(self):
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        result = Query.resolve_biometric_templates(None, self.info, subject_id="s1")

        self.assertEqual(len(result), 1)
        event = BiometricAuditEvent.objects.get(action=ACTION_TEMPLATE_LIST)
        self.assertEqual((event.actor, event.subject_model, event.subject_id), ("reader", SUBJECT_MODEL, "s1"))
        self.assertEqual(event.payload, {"template_ids": [str(self.row.id)]})

    def test_records_nothing_when_disabled(self):
        result = Query.resolve_biometric_templates(None, self.info, subject_id="s1")
        self.assertEqual(len(result), 1)
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)
