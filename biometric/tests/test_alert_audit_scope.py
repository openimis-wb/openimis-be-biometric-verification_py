"""
biometricAlerts, biometricAuditEvents, their node lookups and the alert triage
mutations act only on rows whose subject lies in the caller's location scope
(docs/wb-biometric-dedup-seam.md §6.10, §6.15). A row naming no subject stays
visible. The audit chain verification, its status and the chain head are not
scoped.
"""

from types import SimpleNamespace

import graphene
from django.contrib.auth.models import AnonymousUser
from django.core.cache import cache
from django.core.exceptions import PermissionDenied
from django.db import connection
from graphql_relay import to_global_id

from biometric.apps import BiometricConfig
from biometric.audit_chain import ACTION_ALERT_ACK, ACTION_IDENTIFY, ACTION_VERIFY, record_event
from biometric.models import BiometricAlert, BiometricAuditEvent
from biometric.schema import Mutation, Query
from biometric.subjects import SUBJECT_NOT_FOUND, SubjectRefusedError
from biometric.tests.test_audit_chain import restore_audit_settings_on_cleanup
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase
from biometric.tests.test_subject_scope import _individual, _role

AUDIT, ALERT, VERIFY = 174005, 174006, 174008
EVENT_TYPE = "BiometricAuditEventGQLType"
ALERT_TYPE = "BiometricAlertGQLType"


class _Root(Query, graphene.ObjectType):
    node = graphene.relay.Node.Field()


class TestAlertAndAuditScope(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        from core.test_helpers import create_admin_role, create_test_interactive_user
        from location.test_helpers import assign_user_districts, create_test_village

        restore_audit_settings_on_cleanup(self)
        for name in ("gql_biometric_audit_perms", "gql_biometric_alert_perms", "gql_biometric_audit_verify_perms"):
            self.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))
        BiometricConfig.gql_biometric_audit_perms = [str(AUDIT)]
        BiometricConfig.gql_biometric_alert_perms = [str(ALERT)]
        BiometricConfig.gql_biometric_audit_verify_perms = [str(VERIFY)]
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        cache.clear()
        self.addCleanup(cache.clear)

        village_a = create_test_village({"name": "Audit Village A", "code": "AudViA"})
        village_b = create_test_village({"name": "Audit Village B", "code": "AudViB"})
        self.admin = create_test_interactive_user(username="auditScopeAdmin", roles=[create_admin_role().id])
        self.agent = create_test_interactive_user(
            username="auditScopeAgentA", roles=[_role("Audit agent", (AUDIT, ALERT, VERIFY)).id],
        )
        self.triager = create_test_interactive_user(
            username="auditScopeTriagerA", roles=[_role("Alert triager", (ALERT,)).id],
        )
        district = village_a.parent.parent.code
        assign_user_districts(self.agent, [district])
        assign_user_districts(self.triager, [district])
        cache.clear()
        self.inside = _individual(self.admin, village_a)
        self.outside = _individual(self.admin, village_b)

        self.events = {
            "inside": self._event(ACTION_VERIFY, self.inside),
            "outside": self._event(ACTION_VERIFY, self.outside),
            "none": self._event(ACTION_IDENTIFY, ""),
        }
        self.alerts = {
            name: BiometricAlert.objects.create(
                rule_kind="FAILED_VERIFICATIONS", severity="MEDIUM", title=name, dedupe_key=name,
                subject_model=SUBJECT_MODEL if subject_id else "", subject_id=subject_id,
                trigger_event=self.events[name],
            )
            for name, subject_id in (("inside", self.inside), ("outside", self.outside), ("none", ""))
        }

    def _event(self, action, subject_id):
        return record_event(
            action, actor="agent", subject_model=SUBJECT_MODEL if subject_id else "", subject_id=subject_id,
            payload={"verified": False},
        )

    def _execute(self, query, user):
        return graphene.Schema(query=_Root, mutation=Mutation).execute(
            query, context_value=SimpleNamespace(user=user, headers={}),
        )

    def _titles(self, user):
        result = self._execute("query { biometricAlerts(first: 10) { edges { node { title } } } }", user)
        self.assertIsNone(result.errors, result.errors)
        return sorted(e["node"]["title"] for e in result.data["biometricAlerts"]["edges"])

    def _sequences(self, user):
        result = self._execute("query { biometricAuditEvents(first: 10) { edges { node { sequence } } } }", user)
        self.assertIsNone(result.errors, result.errors)
        return sorted(e["node"]["sequence"] for e in result.data["biometricAuditEvents"]["edges"])

    def test_alerts_of_an_out_of_scope_subject_are_hidden(self):
        self.assertEqual(self._titles(self.agent), ["inside", "none"])
        self.assertEqual(self._titles(self.admin), ["inside", "none", "outside"])

    def test_audit_events_of_an_out_of_scope_subject_are_hidden(self):
        self.assertEqual(
            self._sequences(self.agent), sorted(self.events[n].sequence for n in ("inside", "none")),
        )
        self.assertEqual(self._sequences(self.admin), sorted(e.sequence for e in self.events.values()))

    def test_the_total_count_counts_only_visible_rows(self):
        for field, expected in (("biometricAlerts", 2), ("biometricAuditEvents", 2)):
            with self.subTest(field):
                result = self._execute("query { %s(first: 1) { totalCount } }" % field, self.agent)
                self.assertIsNone(result.errors, result.errors)
                self.assertEqual(result.data[field]["totalCount"], expected)

    def test_filtering_on_an_out_of_scope_subject_returns_nothing(self):
        query = 'query { %s(subjectId: "%s", first: 10) { totalCount } }'
        for field in ("biometricAlerts", "biometricAuditEvents"):
            with self.subTest(field):
                self.assertEqual(self._execute(query % (field, self.outside), self.agent).data[field]["totalCount"], 0)
                self.assertEqual(self._execute(query % (field, self.outside), self.admin).data[field]["totalCount"], 1)

    def test_node_lookups_hide_out_of_scope_rows(self):
        cases = (
            (ALERT_TYPE, self.alerts, "title"),
            (EVENT_TYPE, self.events, "sequence"),
        )
        for type_name, rows, field in cases:
            for name, visible in (("inside", True), ("none", True), ("outside", False)):
                with self.subTest(type=type_name, row=name):
                    query = 'query { node(id: "%s") { ... on %s { %s } } }' % (
                        to_global_id(type_name, rows[name].id), type_name, field,
                    )
                    self.assertEqual(bool((self._execute(query, self.agent).data or {}).get("node")), visible)
                    self.assertTrue((self._execute(query, self.admin).data or {}).get("node"))

    def _triage(self, mutation, alert, user):
        args = 'id: "%s"' % alert.id
        return self._execute("mutation { %s(%s) { state } }" % (mutation, args), user)

    def _assert_refused(self, result):
        self.assertTrue(result.errors, result.data)
        self.assertFalse(any((result.data or {}).values()), result.data)
        error = getattr(result.errors[0], "original_error", None)
        self.assertIsInstance(error, SubjectRefusedError)
        self.assertEqual(result.errors[0].extensions, {"code": SUBJECT_NOT_FOUND})

    def test_acknowledge_and_resolve_refuse_an_out_of_scope_alert(self):
        events_before = BiometricAuditEvent.objects.count()
        for user in (self.agent, self.triager):
            for mutation in ("acknowledgeBiometricAlert", "resolveBiometricAlert"):
                with self.subTest(user=user.username, mutation=mutation):
                    self._assert_refused(self._triage(mutation, self.alerts["outside"], user))
        self.alerts["outside"].refresh_from_db()
        self.assertEqual(self.alerts["outside"].state, BiometricAlert.STATE_NEW)
        self.assertEqual(BiometricAuditEvent.objects.count(), events_before)

    def test_acknowledge_and_resolve_serve_an_alert_in_scope_or_without_subject(self):
        for name in ("inside", "none"):
            with self.subTest(name):
                ack = self._triage("acknowledgeBiometricAlert", self.alerts[name], self.triager)
                self.assertIsNone(ack.errors, ack.errors)
                self.assertEqual(ack.data["acknowledgeBiometricAlert"]["state"], "ACKNOWLEDGED")
                done = self._triage("resolveBiometricAlert", self.alerts[name], self.triager)
                self.assertIsNone(done.errors, done.errors)
                self.assertEqual(done.data["resolveBiometricAlert"]["state"], "RESOLVED")
        self.assertTrue(BiometricAuditEvent.objects.filter(action=ACTION_ALERT_ACK).exists())

    def test_an_admin_triages_an_alert_of_any_subject(self):
        result = self._triage("acknowledgeBiometricAlert", self.alerts["outside"], self.admin)
        self.assertIsNone(result.errors, result.errors)

    def test_a_missing_alert_is_still_reported_as_missing(self):
        alert = SimpleNamespace(id="00000000-0000-0000-0000-000000000000")
        result = self._triage("acknowledgeBiometricAlert", alert, self.triager)
        self.assertTrue(result.errors)
        self.assertNotIsInstance(getattr(result.errors[0], "original_error", None), SubjectRefusedError)

    def test_the_chain_verification_walks_every_event(self):
        total = BiometricAuditEvent.objects.count()
        self.assertEqual(total, 3)

        result = self._execute("mutation { verifyBiometricAuditChain { ok checked headSequence } }", self.agent)
        self.assertIsNone(result.errors, result.errors)
        check = result.data["verifyBiometricAuditChain"]
        self.assertEqual((check["ok"], check["checked"]), (True, total))
        self.assertEqual(check["headSequence"], max(e.sequence for e in self.events.values()))

        status = self._execute("query { biometricAuditChainStatus { ok checked headSequence } }", self.agent)
        self.assertIsNone(status.errors, status.errors)
        self.assertEqual(status.data["biometricAuditChainStatus"], check)

    def test_the_chain_head_counts_every_event(self):
        query = "query { biometricAuditChainHead { headSequence headHash eventCount createdAt } }"
        head = BiometricAuditEvent.objects.order_by("-sequence").first()
        expected = {
            "headSequence": head.sequence, "headHash": head.hash,
            "eventCount": BiometricAuditEvent.objects.count(),
        }
        self.assertEqual(expected["eventCount"], 3)
        for user in (self.agent, self.admin):
            with self.subTest(user=user.username):
                result = self._execute(query, user)
                self.assertIsNone(result.errors, result.errors)
                data = dict(result.data["biometricAuditChainHead"])
                self.assertTrue(data.pop("createdAt"))
                self.assertEqual(data, expected)

    def test_the_chain_head_needs_the_audit_right(self):
        query = "query { biometricAuditChainHead { headSequence } }"
        for user in (self.triager, AnonymousUser()):
            with self.subTest(user=str(user)):
                result = self._execute(query, user)
                self.assertTrue(result.errors)
                self.assertIsInstance(getattr(result.errors[0], "original_error", None), PermissionDenied)
                self.assertIsNone((result.data or {}).get("biometricAuditChainHead"))

    def test_the_chain_head_is_null_on_an_empty_chain(self):
        BiometricAlert.objects.all().delete()
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM biometric_audit_event")
        result = self._execute("query { biometricAuditChainHead { headSequence } }", self.agent)
        self.assertIsNone(result.errors, result.errors)
        self.assertIsNone(result.data["biometricAuditChainHead"])
