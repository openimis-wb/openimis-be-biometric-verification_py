"""
The fused decision of verify_multimodal() (docs/wb-biometric-dedup-seam.md §6.11):
stored as a BiometricMultimodalDecision row, recorded as a verify.multimodal
audit event when audit is on, returned by verifyBiometricMultimodal as
decisionId and read back through biometricMultimodalDecisions (the
verification-records right, 174004).
"""

import json
from types import SimpleNamespace

import graphene
from django.core.exceptions import PermissionDenied
from graphql_relay import to_global_id

from biometric.apps import BiometricConfig
from biometric.audit_chain import ACTION_VERIFY, ACTION_VERIFY_MULTIMODAL
from biometric.models import BiometricAuditEvent, BiometricMultimodalDecision, BiometricVerification
from biometric.schema import Mutation, Query
from biometric.services import verify_multimodal
from biometric.tests.test_admin_schema import _User
from biometric.tests.test_audit_chain import restore_audit_settings_on_cleanup
from biometric.tests.test_impersonation import VECTORS, _ImpersonationTestCase
from biometric.tests.test_multimodal_verify import FUSION, _legs, _MultimodalTestCase
from biometric.tests.test_services import SUBJECT_MODEL

READ_PERMS = ["174004"]
AUDIT_PERMS = ["174005"]

PAYLOAD_KEYS = {
    "decision_id", "outcome", "score", "reasons", "risk_profile", "modalities", "verification_ids", "fallback",
    "device_id",
}


class _Root(Query, graphene.ObjectType):
    node = graphene.relay.Node.Field()


def _execute(query, user):
    return graphene.Schema(query=_Root, mutation=Mutation).execute(
        query, context_value=SimpleNamespace(user=user, headers={}),
    )


def _denied(result):
    return bool(result.errors) and isinstance(getattr(result.errors[0], "original_error", None), PermissionDenied)


class _DecisionTestCase(_MultimodalTestCase):

    def setUp(self):
        super().setUp()
        restore_audit_settings_on_cleanup(self)
        self.addCleanup(setattr, BiometricConfig, "gql_biometric_read_perms", BiometricConfig.gql_biometric_read_perms)
        self.addCleanup(
            setattr, BiometricConfig, "gql_biometric_verify_perms", BiometricConfig.gql_biometric_verify_perms,
        )
        BiometricConfig.gql_biometric_read_perms = list(READ_PERMS)
        BiometricConfig.gql_biometric_audit_perms = list(AUDIT_PERMS)
        BiometricConfig.gql_biometric_verify_perms = ["174002"]


class TestDecisionIsStored(_DecisionTestCase):

    def test_the_fused_decision_and_the_leg_rows(self):
        result = verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 15.0), actor="agent", device_id="tab-1",
                                   fallback=True)

        self.assertEqual(BiometricMultimodalDecision.objects.count(), 1)
        row = BiometricMultimodalDecision.objects.get()
        legs = {r.modality: str(r.id) for r in BiometricVerification.objects.filter(subject_id="s1")}
        self.assertEqual(result.decision_id, str(row.id))
        self.assertEqual([leg.verification_id for leg in result.legs], [legs["face"], legs["fingerprint"]])
        self.assertEqual(row.verification_ids, [legs["face"], legs["fingerprint"]])
        self.assertEqual((row.subject_model, row.subject_id), (SUBJECT_MODEL, "s1"))
        self.assertEqual(row.outcome, result.decision.outcome)
        self.assertEqual(row.outcome, "review")
        self.assertAlmostEqual(row.score, result.decision.score)
        self.assertEqual(row.reasons, ["'fingerprint' score 15.0 below floor 20.0"])
        self.assertEqual(row.modalities, ["face", "fingerprint"])
        self.assertEqual((row.risk_profile, row.actor, row.device_id, row.fallback), ("", "agent", "tab-1", True))

    def test_the_profile_name_is_stored(self):
        BiometricConfig.risk_profiles = {"strict": {"floor_decision": "reject"}}

        result = verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 15.0), actor="agent", risk_profile="strict")

        self.assertEqual(BiometricMultimodalDecision.objects.count(), 1)
        row = BiometricMultimodalDecision.objects.get(id=result.decision_id)
        self.assertEqual((row.outcome, row.risk_profile), ("reject", "strict"))

    def test_a_missing_leg_score_is_stored_as_the_decision_saw_it(self):
        BiometricConfig.risk_profiles = {"two": {"required": ["fingerprint"]}}

        result = verify_multimodal(SUBJECT_MODEL, "s1", _legs(face=0.5), actor="agent", risk_profile="two")

        self.assertEqual(BiometricMultimodalDecision.objects.count(), 1)
        row = BiometricMultimodalDecision.objects.get(id=result.decision_id)
        self.assertEqual(row.outcome, "review")
        self.assertIn("required modality 'fingerprint' has no score", row.reasons)
        self.assertEqual(row.modalities, ["face"])

    def test_refused_legs_or_profile_store_no_decision(self):
        BiometricConfig.risk_profiles = {"loose": {"thresholds": {"accept": 0.1}}}
        for legs, profile in (([], None), (_legs(0.5), "nope"), (_legs(0.5), "loose")):
            with self.subTest(legs=legs, profile=profile):
                with self.assertRaises(Exception):
                    verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent", risk_profile=profile)
        self.assertEqual(BiometricMultimodalDecision.objects.count(), 0)

    def test_a_leg_failing_after_the_checks_stores_no_decision(self):
        from biometric.registry import ProviderRegistry

        provider = ProviderRegistry.get_provider("fingerprint")
        original = provider.extract
        self.addCleanup(setattr, provider, "extract", original)

        def refuse(sample, position=None):
            raise ValueError("no finger")

        provider.extract = refuse
        with self.assertRaises(ValueError):
            verify_multimodal(
                SUBJECT_MODEL, "s1", [{"modality": "face", "device_score": 0.5}, {"modality": "fingerprint",
                                                                                 "sample": b"x"}],
                actor="agent",
            )
        self.assertEqual(BiometricVerification.objects.count(), 1)
        self.assertEqual(BiometricMultimodalDecision.objects.count(), 0)

    def test_stored_with_audit_off_and_no_event(self):
        BiometricConfig.audit = {"enabled": False, "rules": {}}

        verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 50.0), actor="agent")

        self.assertEqual(BiometricMultimodalDecision.objects.count(), 1)
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)


class TestDecisionAuditEvent(_DecisionTestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = {"enabled": True, "rules": {}}

    def test_one_event_after_the_leg_events(self):
        result = verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 15.0), actor="agent", device_id="tab-1")

        events = list(BiometricAuditEvent.objects.order_by("sequence"))
        self.assertEqual([e.action for e in events], [ACTION_VERIFY, ACTION_VERIFY, ACTION_VERIFY_MULTIMODAL])
        event = events[-1]
        self.assertEqual((event.actor, event.subject_model, event.subject_id, event.modality),
                         ("agent", SUBJECT_MODEL, "s1", ""))
        self.assertEqual(set(event.payload), PAYLOAD_KEYS)
        self.assertEqual(event.payload["decision_id"], result.decision_id)
        self.assertEqual(event.payload["verification_ids"], [leg.verification_id for leg in result.legs])
        self.assertEqual(event.payload["outcome"], "review")
        self.assertEqual(event.payload["reasons"], ["'fingerprint' score 15.0 below floor 20.0"])
        self.assertEqual(event.payload["modalities"], ["face", "fingerprint"])
        self.assertEqual((event.payload["device_id"], event.payload["fallback"]), ("tab-1", False))
        self.assertAlmostEqual(event.payload["score"], result.decision.score)


MUTATION = """
mutation {
  verifyBiometricMultimodal(subjectId: "s1", legs: [
    {modality: "face", deviceScore: 0.55},
    {modality: "fingerprint", deviceScore: 40}
  ]) { outcome decisionId legs { modality verificationId } }
}
"""

DECISIONS_QUERY = """
query {
  biometricMultimodalDecisions(%s) {
    totalCount
    pageInfo { hasNextPage }
    edges { node {
      id subjectModel subjectId outcome score reasons riskProfile modalities verificationIds fallback deviceId
      actor createdAt
    } }
  }
}
"""


class TestDecisionOverGraphQL(_DecisionTestCase):

    def test_the_mutation_returns_the_decision_and_leg_ids(self):
        result = _execute(MUTATION, _User(perms=["174002"], username="agent"))

        self.assertIsNone(result.errors, result.errors)
        data = result.data["verifyBiometricMultimodal"]
        row = BiometricMultimodalDecision.objects.get()
        self.assertEqual(data["decisionId"], str(row.id))
        self.assertEqual([leg["verificationId"] for leg in data["legs"]], row.verification_ids)

    def test_rights(self):
        verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 50.0), actor="agent")
        for user in (_User(anonymous=True), _User(perms=["174002", "174003", "174005", "174007"])):
            with self.subTest(perms=user.perms):
                self.assertTrue(_denied(_execute(DECISIONS_QUERY % "first: 5", user)))

    def test_newest_first_filtered_and_paginated(self):
        import datetime

        BiometricConfig.risk_profiles = {"strict": {"floor_decision": "reject"}}
        base = datetime.datetime(2026, 9, 1, 12, 0, 0)
        cases = [("s1", _legs(0.5, 50.0), None), ("s2", _legs(0.5, 15.0), None), ("s1", _legs(0.5, 15.0), "strict")]
        for index, (subject, legs, profile) in enumerate(cases):
            result = verify_multimodal(SUBJECT_MODEL, subject, legs, actor="agent", risk_profile=profile)
            BiometricMultimodalDecision.objects.filter(id=result.decision_id).update(
                created_at=base + datetime.timedelta(days=index),
            )
        reader = _User(perms=READ_PERMS)

        page = _execute(DECISIONS_QUERY % "first: 2", reader)
        self.assertIsNone(page.errors, page.errors)
        data = page.data["biometricMultimodalDecisions"]
        self.assertEqual(data["totalCount"], 3)
        self.assertTrue(data["pageInfo"]["hasNextPage"])
        nodes = [edge["node"] for edge in data["edges"]]
        self.assertEqual([(n["subjectId"], n["outcome"]) for n in nodes], [("s1", "reject"), ("s2", "review")])
        self.assertEqual(nodes[0]["riskProfile"], "strict")
        self.assertEqual(nodes[0]["modalities"], ["face", "fingerprint"])
        self.assertEqual(len(nodes[0]["verificationIds"]), 2)
        self.assertEqual(nodes[1]["reasons"], ["'fingerprint' score 15.0 below floor 20.0"])

        filters = {
            'subjectId: "s1"': ["reject", "accept"],
            'subjectModel: "individual.Individual", outcome: "review"': ["review"],
            'riskProfile: "strict"': ["reject"],
            'createdAt_Gte: "2026-09-02T00:00:00"': ["reject", "review"],
            'createdAt_Lte: "2026-09-02T00:00:00"': ["accept"],
        }
        for args, expected in filters.items():
            with self.subTest(args):
                result = _execute(DECISIONS_QUERY % (args + ", first: 10"), reader)
                self.assertIsNone(result.errors, result.errors)
                self.assertEqual(
                    [e["node"]["outcome"] for e in result.data["biometricMultimodalDecisions"]["edges"]], expected,
                )

    def test_node_lookup_needs_the_read_right(self):
        result = verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 50.0), actor="agent")
        query = 'query { node(id: "%s") { ... on BiometricMultimodalDecisionGQLType { outcome } } }' % (
            to_global_id("BiometricMultimodalDecisionGQLType", result.decision_id)
        )

        denied = _execute(query, _User(perms=["174005", "174003", "174002"]))
        allowed = _execute(query, _User(perms=READ_PERMS))

        self.assertIsNone((denied.data or {}).get("node"))
        self.assertTrue(_denied(denied))
        self.assertIsNone(allowed.errors, allowed.errors)
        self.assertEqual(allowed.data["node"]["outcome"], "accept")


class TestNoOtherIdentityInTheDecision(_ImpersonationTestCase):
    """A suspected leg names another subject on its own row and event; the decision never does."""

    def setUp(self):
        super().setUp()
        restore_audit_settings_on_cleanup(self)
        self.addCleanup(setattr, BiometricConfig, "gql_biometric_read_perms", BiometricConfig.gql_biometric_read_perms)
        BiometricConfig.gql_biometric_read_perms = list(READ_PERMS)
        BiometricConfig.gql_biometric_audit_perms = list(AUDIT_PERMS)
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        BiometricConfig.fusion = {**FUSION, "weights": {"face": 1.0}, "floors": {}}
        self._face("alice", VECTORS[b"alice"])
        self.bob = self._face("bob", VECTORS[b"bob"])
        self._enable()

    def test_the_decision_and_its_event_hold_no_matched_identity(self):
        result = verify_multimodal(SUBJECT_MODEL, "alice", [{"modality": "face", "sample": b"bob-probe"}],
                                   actor="agent")
        self.assertTrue(result.legs[0].impersonation.suspected)
        reader = _User(perms=READ_PERMS + AUDIT_PERMS)

        decisions = _execute(DECISIONS_QUERY % "first: 5", reader)
        events = _execute(
            'query { biometricAuditEvents(action: "%s") { edges { node { subjectId payload } } } }'
            % ACTION_VERIFY_MULTIMODAL, reader,
        )

        self.assertIsNone(decisions.errors, decisions.errors)
        self.assertIsNone(events.errors, events.errors)
        dumped = json.dumps([decisions.data, events.data])
        self.assertEqual(len(events.data["biometricAuditEvents"]["edges"]), 1)
        self.assertEqual(events.data["biometricAuditEvents"]["edges"][0]["node"]["subjectId"], "alice")
        for identity in ("bob", str(self.bob.id)):
            self.assertNotIn(identity, dumped)
        stored = BiometricAuditEvent.objects.get(action=ACTION_VERIFY_MULTIMODAL)
        self.assertNotIn("bob", json.dumps(stored.payload))
