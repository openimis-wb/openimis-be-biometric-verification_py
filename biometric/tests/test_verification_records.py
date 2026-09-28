"""
Admin read surface (docs/wb-biometric-dedup-seam.md §6.12, §6.14):
biometricVerificationRecords, a paginated and filtered connection over the
BiometricVerification rows under the verification-records right (174004),
and biometricErasureFilterValues, the distinct erasedBy / subjectModel values
of the erasure tombstones under the audit right (174005).
"""

import datetime
import json
from types import SimpleNamespace
from unittest.mock import patch

import graphene
from django.core.exceptions import PermissionDenied
from django.test import TestCase
from graphql_relay import to_global_id

from biometric.apps import BiometricConfig
from biometric.models import BiometricErasure, BiometricVerification
from biometric.schema import Mutation, Query
from biometric.tests.test_admin_schema import _User

READ_PERMS = ["174004"]
AUDIT_PERMS = ["174005"]
IDENTIFY_PERMS = ["174003"]
SETTINGS = ("gql_biometric_read_perms", "gql_biometric_audit_perms", "gql_biometric_identify_perms")


class _Root(Query, graphene.ObjectType):
    node = graphene.relay.Node.Field()


def _execute(query, user):
    return graphene.Schema(query=_Root, mutation=Mutation).execute(
        query, context_value=SimpleNamespace(user=user, headers={}),
    )


def _denied(result):
    return bool(result.errors) and isinstance(getattr(result.errors[0], "original_error", None), PermissionDenied)


class _RightsMixin:

    def setUp(self):
        for name in SETTINGS:
            self.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))
        super().setUp()
        BiometricConfig.gql_biometric_read_perms = list(READ_PERMS)
        BiometricConfig.gql_biometric_audit_perms = list(AUDIT_PERMS)
        BiometricConfig.gql_biometric_identify_perms = list(IDENTIFY_PERMS)


RECORDS_QUERY = """
query {
  biometricVerificationRecords(%s) {
    totalCount
    pageInfo { hasNextPage }
    edges { node {
      id subjectModel subjectId modality score threshold verified origin fallback deviceId actor createdAt
      riskProfile impersonationSkipReason templateSkipReason
      impersonation { status suspected threshold margin topK claimedScore matchedSubjectModel matchedSubjectId
                      matchedScore candidates { subjectId score suspect } error latencyMs }
    } }
  }
}
"""

BASE = datetime.datetime(2026, 9, 1, 12, 0, 0)


def _verification(subject_id, modality, day, **fields):
    row = BiometricVerification.objects.create(
        subject_model=fields.pop("subject_model", "individual.Individual"), subject_id=subject_id, modality=modality,
        score=fields.pop("score", 0.9), threshold=0.68, verified=fields.pop("verified", True),
        origin=fields.pop("origin", "server"), actor="agent", context={"site": "SENTINEL-CONTEXT"}, **fields,
    )
    BiometricVerification.objects.filter(id=row.id).update(created_at=BASE + datetime.timedelta(days=day))
    return row


SUSPECTED = {
    "impersonation_status": "ok",
    "impersonation_suspected": True,
    "impersonation_subject_model": "individual.Individual",
    "impersonation_subject_id": "bob",
    "impersonation_score": 0.97,
    "impersonation_evidence": {
        "threshold": 0.62, "margin": None, "top_k": 7, "claimed_score": 0.4,
        "candidates": [{"subject_model": "individual.Individual", "subject_id": "bob", "template_id": "t-bob",
                        "score": 0.97, "suspect": True}],
        "error": "", "latency_ms": 3.5,
    },
}


class TestVerificationRecords(_RightsMixin, TestCase):

    def setUp(self):
        super().setUp()
        _verification("alice", "face", 0, **SUSPECTED)
        _verification("alice", "fingerprint", 1, origin="device", verified=False, score=10.0,
                      impersonation_skip_reason="device_path_disabled")
        _verification("carol", "face", 2, subject_model="social_protection.Beneficiary",
                      template_skip_reason="preprocessing_mismatch", score=None, verified=False)
        self.reader = _User(perms=READ_PERMS)

    def _nodes(self, args, user=None):
        result = _execute(RECORDS_QUERY % args, user or self.reader)
        self.assertIsNone(result.errors, result.errors)
        return result.data["biometricVerificationRecords"]

    def test_rights(self):
        for user in (_User(anonymous=True), _User(perms=["174002", "174003", "174005", "174007"])):
            with self.subTest(perms=user.perms):
                self.assertTrue(_denied(_execute(RECORDS_QUERY % "first: 5", user)))

    def test_newest_first_paginated_with_the_probe_evidence(self):
        data = self._nodes("first: 2")

        self.assertEqual(data["totalCount"], 3)
        self.assertTrue(data["pageInfo"]["hasNextPage"])
        nodes = [edge["node"] for edge in data["edges"]]
        self.assertEqual(
            [(n["subjectId"], n["modality"]) for n in nodes], [("carol", "face"), ("alice", "fingerprint")],
        )
        self.assertEqual(nodes[0]["templateSkipReason"], "preprocessing_mismatch")
        self.assertIsNone(nodes[0]["score"])
        self.assertEqual(nodes[1]["impersonationSkipReason"], "device_path_disabled")
        self.assertIsNone(nodes[1]["impersonation"])

        last = self._nodes('first: 1, subjectId: "alice", modality: "face"')["edges"][0]["node"]
        self.assertEqual(last["impersonation"]["topK"], 7)
        self.assertEqual(last["impersonation"]["status"], "ok")
        self.assertTrue(last["impersonation"]["suspected"])
        self.assertEqual((last["impersonation"]["claimedScore"], last["impersonation"]["matchedScore"]), (0.4, 0.97))
        self.assertEqual((last["impersonationSkipReason"], last["templateSkipReason"]), ("", ""))

    def test_other_identities_need_the_identify_right(self):
        without = self._nodes('first: 1, suspected: true')["edges"][0]["node"]["impersonation"]
        self.assertIsNone(without["matchedSubjectId"])
        self.assertIsNone(without["matchedSubjectModel"])
        self.assertEqual(without["candidates"], [])
        self.assertNotIn("bob", json.dumps(self._nodes("first: 10")))

        with_identify = self._nodes('first: 1, suspected: true', _User(perms=READ_PERMS + IDENTIFY_PERMS))
        impersonation = with_identify["edges"][0]["node"]["impersonation"]
        self.assertEqual(impersonation["matchedSubjectId"], "bob")
        self.assertEqual(impersonation["candidates"], [{"subjectId": "bob", "score": 0.97, "suspect": True}])

    def test_filters(self):
        cases = {
            'subjectId: "alice"': [("alice", "fingerprint"), ("alice", "face")],
            'subjectModel: "social_protection.Beneficiary"': [("carol", "face")],
            'modality: "face"': [("carol", "face"), ("alice", "face")],
            "suspected: true": [("alice", "face")],
            "suspected: false": [("carol", "face"), ("alice", "fingerprint")],
            'createdAt_Gte: "2026-09-02T00:00:00"': [("carol", "face"), ("alice", "fingerprint")],
            'createdAt_Lte: "2026-09-02T00:00:00"': [("alice", "face")],
            'subjectId: "alice", modality: "face", suspected: false': [],
        }
        for args, expected in cases.items():
            with self.subTest(args):
                data = self._nodes(args + ", first: 10")
                self.assertEqual([(e["node"]["subjectId"], e["node"]["modality"]) for e in data["edges"]], expected)

    def test_raw_columns_are_not_exposed(self):
        for field in ("context", "impersonationEvidence", "impersonationSubjectId", "impersonationSubjectModel"):
            with self.subTest(field):
                result = _execute("query { biometricVerificationRecords(first: 1) { edges { node { %s } } } }" % field,
                                  self.reader)
                self.assertTrue(result.errors)
                self.assertIn("Cannot query field", str(result.errors[0]))

    def test_node_lookup_needs_the_read_right(self):
        row = BiometricVerification.objects.get(subject_id="carol")
        query = 'query { node(id: "%s") { ... on BiometricVerificationGQLType { subjectId } } }' % (
            to_global_id("BiometricVerificationGQLType", row.id)
        )

        denied = _execute(query, _User(perms=["174005", "174003", "174002"]))
        allowed = _execute(query, self.reader)

        self.assertIsNone((denied.data or {}).get("node"))
        self.assertTrue(_denied(denied))
        self.assertIsNone(allowed.errors, allowed.errors)
        self.assertEqual(allowed.data["node"]["subjectId"], "carol")


FILTER_VALUES_QUERY = "query { biometricErasureFilterValues { erasedBy subjectModel } }"


class TestErasureFilterValues(_RightsMixin, TestCase):

    def _erasure(self, subject_model, erased_by):
        BiometricErasure.objects.create(
            subject_model=subject_model, subject_id="s", modalities=["face"], erased={"face": 1},
            reason="retention", erased_by=erased_by,
        )

    def _values(self):
        result = _execute(FILTER_VALUES_QUERY, _User(perms=AUDIT_PERMS))
        self.assertIsNone(result.errors, result.errors)
        return result.data["biometricErasureFilterValues"]

    def test_rights(self):
        for user in (_User(anonymous=True), _User(perms=["174004", "174007", "174003"])):
            with self.subTest(perms=user.perms):
                self.assertTrue(_denied(_execute(FILTER_VALUES_QUERY, user)))

    def test_empty(self):
        self.assertEqual(self._values(), {"erasedBy": [], "subjectModel": []})

    def test_distinct_and_sorted(self):
        for subject_model, erased_by in (
            ("individual.Individual", "retention"), ("individual.Individual", "admin"),
            ("social_protection.Beneficiary", "retention"), ("individual.Individual", "zed"),
        ):
            self._erasure(subject_model, erased_by)

        self.assertEqual(self._values(), {
            "erasedBy": ["admin", "retention", "zed"],
            "subjectModel": ["individual.Individual", "social_protection.Beneficiary"],
        })

    def test_capped(self):
        for index in range(4):
            self._erasure(f"model.M{index}", f"user{index}")

        with patch("biometric.schema.ERASURE_FILTER_VALUES_LIMIT", 2):
            values = self._values()

        self.assertEqual(values, {"erasedBy": ["user0", "user1"], "subjectModel": ["model.M0", "model.M1"]})
