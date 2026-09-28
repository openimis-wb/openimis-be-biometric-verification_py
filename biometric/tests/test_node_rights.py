"""
The relay node field of the assembled openIMIS schema (openIMIS/schema.py
declares node = graphene.relay.Node.Field()) resolves a global id through the
type's get_queryset, not through the module's query resolvers. The audit
event, alert and erasure types check the audit right there too.
"""

from types import SimpleNamespace

import graphene
from django.core.exceptions import PermissionDenied
from django.test import TestCase
from graphql_relay import to_global_id

from biometric.apps import BiometricConfig
from biometric.audit_chain import ACTION_VERIFY, record_event
from biometric.models import BiometricAlert, BiometricErasure
from biometric.schema import Mutation, Query
from biometric.tests.test_audit_chain import AuditConfigMixin


class _Root(Query, graphene.ObjectType):
    node = graphene.relay.Node.Field()


class _User:

    def __init__(self, perms=()):
        self.perms = set(perms)
        self.username = "reader"
        self.is_anonymous = False
        self.is_authenticated = True
        self.id = 1

    def has_perms(self, perms):
        return all(p in self.perms for p in perms)


class TestNodeLookupNeedsTheAuditRight(AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        BiometricConfig.gql_biometric_audit_perms = ["174005"]
        event = record_event(ACTION_VERIFY, actor="agent", subject_model="individual.Individual", subject_id="s1",
                             modality="face", payload={"verified": False})
        alert = BiometricAlert.objects.create(
            rule_kind="FAILED_VERIFICATIONS", severity="MEDIUM", title="t", dedupe_key="k",
            subject_model="individual.Individual", subject_id="s1", trigger_event=event,
        )
        erasure = BiometricErasure.objects.create(
            subject_model="individual.Individual", subject_id="s1", modalities=["face"], erased={"face": 1},
            reason="retention", erased_by="retention",
        )
        self.ids = {
            "BiometricAuditEventGQLType": (to_global_id("BiometricAuditEventGQLType", event.id), "sequence"),
            "BiometricAlertGQLType": (to_global_id("BiometricAlertGQLType", alert.id), "ruleKind"),
            "BiometricErasureGQLType": (to_global_id("BiometricErasureGQLType", erasure.id), "subjectId"),
        }

    def _node(self, type_name, user):
        global_id, field = self.ids[type_name]
        query = 'query { node(id: "%s") { ... on %s { %s } } }' % (global_id, type_name, field)
        schema = graphene.Schema(query=_Root, mutation=Mutation)
        return schema.execute(query, context_value=SimpleNamespace(user=user, headers={}))

    def test_without_the_audit_right_nothing_is_returned(self):
        for type_name in self.ids:
            with self.subTest(type_name):
                result = self._node(type_name, _User(perms=["174004", "174003", "174006", "174007"]))
                self.assertIsNone((result.data or {}).get("node"))
                self.assertTrue(result.errors)
                self.assertIsInstance(getattr(result.errors[0], "original_error", None), PermissionDenied)

    def test_with_the_audit_right_the_node_resolves(self):
        for type_name, (_, field) in self.ids.items():
            with self.subTest(type_name):
                result = self._node(type_name, _User(perms=["174005"]))
                self.assertIsNone(result.errors, result.errors)
                self.assertIsNotNone(result.data["node"][field])
